import os
import datetime
from tqdm import tqdm
from typing import List

from sarathi.config import ModelConfig, ParallelConfig, MetricsConfig, SystemConfig, WorkerConfig, \
    ReplicaConfig, OptSarathiSchedulerConfig, SarathiSchedulerConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.utils.prompt_utils import get_prompts_from_dataset

BASE_OUTPUT_DIR = "./offline_inference_output"

# 采样后端：
# - flashinfer: 更快，但某些环境下可能触发 CUDA illegal memory access
# - torch: 更稳（慢一些），用于稳定跑完/排查问题
os.environ.setdefault("SARATHI_SAMPLING_BACKEND", "torch")

# 是否对超出 `max_model_len` 的 prompt 做截断（left-truncate，保留末尾 tokens）。
# - True: 不丢请求，但会改变 prompt 内容（更贴近“实际服务端截断”行为）
# - False: 超长 prompt 会因为 `max_new_tokens_budget < 1` 被跳过（skipped）

TRUNCATE_OVERLONG_PROMPTS = True
# 为生成预留的最小 token 数（避免 prompt 把上下文窗口全部占满导致 decode 越界）。
# 注意：这只是“预留空间”的下限，真正的每条请求 max_tokens 会在 generate() 中按预算动态裁剪。
MIN_OUTPUT_TOKENS = 1

prompts = get_prompts_from_dataset("dataset/ShareGPT_V3_unfiltered_cleaned_split.json", 200, random_sample=True)


sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

replica_config = ReplicaConfig(
    output_dir=output_dir,
)

model_config = ModelConfig(
    model="Qwen/Qwen3-8B",
    max_model_len=1024,
)

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

scheduler_config = SarathiSchedulerConfig(
    chunk_size=256,
    max_num_seqs=32,
    policy_name="fcfs",
    enable_select_stats_csv=True,
)

metrics_config = MetricsConfig(
    write_metrics=True,
    enable_chrome_trace=True,
)

worker_config = WorkerConfig(
    gpu_memory_utilization=0.7
)

system_config = SystemConfig(
    replica_config=replica_config,
    model_config=model_config,
    parallel_config=parallel_config,
    scheduler_config=scheduler_config,
    metrics_config=metrics_config,
    worker_config=worker_config,
)

llm_engine = LLMEngine.from_system_config(system_config)


def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> List[RequestOutput]:
    """
    离线批量推理入口：
    1) 对每条 prompt 先 tokenize 得到 prompt_len（不能用字符串长度近似 token 数）
    2) 动态裁剪每条请求的 max_tokens，使其满足：prompt_len + max_tokens <= max_model_len
       否则 decode 时 position 会超过模型支持长度，触发 CUDA illegal memory access / gather OOB。
    3) 可选：对超长 prompt 做 left-truncate（保留末尾 tokens），确保仍有输出空间。
    """
    max_model_len = llm_engine.get_model_config().max_model_len
    if max_model_len is None:
        raise ValueError("model_config.max_model_len is None; please set it explicitly.")
    if max_model_len < 1:
        raise ValueError(f"model_config.max_model_len must be >= 1, got {max_model_len}.")

    truncated_prompts = 0
    skipped_prompts = 0

    for prompt in prompts:
        # 关键：必须用 tokenizer 统计 prompt token 数（字符串长度和 token 数没有稳定映射关系）
        prompt_token_ids = llm_engine.tokenizer.encode(prompt)
        if TRUNCATE_OVERLONG_PROMPTS:
            # 预留至少 MIN_OUTPUT_TOKENS 的生成空间；并至少保留 1 个 prompt token
            # （部分模型/实现对空 prompt 的边界行为可能不一致）。
            max_prompt_len = max(max_model_len - MIN_OUTPUT_TOKENS, 1)
            if len(prompt_token_ids) > max_prompt_len:
                # left-truncate：保留末尾 tokens（更接近“对话越长保留最近上下文”的常见策略）
                prompt_token_ids = prompt_token_ids[-max_prompt_len:]
                # 同步更新 prompt 文本，使输出/日志中的 prompt 与 token_ids 对齐。
                # 注意：这会改变原始数据集的 prompt 文本（只影响本次离线推理过程）。
                prompt = llm_engine.tokenizer.decode(
                    prompt_token_ids, skip_special_tokens=True
                )
                truncated_prompts += 1

        # 每条请求的生成 token 预算：必须保证总长度不超过 max_model_len
        max_new_tokens_budget = max_model_len - len(prompt_token_ids)
        if max_new_tokens_budget < 1:
            # prompt 已经占满（或超过）上下文窗口，没有生成空间。
            # 如果 TRUNCATE_OVERLONG_PROMPTS=True，理论上这里很少发生（除非 max_model_len 很小）。
            skipped_prompts += 1
            continue

        # 每条请求使用独立的 SamplingParams：
        # - 复用温度/top_p/top_k/stop 等
        # - max_tokens 按预算动态裁剪，避免 prompt_len + max_tokens > max_model_len
        per_request_sampling_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            stop=sampling_params.stop,
            ignore_eos=sampling_params.ignore_eos,
            max_tokens=min(sampling_params.max_tokens, max_new_tokens_budget),
        )
        # 显式传入 prompt_token_ids：
        # - 避免 engine 内部再次 encode（重复开销）
        # - 保证我们按 token_len 做的预算/截断与实际执行一致
        llm_engine.add_request(
            prompt, per_request_sampling_params, prompt_token_ids=prompt_token_ids
        )

    num_requests = llm_engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")

    # Run the engine
    outputs: List[RequestOutput] = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)

    pbar.close()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.seq_id))
    if truncated_prompts or skipped_prompts:
        print(
            f"Prompt length guard: truncated={truncated_prompts}, skipped={skipped_prompts} "
            f"(max_model_len={max_model_len}, requested_max_tokens={sampling_params.max_tokens})."
        )
    return outputs


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = generate(llm_engine, prompts, sampling_params)

llm_engine.pull_worker_metrics()
llm_engine.plot_metrics()
