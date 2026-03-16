import os
import datetime
from tqdm import tqdm
from typing import List, Tuple

from sarathi.config import ModelConfig, ParallelConfig, MetricsConfig, SystemConfig, WorkerConfig, \
    ReplicaConfig, OptSarathiSchedulerConfig, SarathiSchedulerConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.utils.prompt_utils import get_prompts_from_dataset, prompt_arrival_time_smooth

BASE_OUTPUT_DIR = "./offline_inference_output"
PROMPT_AMOUNT = 1000

os.environ.setdefault("SARATHI_SAMPLING_BACKEND", "torch")

prompts = get_prompts_from_dataset("dataset/ShareGPT_V3_unfiltered_cleaned_split.json", PROMPT_AMOUNT, random_sample=True)
prompts_arrivaltime = prompt_arrival_time_smooth(PROMPT_AMOUNT, 0.1)


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

# 使用 Sarathi 调度器，将测试结果输出到 csv 文件中，
# 使用这些数据进行训练，以让 OptSarathi 使用时间预算。
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


def build_request_sampling_params(
    base_sampling_params: SamplingParams,
    max_tokens: int,
) -> SamplingParams:
    return SamplingParams(
        temperature=base_sampling_params.temperature,
        top_p=base_sampling_params.top_p,
        top_k=base_sampling_params.top_k,
        stop=base_sampling_params.stop,
        ignore_eos=base_sampling_params.ignore_eos,
        max_tokens=max_tokens,
    )


def filter_prompts_by_len(
    llm_engine: LLMEngine,
    prompts: List[str],
    prompts_arrivaltime: List[float],
    sampling_params: SamplingParams,
) -> List[Tuple[str, List[int], float, SamplingParams]]:
    if len(prompts) != len(prompts_arrivaltime):
        raise ValueError(
            "prompts 与 prompts_arrivaltime 的长度不一致: "
            f"{len(prompts)} != {len(prompts_arrivaltime)}"
        )

    prompt_limit = llm_engine.get_model_config().max_model_len
    valid_prompts: List[Tuple[str, List[int], float, SamplingParams]] = []
    ignored_prompt_lens: List[int] = []
    capped_request_count = 0
    zero_budget_count = 0

    for prompt, arrival_time in zip(prompts, prompts_arrivaltime):
        prompt_token_ids = llm_engine.tokenizer.encode(prompt)
        if len(prompt_token_ids) > prompt_limit:
            ignored_prompt_lens.append(len(prompt_token_ids))
            continue

        remaining_decode_budget = prompt_limit - len(prompt_token_ids)
        if remaining_decode_budget <= 0:
            zero_budget_count += 1
            continue

        request_max_tokens = min(sampling_params.max_tokens, remaining_decode_budget)
        if request_max_tokens < sampling_params.max_tokens:
            capped_request_count += 1

        request_sampling_params = build_request_sampling_params(
            sampling_params,
            request_max_tokens,
        )
        valid_prompts.append(
            (prompt, prompt_token_ids, arrival_time, request_sampling_params)
        )

    if ignored_prompt_lens:
        print(
            f"在进入调度前，已过滤掉 {len(ignored_prompt_lens)} 条长度超过 "
            f"{prompt_limit} tokens 的 prompt "
            f"（最长为 {max(ignored_prompt_lens)} tokens）。"
        )

    if zero_budget_count:
        print(
            f"另有 {zero_budget_count} 条 prompt 虽未超过 {prompt_limit} tokens，"
            "但没有剩余 decode 空间，因此已跳过。"
        )

    if capped_request_count:
        print(
            f"有 {capped_request_count} 条 prompt 的可生成 token 数已自动收紧，"
            "以保证 prompt 长度与生成长度之和不超过 max_model_len。"
        )

    print(f"即将向引擎提交 {len(valid_prompts)} 条 prompt。")
    return valid_prompts


def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    prompts_arrivaltime: List[float],
    sampling_params: SamplingParams,
) -> List[RequestOutput]:
    valid_prompts = filter_prompts_by_len(
        llm_engine,
        prompts,
        prompts_arrivaltime,
        sampling_params,
    )

    for prompt, prompt_token_ids, arrival_time, request_sampling_params in valid_prompts:
        llm_engine.add_request(
            prompt,
            request_sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
        )

    num_requests = len(valid_prompts)
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
    return outputs


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = generate(llm_engine, prompts, prompts_arrivaltime, sampling_params)

llm_engine.pull_worker_metrics()
llm_engine.plot_metrics()
