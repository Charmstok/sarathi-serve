import datetime
from tqdm import tqdm
from typing import List

from sarathi.config import ModelConfig, ParallelConfig, MetricsConfig, SystemConfig, WorkerConfig, \
    ReplicaConfig, OptSarathiSchedulerConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.utils.prompt_utils import get_prompts_from_dataset

BASE_OUTPUT_DIR = "./offline_inference_output"
TRUNCATE_OVERLONG_PROMPTS = True
MIN_OUTPUT_TOKENS = 1

prompts = get_prompts_from_dataset("dataset/ShareGPT_V3_unfiltered_cleaned_split.json", 150, random_sample=True)


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

scheduler_config = OptSarathiSchedulerConfig(
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
    max_model_len = llm_engine.get_model_config().max_model_len
    if max_model_len is None:
        raise ValueError("model_config.max_model_len is None; please set it explicitly.")
    if max_model_len < 1:
        raise ValueError(f"model_config.max_model_len must be >= 1, got {max_model_len}.")

    truncated_prompts = 0
    skipped_prompts = 0

    for prompt in prompts:
        prompt_token_ids = llm_engine.tokenizer.encode(prompt)
        if TRUNCATE_OVERLONG_PROMPTS:
            max_prompt_len = max(max_model_len - MIN_OUTPUT_TOKENS, 1)
            if len(prompt_token_ids) > max_prompt_len:
                prompt_token_ids = prompt_token_ids[-max_prompt_len:]
                prompt = llm_engine.tokenizer.decode(
                    prompt_token_ids, skip_special_tokens=True
                )
                truncated_prompts += 1

        max_new_tokens_budget = max_model_len - len(prompt_token_ids)
        if max_new_tokens_budget < 1:
            skipped_prompts += 1
            continue

        per_request_sampling_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            stop=sampling_params.stop,
            ignore_eos=sampling_params.ignore_eos,
            max_tokens=min(sampling_params.max_tokens, max_new_tokens_budget),
        )
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
