import datetime
from tqdm import tqdm
from typing import List

from sarathi.config import ModelConfig, ParallelConfig, MetricsConfig, SystemConfig, WorkerConfig, \
    ReplicaConfig, OptSarathiSchedulerConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.utils.output_utils import dump_run_config
from sarathi.utils.prompt_utils import *

BASE_OUTPUT_DIR = "./offline_inference_output"

# 请求数
PROMPTS_NUMBER = 1000
# 时间预算
TARGET_TIME = 110
# 请求到达系统的间隔时间
ARRIVAL_INTERVAL_S = 0.1

prompts = get_prompts_from_dataset("dataset/ShareGPT_V3_unfiltered_cleaned_split.json", PROMPTS_NUMBER, random_sample=True, seed=42)
prompts_arrivaltime = prompt_arrival_time_smooth(len(prompts), ARRIVAL_INTERVAL_S)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-时间预算-{TARGET_TIME}ms"

replica_config = ReplicaConfig(
    output_dir=output_dir,
)

model_config = ModelConfig(
    model="Qwen/Qwen3-8B",
    max_model_len=4096,
)

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

scheduler_config = OptSarathiSchedulerConfig(
    target_time=TARGET_TIME,
    chunk_size=1024,
    max_num_seqs=256,
    enable_select_stats_csv=True,
    chunk_score_underfill_penalty=1.0,
    chunk_score_overflow_penalty=2.5
)

metrics_config = MetricsConfig(
    write_metrics=True,
    enable_chrome_trace=True,
)

worker_config = WorkerConfig(
    gpu_memory_utilization=0.6
)

system_config = SystemConfig(
    replica_config=replica_config,
    model_config=model_config,
    parallel_config=parallel_config,
    scheduler_config=scheduler_config,
    metrics_config=metrics_config,
    worker_config=worker_config,
)


dump_run_config(
    output_dir=output_dir,
    script=__file__,
    base_output_dir=BASE_OUTPUT_DIR,
    prompts_number=PROMPTS_NUMBER,
    target_time=TARGET_TIME,
    arrival_interval_s=ARRIVAL_INTERVAL_S,
    sampling_params=sampling_params,
    replica_config=replica_config,
    model_config=model_config,
    parallel_config=parallel_config,
    scheduler_config=scheduler_config,
    metrics_config=metrics_config,
    worker_config=worker_config,
    system_config=system_config,
)

llm_engine = LLMEngine.from_system_config(system_config)


def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    prompts_arrivaltime: List[float],
    sampling_params: SamplingParams,
) -> List[RequestOutput]:
    if len(prompts) != len(prompts_arrivaltime):
        raise ValueError(
            "prompts 与 prompts_arrivaltime 的长度不一致: "
            f"{len(prompts)} != {len(prompts_arrivaltime)}"
        )

    for prompt, arrival_time in zip(prompts, prompts_arrivaltime):
        llm_engine.add_request(
            prompt,
            sampling_params,
            arrival_time=arrival_time,
        )
    num_requests = len(prompts)
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
# 处理输出
# process_and_save_outputs(outputs)

llm_engine.pull_worker_metrics()
llm_engine.plot_metrics()
