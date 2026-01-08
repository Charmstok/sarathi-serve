import datetime
from tqdm import tqdm
from typing import List, Optional

from sarathi.config import ModelConfig, ParallelConfig, MetricsConfig, SystemConfig, WorkerConfig, \
    ReplicaConfig, OptSarathiSchedulerConfig, SarathiSchedulerConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.utils.prompt_utils import *

BASE_OUTPUT_DIR = "./offline_inference_output"

PROMPTS_NUMBER = 200

prompts = get_prompts_from_dataset("dataset/ShareGPT_V3_unfiltered_cleaned_split.json", PROMPTS_NUMBER, random_sample=False)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

replica_config = ReplicaConfig(
    output_dir=output_dir,
)

model_config = ModelConfig(
    model="Qwen/Qwen3-8B",
    max_model_len=512,
)

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

scheduler_config = OptSarathiSchedulerConfig(
    target_time=100.0,
    max_num_seqs=32,
    enable_select_stats_csv=True,
)
# scheduler_config = SarathiSchedulerConfig(
#     chunk_size=256,
#     max_num_seqs=32,
#     enable_select_stats_csv=True
# )

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
    for prompt in prompts:
        llm_engine.add_request(prompt, sampling_params)
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
    return outputs


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = generate(llm_engine, prompts, sampling_params)
# 处理输出
# process_and_save_outputs(outputs)

llm_engine.pull_worker_metrics()
llm_engine.plot_metrics()
