import argparse
import datetime
from typing import List

from tqdm import tqdm

from sarathi.config import (
    ModelConfig,
    ParallelConfig,
    MetricsConfig,
    SystemConfig,
    WorkerConfig,
    ReplicaConfig,
    OptSarathiSchedulerConfig,
)
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.utils.prompt_utils import get_prompts_from_dataset

BASE_OUTPUT_DIR = "./offline_inference_output"

prompts = get_prompts_from_dataset("dataset/ShareGPT_V3_unfiltered_cleaned_split.json", 50, random_sample=False)


sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

replica_config = ReplicaConfig(
    output_dir=output_dir,
)

model_config = ModelConfig(
    model="Qwen/Qwen3-8B",
    max_model_len=2048,
)

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> None:
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


def main(
        min_chunk_threshold: int
    ) -> None:
    scheduler_config = OptSarathiSchedulerConfig(
        chunk_size=256,
        max_num_seqs=10,
        policy_name="fcfs",
        min_chunk_threshold=min_chunk_threshold,
    )

    metrics_config = MetricsConfig(
        write_metrics=True,
        enable_chrome_trace=True,
    )

    worker_config = WorkerConfig(
        # gpu_memory_utilization=0.8
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

    generate(llm_engine, prompts, sampling_params)

    llm_engine.pull_worker_metrics()
    llm_engine.plot_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行离线阈值测试。")
    parser.add_argument(
        "--min_chunk_threshold",
        type=int,
        default=1,
        help="OptSarathiScheduler 使用的最小分块阈值。",
    )
    args = parser.parse_args()
    main(args.min_chunk_threshold)
