import datetime
from tqdm import tqdm
from typing import List

from sarathi.config import ModelConfig, ParallelConfig, MetricsConfig, SystemConfig, WorkerConfig, \
    ReplicaConfig, OptSarathiSchedulerConfig, VllmSchedulerConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.utils.prompt_utils import get_prompts_from_dataset
from sarathi.utils.output_utils import process_and_save_outputs

BASE_OUTPUT_DIR = "./offline_inference_output"

# Sample prompts.
# prompts = [
#     "你好，你是什么大模型？是Qwen系列吗？如果是你当前是最新版本吗？",
#     "介绍一下西安电子科技大学吧。",
#     "自注意力机制是什么？解释具体原理。",
#     "未来AI的特征是什么？介绍一下。",
# ]

prompts = get_prompts_from_dataset("dataset/ShareGPT_V3_unfiltered_cleaned_split.json", 1000, random_sample=False)


sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

replica_config = ReplicaConfig(
    output_dir=output_dir,
)

model_config = ModelConfig(
    model="Qwen/Qwen3-8B",
    max_model_len=2048,
)

"""

支持的模型：  TinyLlama/TinyLlama-1.1B-Chat-v1.0

"""

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

scheduler_config = OptSarathiSchedulerConfig(
    chunk_size=256,
    max_num_seqs=10,
    policy_name="fcfs",
)

# scheduler_config = VllmSchedulerConfig()

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
process_and_save_outputs(outputs)

llm_engine.pull_worker_metrics()
llm_engine.plot_metrics()
