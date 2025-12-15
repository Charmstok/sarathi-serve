from tqdm import tqdm
from typing import List

from sarathi.config import ModelConfig, ParallelConfig, MetricsConfig, SystemConfig, WorkerConfig, \
    ReplicaConfig, OptSarathiSchedulerConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput


# Sample prompts.
prompts = [
    "你好，你是什么大模型？是Qwen系列吗？如果是你，当前是最新版本吗？",
    "介绍一下西安电子科技大学吧。",
    "自注意力机制是什么？解释具体原理。",
    "未来AI的特征是什么？介绍一下。",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

replica_config = ReplicaConfig(

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
    chunk_size=256,
    max_num_seqs=10,
)

metrics_config = MetricsConfig(
    write_metrics=False,
    enable_chrome_trace=True,
)

worker_config = WorkerConfig(

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

    outputs = sorted(outputs, key=lambda x: int(x.seq_id))
    return outputs


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = generate(llm_engine, prompts, sampling_params)
for output in outputs:
    prompt = getattr(output, 'prompt', '')
    generated_text = getattr(output, 'text', '')

    # 打印日志
    print("===========================================================")
    print(f"Prompt: {prompt!r}")
    print("-----------------------------------------------------------")
    print(f"Generated text: {generated_text!r}")
    print("===========================================================")

