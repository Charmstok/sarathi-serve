from time import time
from typing import List

from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer

from sarathi.config import (
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
    ReplicaConfig,
    SarathiSchedulerConfig,
    SystemConfig,
    WorkerConfig,
)
from sarathi import LLMEngine, RequestOutput, SamplingParams


MODEL_NAME = "Qwen/Qwen3-32B"
# Reuse the cached Qwen3 tokenizer from 8B because the 32B snapshot on this
# machine currently lacks tokenizer files and otherwise encodes prompts to [].
TOKENIZER_NAME = "Qwen/Qwen3-8B"


prompts = [
    "你好，你是什么大模型？请介绍一下自己。",
    "介绍一下西安电子科技大学吧。",
    "2026年，作为一个软件工程的毕业生，未来该从哪个方面提升自己的编程能力？",
]

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048)

replica_config = ReplicaConfig()

model_config = ModelConfig(
    model=MODEL_NAME,
    max_model_len=4096,
)

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

scheduler_config = SarathiSchedulerConfig(
    chunk_size=512,
    max_num_seqs=2,
)

metrics_config = MetricsConfig(
    write_metrics=False,
    enable_chrome_trace=False,
)

worker_config = WorkerConfig(
    gpu_memory_utilization=0.9,
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


def load_prompt_tokenizer() -> AutoTokenizer:
    tokenizer_path = snapshot_download(TOKENIZER_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    return tokenizer


def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> List[RequestOutput]:
    tokenizer = load_prompt_tokenizer()
    llm_engine.tokenizer = tokenizer
    llm_engine.seq_manager.tokenizer = tokenizer

    for prompt in prompts:
        prompt_token_ids = tokenizer.encode(prompt)
        if not prompt_token_ids:
            raise RuntimeError(f"Tokenizer produced empty prompt ids for: {prompt!r}")
        llm_engine.add_request(
            prompt,
            sampling_params,
            prompt_token_ids=prompt_token_ids,
        )

    num_requests = llm_engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")
    start_time = time()
    step_idx = 0

    outputs: List[RequestOutput] = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        step_idx += 1
        if step_idx % 20 == 0:
            elapsed = time() - start_time
            pbar.set_postfix_str(
                f"steps={step_idx} unfinished={llm_engine.get_num_unfinished_requests()} elapsed={elapsed:.1f}s"
            )
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)

    pbar.close()
    outputs = sorted(outputs, key=lambda x: int(x.seq_id))
    return outputs


outputs = generate(llm_engine, prompts, sampling_params)
for output in outputs:
    prompt = getattr(output, "prompt", "")
    generated_text = getattr(output, "text", "")

    print("===========================================================")
    print(f"Prompt: {prompt!r}")
    print("-----------------------------------------------------------")
    print(f"Generated text: {generated_text!r}")
    print("===========================================================")
