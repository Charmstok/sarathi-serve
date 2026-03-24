import datetime
import os
from typing import Any, Dict, List

from tqdm import tqdm

from sarathi import LLMEngine, RequestOutput, SamplingParams
from sarathi.config import (
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
    ReplicaConfig,
    SarathiSchedulerConfig,
    SystemConfig,
    WorkerConfig,
)
from sarathi.utils.output_utils import dump_run_config
from sarathi.utils.prompt_utils import (
    build_heterogeneous_prompt_dataset,
    prompt_arrival_time_smooth,
)

BASE_OUTPUT_DIR = "./offline_inference_output"
DATA_SOURCE = "dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

os.environ.setdefault("SARATHI_SAMPLING_BACKEND", "torch")

# 请求数
PROMPTS_NUMBER = 200
# token预算
CHUNK_SIZE = 256
# 请求到达系统的间隔时间
ARRIVAL_INTERVAL_S = 0.1

# 混合异构 prompt 参数
HETEROGENEOUS_PROMPT_SEED = 42
SHORT_PROMPT_MIN_TOKENS = 10
SHORT_PROMPT_MAX_TOKENS = 30
LONG_PROMPT_MIN_TOKENS = 500
LONG_PROMPT_MAX_TOKENS = 512
SHORT_LONG_RATIO = (1, 1)
MIN_DECODE_TOKENS = 16
SAVE_HETEROGENEOUS_PROMPTS = True

BASE_SAMPLING_PARAMS = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=LONG_PROMPT_MAX_TOKENS)

output_dir = (
    f"{BASE_OUTPUT_DIR}/"
    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    f"-heterogeneous_prompt-token预算-{CHUNK_SIZE}tokens"
)

replica_config = ReplicaConfig(
    output_dir=output_dir,
)
replica_output_dir = replica_config.output_dir
os.makedirs(replica_output_dir, exist_ok=True)

model_config = ModelConfig(
    model="Qwen/Qwen3-8B",
    max_model_len=1024,
)

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)

scheduler_config = SarathiSchedulerConfig(
    chunk_size=CHUNK_SIZE,
    max_num_seqs=32,
    enable_select_stats_csv=True,
)

metrics_config = MetricsConfig(
    write_metrics=True,
    enable_chrome_trace=True,
)

worker_config = WorkerConfig(
    gpu_memory_utilization=0.7,
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

heterogeneous_prompt_path = os.path.join(replica_output_dir, "heterogeneous_prompts.json")
prompt_records = build_heterogeneous_prompt_dataset(
    data_source=DATA_SOURCE,
    short_prompt_min_tokens=SHORT_PROMPT_MIN_TOKENS,
    short_prompt_max_tokens=SHORT_PROMPT_MAX_TOKENS,
    long_prompt_min_tokens=LONG_PROMPT_MIN_TOKENS,
    long_prompt_max_tokens=LONG_PROMPT_MAX_TOKENS,
    prompt_num=PROMPTS_NUMBER,
    save=SAVE_HETEROGENEOUS_PROMPTS,
    seed=HETEROGENEOUS_PROMPT_SEED,
    token_count_fn=lambda text: len(llm_engine.tokenizer.encode(text)),
    min_decode_tokens=MIN_DECODE_TOKENS,
    short_long_ratio=SHORT_LONG_RATIO,
    output_path=heterogeneous_prompt_path,
)
prompts_arrivaltime = prompt_arrival_time_smooth(len(prompt_records), ARRIVAL_INTERVAL_S)


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


def summarize_prompt_records(prompt_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    prefill_tokens = [int(record["num_prefill_tokens"]) for record in prompt_records]
    decode_tokens = [int(record["num_decode_tokens"]) for record in prompt_records]
    pd_ratios = [float(record["pd_ratio"]) for record in prompt_records]

    decode_regime_counts: Dict[str, int] = {}
    prompt_type_counts: Dict[str, int] = {}
    for record in prompt_records:
        regime = str(record["decode_regime"])
        decode_regime_counts[regime] = decode_regime_counts.get(regime, 0) + 1
        prompt_type = str(record["prompt_type"])
        prompt_type_counts[prompt_type] = prompt_type_counts.get(prompt_type, 0) + 1

    return {
        "num_requests": len(prompt_records),
        "prefill_tokens_min": min(prefill_tokens),
        "prefill_tokens_max": max(prefill_tokens),
        "decode_tokens_min": min(decode_tokens),
        "decode_tokens_max": max(decode_tokens),
        "pd_ratio_min": min(pd_ratios),
        "pd_ratio_max": max(pd_ratios),
        "prompt_type_counts": prompt_type_counts,
        "decode_regime_counts": decode_regime_counts,
    }


prompt_summary = summarize_prompt_records(prompt_records)
print(f"异构 prompt 统计: {prompt_summary}")

dump_run_config(
    output_dir=output_dir,
    script=__file__,
    base_output_dir=BASE_OUTPUT_DIR,
    data_source=DATA_SOURCE,
    prompts_number=PROMPTS_NUMBER,
    chunk_size=CHUNK_SIZE,
    arrival_interval_s=ARRIVAL_INTERVAL_S,
    base_sampling_params=BASE_SAMPLING_PARAMS,
    heterogeneous_prompt_config={
        "seed": HETEROGENEOUS_PROMPT_SEED,
        "short_prompt_min_tokens": SHORT_PROMPT_MIN_TOKENS,
        "short_prompt_max_tokens": SHORT_PROMPT_MAX_TOKENS,
        "long_prompt_min_tokens": LONG_PROMPT_MIN_TOKENS,
        "long_prompt_max_tokens": LONG_PROMPT_MAX_TOKENS,
        "short_long_ratio": SHORT_LONG_RATIO,
        "min_decode_tokens": MIN_DECODE_TOKENS,
        "save": SAVE_HETEROGENEOUS_PROMPTS,
        "output_path": heterogeneous_prompt_path,
    },
    heterogeneous_prompt_summary=prompt_summary,
    replica_config=replica_config,
    model_config=model_config,
    parallel_config=parallel_config,
    scheduler_config=scheduler_config,
    metrics_config=metrics_config,
    worker_config=worker_config,
    system_config=system_config,
)


def generate(
    llm_engine: LLMEngine,
    prompt_records: List[Dict[str, Any]],
    prompts_arrivaltime: List[float],
    base_sampling_params: SamplingParams,
) -> List[RequestOutput]:
    if len(prompt_records) != len(prompts_arrivaltime):
        raise ValueError(
            "prompt_records 与 prompts_arrivaltime 的长度不一致: "
            f"{len(prompt_records)} != {len(prompts_arrivaltime)}"
        )

    for prompt_record, arrival_time in zip(prompt_records, prompts_arrivaltime):
        prompt = str(prompt_record["prompt"])
        prompt_token_ids = llm_engine.tokenizer.encode(prompt)
        expected_prompt_tokens = int(prompt_record["num_prefill_tokens"])
        prompt_limit = llm_engine.get_model_config().max_model_len

        if len(prompt_token_ids) != expected_prompt_tokens:
            raise ValueError(
                "prompt token 数与异构数据集记录不一致: "
                f"{len(prompt_token_ids)} != {expected_prompt_tokens}"
            )

        remaining_decode_budget = prompt_limit - len(prompt_token_ids)
        if remaining_decode_budget <= 0:
            raise ValueError(
                "prompt 长度已经占满 max_model_len，无法继续分配 decode budget: "
                f"{len(prompt_token_ids)} >= {prompt_limit}"
            )

        request_sampling_params = build_request_sampling_params(
            base_sampling_params,
            min(int(prompt_record["num_decode_tokens"]), remaining_decode_budget),
        )

        llm_engine.add_request(
            prompt,
            request_sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
        )

    num_requests = len(prompt_records)
    pbar = tqdm(total=num_requests, desc="Processed prompts")

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


outputs = generate(
    llm_engine,
    prompt_records,
    prompts_arrivaltime,
    BASE_SAMPLING_PARAMS,
)

llm_engine.pull_worker_metrics()
llm_engine.plot_metrics()
