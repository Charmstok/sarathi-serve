import json
import os
import csv
import hashlib
import math
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

_CJK_OR_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
    re.UNICODE,
)


def _load_dataset(data_source: Union[str, Sequence[dict]]) -> List[dict]:
    if isinstance(data_source, str):
        try:
            with open(data_source, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: 文件 {data_source} 未找到。")
            return []
        except json.JSONDecodeError:
            print(f"Error: 文件 {data_source} 不是有效的 JSON 格式。")
            return []
    else:
        data = list(data_source)

    if not isinstance(data, list):
        print("Error: 数据源必须是 list[dict] 或 JSON 文件路径。")
        return []
    return data


def _extract_first_human_prompts(data: Sequence[dict]) -> List[Dict[str, Any]]:
    extracted_data: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        conversations = item.get("conversations", [])
        if not isinstance(conversations, list):
            continue

        item_id = str(item.get("id", idx))
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            if turn.get("from") != "human":
                continue

            prompt_text = turn.get("value")
            if not isinstance(prompt_text, str):
                break
            prompt_text = prompt_text.strip()
            if not prompt_text:
                break

            extracted_data.append(
                {
                    "source_index": idx,
                    "source_id": item_id,
                    "prompt": prompt_text,
                }
            )
            break
    return extracted_data


def _stable_digest(seed: int, *parts: Any) -> str:
    joined = "||".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _stable_unit_float(seed: int, *parts: Any) -> float:
    digest = hashlib.sha256(
        "||".join(str(part) for part in (seed, *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)


def _approximate_token_count(text: str) -> int:
    if not text:
        return 0
    return len(_CJK_OR_TOKEN_PATTERN.findall(text))


def _build_token_counter(
    tokenizer: Optional[Any] = None,
    token_count_fn: Optional[Callable[[str], int]] = None,
) -> Tuple[Callable[[str], int], str]:
    if tokenizer is not None and token_count_fn is not None:
        raise ValueError("tokenizer 与 token_count_fn 只能传入一个。")

    if token_count_fn is not None:
        return token_count_fn, "custom_fn"

    if tokenizer is not None:
        if not hasattr(tokenizer, "encode"):
            raise TypeError("tokenizer 必须提供 encode(text) 接口。")

        def count_tokens(text: str) -> int:
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(text)
            return len(token_ids)

        return count_tokens, f"tokenizer:{tokenizer.__class__.__name__}"

    return _approximate_token_count, "approximate_regex"


def _bucket_edges(
    min_tokens: int,
    max_tokens: int,
    length_bins: int,
    use_log_bins: bool,
) -> List[int]:
    if length_bins <= 1 or min_tokens == max_tokens:
        return [min_tokens, max_tokens + 1]

    edges = [min_tokens]

    if use_log_bins and min_tokens > 0 and max_tokens / min_tokens >= 4:
        log_min = math.log(min_tokens)
        log_max = math.log(max_tokens + 1)
        for idx in range(1, length_bins):
            edge = int(round(math.exp(log_min + (log_max - log_min) * idx / length_bins)))
            if min_tokens < edge <= max_tokens:
                edges.append(edge)
    else:
        span = max_tokens - min_tokens + 1
        for idx in range(1, length_bins):
            edge = min_tokens + math.floor(span * idx / length_bins)
            if min_tokens < edge <= max_tokens:
                edges.append(edge)

    edges = sorted(set(edges))
    if edges[-1] != max_tokens + 1:
        edges.append(max_tokens + 1)
    return edges


def _bucket_index(token_count: int, edges: List[int]) -> int:
    for idx in range(len(edges) - 1):
        if edges[idx] <= token_count < edges[idx + 1]:
            return idx
    return len(edges) - 2


def _bucket_label(bucket_idx: int, edges: List[int]) -> str:
    start = edges[bucket_idx]
    end = edges[bucket_idx + 1] - 1
    return f"{start}-{end}"


def _interleave_bucket_indices(num_buckets: int) -> List[int]:
    order: List[int] = []
    left = 0
    right = num_buckets - 1

    while left <= right:
        order.append(left)
        if left != right:
            order.append(right)
        left += 1
        right -= 1

    return order


def _clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def _normalize_short_long_ratio(short_long_ratio: Tuple[int, int]) -> Tuple[int, int]:
    if len(short_long_ratio) != 2:
        raise ValueError(
            "short_long_ratio 必须是长度为 2 的二元组，例如 (1, 1) 或 (3, 7)。"
        )

    short_ratio = int(short_long_ratio[0])
    long_ratio = int(short_long_ratio[1])
    if short_ratio <= 0 or long_ratio <= 0:
        raise ValueError(
            "short_long_ratio 中的两个值都必须 > 0，"
            f"got {short_long_ratio}"
        )

    gcd_value = math.gcd(short_ratio, long_ratio)
    return short_ratio // gcd_value, long_ratio // gcd_value


def _compute_group_sample_counts(
    prompt_num: int,
    short_long_ratio: Tuple[int, int],
) -> Tuple[int, int]:
    short_ratio, long_ratio = _normalize_short_long_ratio(short_long_ratio)
    total_ratio = short_ratio + long_ratio

    short_target = prompt_num * short_ratio / total_ratio
    long_target = prompt_num * long_ratio / total_ratio

    short_count = math.floor(short_target)
    long_count = math.floor(long_target)
    remaining = prompt_num - short_count - long_count

    fractional_parts = [
        ("short", short_target - short_count),
        ("long", long_target - long_count),
    ]
    fractional_parts.sort(key=lambda item: (-item[1], item[0]))

    for idx in range(remaining):
        if fractional_parts[idx % 2][0] == "short":
            short_count += 1
        else:
            long_count += 1

    return short_count, long_count


def _build_short_long_schedule(
    short_count: int,
    long_count: int,
    seed: int,
) -> List[str]:
    schedule_items = [
        ("short", idx)
        for idx in range(short_count)
    ] + [
        ("long", idx)
        for idx in range(long_count)
    ]

    schedule_items.sort(
        key=lambda item: _stable_digest(
            seed,
            "short_long_schedule",
            item[0],
            item[1],
        )
    )
    return [item[0] for item in schedule_items]


def _assign_two_group_decode_tokens(
    prompt_group: str,
    sample_key: str,
    position: int,
    seed: int,
    min_decode_tokens: int,
    short_prompt_min_tokens: int,
    short_prompt_max_tokens: int,
    long_prompt_min_tokens: int,
    long_prompt_max_tokens: int,
) -> Tuple[int, str]:
    jitter = _stable_unit_float(seed, "decode", prompt_group, sample_key, position)

    if prompt_group == "short":
        decode_low = max(min_decode_tokens, long_prompt_min_tokens)
        decode_high = max(decode_low, long_prompt_max_tokens)
        decode_tokens = _clamp_int(
            decode_low + jitter * (decode_high - decode_low),
            decode_low,
            decode_high,
        )
        return decode_tokens, "decode_heavy"

    if prompt_group == "long":
        decode_low = min_decode_tokens
        decode_high = max(decode_low, short_prompt_max_tokens)
        decode_tokens = _clamp_int(
            decode_low + jitter * (decode_high - decode_low),
            decode_low,
            decode_high,
        )
        return decode_tokens, "prefill_heavy"

    raise ValueError(f"未知的 prompt_group: {prompt_group}")


def _default_saved_path(
    data_source: Union[str, Sequence[dict]],
    prompt_num: int,
    short_prompt_min_tokens: int,
    short_prompt_max_tokens: int,
    long_prompt_min_tokens: int,
    long_prompt_max_tokens: int,
    short_long_ratio: Tuple[int, int],
    seed: int,
) -> str:
    if isinstance(data_source, str):
        dir_name = os.path.dirname(data_source) or "."
        base_name = os.path.splitext(os.path.basename(data_source))[0]
    else:
        dir_name = "."
        base_name = "in_memory_dataset"

    file_name = (
        f"{base_name}_heterogeneous_prompts_seed{seed}"
        f"_n{prompt_num}"
        f"_short{short_prompt_min_tokens}-{short_prompt_max_tokens}"
        f"_long{long_prompt_min_tokens}-{long_prompt_max_tokens}"
        f"_ratio{short_long_ratio[0]}-{short_long_ratio[1]}.json"
    )
    return os.path.join(dir_name, file_name)


def get_prompts_from_dataset(
        data_source: Union[str, Sequence[dict]],
        samples_num: int,
        random_sample: bool = False,
        save: bool = False,
        seed: int = 42,
) -> List[str]:
    """
    从ShareGPT格式的数据集中提取每个 ID 对应的第一个 human 提示词。
    目前测试的数据集：[https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main]

    Args:
        data_source (Union[str, List[dict]]): JSON 数据集的路径(str)
        samples_num (int): 需要的数据样本的数量。
        random_sample (bool): 是否随机打乱读取。
        save (bool): 是否需要将结果保存到数据来源的目录下。

    Returns:
        List[str]: 提取出的提示词字符串列表。
    """

    data = _load_dataset(data_source)
    if not data:
        return []

    extracted_data = _extract_first_human_prompts(data)
    # 仅用于返回 / 保存的 prompt 列表
    selected_items: List[Dict[str, Any]] = []
    prompts: List[str] = []

    # 创建一个索引列表 [0, 1, 2, ... len(extracted_data)-1]
    data_indices = list(range(len(extracted_data)))

    # 如果需要随机，打乱索引列表
    # 打乱的是索引而不是 data 本身，这样可以避免修改传入的原始 list，且内存开销小
    if random_sample:
        random.Random(seed).shuffle(data_indices)

    # 遍历每一条数据 (对应每个 id)
    for idx in data_indices:
        if samples_num is not None and len(prompts) >= samples_num:
            break

        item = extracted_data[idx]
        selected_items.append(item)
        prompts.append(item["prompt"])

    if save:
        # 生成 CSV 文件
        try:
            if isinstance(data_source, str):
                dir_name = os.path.dirname(data_source) or "."
                base_name = os.path.basename(data_source)
                file_name_without_ext = os.path.splitext(base_name)[0]
            else:
                dir_name = "."
                file_name_without_ext = "in_memory_dataset"
            now = time.monotonic()

            csv_filename = f"{file_name_without_ext}_extracted-{samples_num}-{now}.csv"
            csv_path = os.path.join(dir_name, csv_filename)

            # 写入 CSV
            with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                # 写入表头
                writer.writerow(["id", "prompt"])
                # 写入数据内容
                writer.writerows(
                    (item["source_id"], item["prompt"])
                    for item in selected_items
                )

            print(f"Success: 已提取 {len(prompts)} 条数据，CSV 文件已保存至: {csv_path}")

        except Exception as e:
            print(f"Warning: CSV 文件保存失败: {e}")

    return prompts


def build_heterogeneous_prompt_dataset(
        data_source: Union[str, Sequence[dict]],
        short_prompt_min_tokens: int,
        short_prompt_max_tokens: int,
        long_prompt_min_tokens: int,
        long_prompt_max_tokens: int,
        prompt_num: int,
        save: bool = False,
        *,
        seed: int = 42,
        tokenizer: Optional[Any] = None,
        token_count_fn: Optional[Callable[[str], int]] = None,
        min_decode_tokens: int = 16,
        short_long_ratio: Tuple[int, int] = (1, 1),
        output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    构造一个“长度异构 + prefill/decode 混合明显”的可复现 prompt 数据集。

    默认行为：
    1. 从 ShareGPT 风格数据集中提取每条样本的首个 human prompt。
    2. 仅保留两种 prompt：
       - short prompt: [short_prompt_min_tokens, short_prompt_max_tokens]
       - long prompt: [long_prompt_min_tokens, long_prompt_max_tokens]
    3. 按 short_long_ratio 采样后随机打散 short / long 顺序，且同一 seed 下可复现。
    4. short prompt 默认分配更长 decode，long prompt 默认分配更短 decode，
       使 prefill / decode 的混合变化更明显。

    Args:
        data_source: JSON 文件路径或已加载的数据集。
        short_prompt_min_tokens: 短 prompt 的最小 token 数。
        short_prompt_max_tokens: 短 prompt 的最大 token 数。
        long_prompt_min_tokens: 长 prompt 的最小 token 数。
        long_prompt_max_tokens: 长 prompt 的最大 token 数。
        prompt_num: 需要生成的 prompt 数量。
        save: 是否将结果保存为 JSON 数据集。
        seed: 复现用随机种子。
        tokenizer: 可选，若提供则用其 encode 统计 token 数。
        token_count_fn: 可选，自定义 token 计数函数。与 tokenizer 二选一。
        min_decode_tokens: 每条请求允许的最小 decode token 数。
        short_long_ratio: short / long 两种 prompt 的采样比例，默认 (1, 1)。
        output_path: save=True 时的保存路径；为空则使用稳定的默认文件名。

    Returns:
        List[Dict[str, Any]]:
            每条记录包含以下字段：
            - source_index
            - source_id
            - prompt
            - num_prefill_tokens
            - num_decode_tokens
            - total_tokens
            - pd_ratio
            - prompt_type
            - length_bucket
            - decode_regime
    """
    if short_prompt_min_tokens <= 0:
        raise ValueError(
            f"short_prompt_min_tokens 必须 > 0, got {short_prompt_min_tokens}"
        )
    if short_prompt_max_tokens < short_prompt_min_tokens:
        raise ValueError(
            "short_prompt_max_tokens 不能小于 short_prompt_min_tokens: "
            f"{short_prompt_max_tokens} < {short_prompt_min_tokens}"
        )
    if long_prompt_min_tokens <= 0:
        raise ValueError(
            f"long_prompt_min_tokens 必须 > 0, got {long_prompt_min_tokens}"
        )
    if long_prompt_max_tokens < long_prompt_min_tokens:
        raise ValueError(
            "long_prompt_max_tokens 不能小于 long_prompt_min_tokens: "
            f"{long_prompt_max_tokens} < {long_prompt_min_tokens}"
        )
    if short_prompt_max_tokens >= long_prompt_min_tokens:
        raise ValueError(
            "短 prompt 区间与长 prompt 区间不能重叠，要求 "
            f"short_prompt_max_tokens < long_prompt_min_tokens，"
            f"got {short_prompt_max_tokens} >= {long_prompt_min_tokens}"
        )
    if prompt_num <= 0:
        raise ValueError(f"prompt_num 必须 > 0, got {prompt_num}")
    if min_decode_tokens <= 0:
        raise ValueError(f"min_decode_tokens 必须 > 0, got {min_decode_tokens}")

    short_long_ratio = _normalize_short_long_ratio(short_long_ratio)
    short_target_count, long_target_count = _compute_group_sample_counts(
        prompt_num=prompt_num,
        short_long_ratio=short_long_ratio,
    )

    data = _load_dataset(data_source)
    if not data:
        return []

    token_counter, token_counter_name = _build_token_counter(tokenizer, token_count_fn)
    raw_prompts = _extract_first_human_prompts(data)
    if not raw_prompts:
        return []

    short_candidates: List[Dict[str, Any]] = []
    long_candidates: List[Dict[str, Any]] = []
    for item in raw_prompts:
        prompt = item["prompt"]
        prompt_tokens = token_counter(prompt)
        if short_prompt_min_tokens <= prompt_tokens <= short_prompt_max_tokens:
            prompt_type = "short"
            length_bucket = f"{short_prompt_min_tokens}-{short_prompt_max_tokens}"
        elif long_prompt_min_tokens <= prompt_tokens <= long_prompt_max_tokens:
            prompt_type = "long"
            length_bucket = f"{long_prompt_min_tokens}-{long_prompt_max_tokens}"
        else:
            continue

        prompt_digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        candidate = {
            **item,
            "prompt_type": prompt_type,
            "num_prefill_tokens": prompt_tokens,
            "length_bucket": length_bucket,
            "sample_key": f'{item["source_index"]}:{item["source_id"]}:{prompt_digest}',
        }
        if prompt_type == "short":
            short_candidates.append(candidate)
        else:
            long_candidates.append(candidate)

    if len(short_candidates) < short_target_count:
        raise ValueError(
            "短 prompt 样本不足。"
            f"需要 {short_target_count} 条，实际只有 {len(short_candidates)} 条。"
            f"当前短 prompt 区间为 [{short_prompt_min_tokens}, {short_prompt_max_tokens}]。"
        )

    if len(long_candidates) < long_target_count:
        raise ValueError(
            "长 prompt 样本不足。"
            f"需要 {long_target_count} 条，实际只有 {len(long_candidates)} 条。"
            f"当前长 prompt 区间为 [{long_prompt_min_tokens}, {long_prompt_max_tokens}]。"
        )

    short_candidates.sort(
        key=lambda item: _stable_digest(
            seed,
            "short_candidate",
            item["sample_key"],
        )
    )
    long_candidates.sort(
        key=lambda item: _stable_digest(
            seed,
            "long_candidate",
            item["sample_key"],
        )
    )

    selected_short = short_candidates[:short_target_count]
    selected_long = long_candidates[:long_target_count]

    schedule = _build_short_long_schedule(
        short_count=short_target_count,
        long_count=long_target_count,
        seed=seed,
    )

    if len(schedule) != prompt_num:
        raise RuntimeError(
            "短/长 prompt 调度计划长度异常: "
            f"{len(schedule)} != {prompt_num}"
        )

    selected_items: List[Dict[str, Any]] = []
    short_idx = 0
    long_idx = 0
    for prompt_type in schedule:
        if prompt_type == "short":
            selected_items.append(selected_short[short_idx])
            short_idx += 1
        else:
            selected_items.append(selected_long[long_idx])
            long_idx += 1

    records: List[Dict[str, Any]] = []
    for position, item in enumerate(selected_items):
        decode_tokens, decode_regime = _assign_two_group_decode_tokens(
            prompt_group=str(item["prompt_type"]),
            sample_key=item["sample_key"],
            position=position,
            seed=seed,
            min_decode_tokens=min_decode_tokens,
            short_prompt_min_tokens=short_prompt_min_tokens,
            short_prompt_max_tokens=short_prompt_max_tokens,
            long_prompt_min_tokens=long_prompt_min_tokens,
            long_prompt_max_tokens=long_prompt_max_tokens,
        )
        records.append(
            {
                "source_index": item["source_index"],
                "source_id": item["source_id"],
                "prompt": item["prompt"],
                "num_prefill_tokens": item["num_prefill_tokens"],
                "num_decode_tokens": decode_tokens,
                "total_tokens": item["num_prefill_tokens"] + decode_tokens,
                "pd_ratio": round(item["num_prefill_tokens"] / decode_tokens, 6),
                "prompt_type": item["prompt_type"],
                "length_bucket": item["length_bucket"],
                "decode_regime": decode_regime,
            }
        )

    if save:
        if output_path is None:
            output_path = _default_saved_path(
                data_source=data_source,
                prompt_num=prompt_num,
                short_prompt_min_tokens=short_prompt_min_tokens,
                short_prompt_max_tokens=short_prompt_max_tokens,
                long_prompt_min_tokens=long_prompt_min_tokens,
                long_prompt_max_tokens=long_prompt_max_tokens,
                short_long_ratio=short_long_ratio,
                seed=seed,
            )

        payload = {
            "metadata": {
                "data_source": data_source if isinstance(data_source, str) else "in_memory_dataset",
                "seed": seed,
                "prompt_num": prompt_num,
                "short_prompt_min_tokens": short_prompt_min_tokens,
                "short_prompt_max_tokens": short_prompt_max_tokens,
                "long_prompt_min_tokens": long_prompt_min_tokens,
                "long_prompt_max_tokens": long_prompt_max_tokens,
                "short_long_ratio": list(short_long_ratio),
                "short_prompt_count": short_target_count,
                "long_prompt_count": long_target_count,
                "min_decode_tokens": min_decode_tokens,
                "token_count_method": token_counter_name,
            },
            "records": records,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Success: 已生成 {len(records)} 条异构 prompt，保存至: {output_path}")

    return records


def unpack_heterogeneous_prompt_dataset(
        prompt_records: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[int]]:
    """
    将 build_heterogeneous_prompt_dataset 的输出拆成 prompt 列表和 decode token 列表。
    """
    prompts = [record["prompt"] for record in prompt_records]
    decode_tokens = [int(record["num_decode_tokens"]) for record in prompt_records]
    return prompts, decode_tokens


def prompt_arrival_time_smooth(
        num_requests: int,
        interval_s: float = 0.1,
        start_time: Optional[float] = None,
) -> List[float]:
    """
    离线测试中，模拟请求平滑到达系统的时间序列。

    Args:
        num_requests: 请求数量
        interval_s: 请求“到达”的时间间隔
        start_time: 开始时间（第一个请求的到达时间）

    Returns:
        List[float]: 请求到达时间的列表
    """
    if interval_s <= 0:
        raise ValueError(f"interval_s 必须 > 0, got {interval_s}")
    
    if start_time is None:
        start_time = time.monotonic()
    return [start_time + i * interval_s for i in range(num_requests)]

def prompt_arrival_time_clustered(
        num_requests: int,
        interval_s: float = 0.1,
        start_time: Optional[float] = None,
        cluster_start_pct: float = 0.0,
        cluster_end_pct: float = 0.0,
) -> List[float]:
    """
    离线测试中，模拟“某一时刻开始，请求聚集”的到达时间序列。

    聚集表现：在聚集区间内，请求到达间隔固定为 1ms；其余区间使用 interval_s。

    Args:
        num_requests: 请求数量
        interval_s: 非聚集区间的请求到达间隔（秒）
        start_time: 开始时间（第一个请求的到达时间，默认 time.monotonic()）
        cluster_start_pct: 从第多少比例的请求开始聚集（[0, 1]），按索引计算为 int(num_requests * pct)
        cluster_end_pct: 到第多少比例的请求停止聚集（[0, 1]，且 >= cluster_start_pct），区间为 [start_idx, end_idx)

    Returns:
        List[float]: 请求到达时间列表（长度为 num_requests）
    """
    if num_requests <= 0:
        return []

    if interval_s <= 0:
        raise ValueError(f"interval_s 必须 > 0, got {interval_s}")

    if not (0.0 <= cluster_start_pct <= 1.0):
        raise ValueError(f"cluster_start_pct 必须在区间 [0, 1] 内, {cluster_start_pct}")
    if not (0.0 <= cluster_end_pct <= 1.0):
        raise ValueError(f"cluster_end_pct 必须在区间 [0, 1] 内, {cluster_end_pct}")
    if cluster_end_pct < cluster_start_pct:
        raise ValueError(
            f"cluster_end_pct 必须 >= cluster_start_pct, {cluster_end_pct} < {cluster_start_pct}"
        )

    if start_time is None:
        start_time = time.monotonic()

    cluster_interval_s = 0.001
    start_idx = int(num_requests * cluster_start_pct)
    end_idx = int(num_requests * cluster_end_pct)
    start_idx = max(0, min(num_requests, start_idx))
    end_idx = max(start_idx, min(num_requests, end_idx))

    arrival_times: List[float] = [start_time]
    current_time = start_time

    for request_idx in range(1, num_requests):
        delta = cluster_interval_s if start_idx <= request_idx < end_idx else interval_s
        current_time += delta
        arrival_times.append(current_time)

    return arrival_times
