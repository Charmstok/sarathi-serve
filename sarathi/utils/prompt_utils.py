import time
import json
import random
import os
import csv
from typing import List, Optional


def get_prompts_from_dataset(
        data_source: str,
        samples_num: int,
        random_sample: bool = False,
        save: bool = False,
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

    # 如果传入的是文件路径，则读取文件
    if isinstance(data_source, str):
        try:
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: 文件 {data_source} 未找到。")
            return []
        except json.JSONDecodeError:
            print(f"Error: 文件 {data_source} 不是有效的 JSON 格式。")
            return []

    # 用于保存最终结果的列表：[(id, prompt), (id, prompt), ...]
    extracted_data = []
    # 仅用于返回的 prompt 列表
    prompts = []

    # 创建一个索引列表 [0, 1, 2, ... len(data)-1]
    data_indices = list(range(len(data)))

    # 如果需要随机，打乱索引列表
    # 打乱的是索引而不是 data 本身，这样可以避免修改传入的原始 list，且内存开销小
    if random_sample:
        random.shuffle(data_indices)

    # 遍历每一条数据 (对应每个 id)
    for idx in data_indices:
        if samples_num is not None and len(prompts) >= samples_num:
            break

        item = data[idx]
        conversations = item.get("conversations", [])
        item_id = item.get("id", "Unknown")  # 获取 ID，如果不存在则标记为 Unknown

        # 遍历对话列表，寻找第一个 human
        for turn in conversations:
            if turn.get("from") == "human":
                prompt_text = turn.get("value")

                # 收集数据
                prompts.append(prompt_text)
                extracted_data.append((item_id, prompt_text))

                # 找到第一个后通过 break 跳出当前对话循环，进入下一个 data item
                break

    if save:
        # 生成 CSV 文件
        try:
            # 获取源文件所在的目录
            dir_name = os.path.dirname(data_source)
            # 获取源文件名（不带扩展名）
            base_name = os.path.basename(data_source)
            file_name_without_ext = os.path.splitext(base_name)[0]
            now = time.monotonic()

            csv_filename = f"{file_name_without_ext}_extracted-{samples_num}-{now}.csv"
            csv_path = os.path.join(dir_name, csv_filename)

            # 写入 CSV
            with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
                writer = csv.writer(csv_file)
                # 写入表头
                writer.writerow(['id', 'prompt'])
                # 写入数据内容
                writer.writerows(extracted_data)

            print(f"Success: 已提取 {len(prompts)} 条数据，CSV 文件已保存至: {csv_path}")

        except Exception as e:
            print(f"Warning: CSV 文件保存失败: {e}")

    return prompts

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
