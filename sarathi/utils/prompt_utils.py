import time
import json
import random
import os
import csv
from typing import List


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