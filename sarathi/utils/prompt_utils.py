import json
from typing import List, Union


def get_prompts_from_dataset(
        data_source: str,
        samples_num: int
) -> List[str]:
    ######################################################################
    # 从ShareGPT格式的数据集中提取每个 ID 对应的第一个 human 提示词。
    # 目前测试的数据集：[https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main]
    #
    # Args:
    #     data_source (Union[str, List[dict]]): JSON 数据集的路径(str)
    #     samples_num (int): 需要的数据样本的数量。
    #
    # Returns:
    #     List[str]: 提取出的提示词字符串列表。
    ######################################################################

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

    prompts = []

    # 遍历每一条数据 (对应每个 id)
    for item in data:
        if samples_num is not None and len(prompts) >= samples_num:
            break

        conversations = item.get("conversations", [])

        # 遍历对话列表，寻找第一个 human
        for turn in conversations:
            if turn.get("from") == "human":
                # 找到后提取 value
                prompts.append(turn.get("value"))
                # 找到第一个后通过 break 跳出当前对话循环，进入下一个 data item
                break

    return prompts