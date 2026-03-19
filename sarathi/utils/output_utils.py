import csv
from datetime import datetime
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import os
from typing import Any, List


def process_and_save_outputs(
        outputs: List[Any],
        csv_filename: str = "generation_results"
) -> None:
    """
    处理输出结果：打印到控制台并保存为 CSV 文件。

    Args:
        outputs: 包含 prompt 和 text 属性的输出对象列表
        csv_filename: 保存的 CSV 文件名，默认为 "generation_results.csv"
    """

    BASE_OUTPUT_DIR = "./offline_inference_output"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    full_filename = f"{csv_filename}-{timestamp}.csv"
    csv_filename = os.path.join(BASE_OUTPUT_DIR, full_filename)

    # 用于收集待写入 CSV 的数据
    csv_data = []

    print(f"正在处理 {len(outputs)} 条结果...")

    # 遍历输出，打印并收集数据
    for output in outputs:
        prompt = getattr(output, 'prompt', '')
        generated_text = getattr(output, 'text', '')

        # 收集数据
        csv_data.append([prompt, generated_text])

        # 打印日志
        print("===========================================================")
        print(f"Prompt: {prompt!r}")
        print("-----------------------------------------------------------")
        print(f"Generated text: {generated_text!r}")
        print("===========================================================")

    # 写入 CSV 文件
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow(['Prompt', 'Generated Text'])

            # 写入内容
            writer.writerows(csv_data)

        abs_path = os.path.abspath(csv_filename)
        print(f"\n[Success] CSV 文件已成功保存至:\n{abs_path}")

    except Exception as e:
        print(f"\n[Error] 保存 CSV 文件时发生错误: {e}")


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _to_serializable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dump_run_config(
    *,
    output_dir: str,
    config_name: str = "config.json",
    **payload: Any,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, config_name)
    serializable_payload = {
        key: _to_serializable(value)
        for key, value in payload.items()
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(serializable_payload, f, ensure_ascii=False, indent=2)
    return config_path
