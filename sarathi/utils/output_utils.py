import csv
from datetime import datetime
import os
from typing import List, Any


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