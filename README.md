# Opt-Sarathi-Serve

Sarathi-Serve 是一个高吞吐、低延迟的大模型推理框架，技术细节详见 [OSDI'24 paper](https://www.usenix.org/conference/osdi24/presentation/agrawal) 论文。

本项目基于 Sarathi-Serve 做更进一步的优化。

---

## 工作

### 适配模型

- Qwen3/Qwen3-8B
- Qwen/Qwen-7B

### 测试的数据集

- [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main)

---

## 运行

### CUDA

本项目基于 CUDA 12.8 on NVIDIA 3090 and 4090.

### Clone 项目

```sh
git clone https://github.com/Charmstok/sarathi-serve.git
```

### conda 环境

创建 python 3.11 环境

```sh
conda create -p ./env python=3.11  
```

激活创建的环境，

```sh
conda activate ./env
```

### 安装本项目需要的依赖以及 cuda 环境

```sh
pip install -r requirements.txt
pip install -e .
```

### 额外适配的模型

- Qwen3
- openPangu(https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1)

### 如何使用本项目？

#### 方式一: 简单对话

```shell
python example/chat/chat_only.py
```

#### 方式二: 离线测试

基准测试得到的各项指标可以在 offline_inference_output 目录下查看.

```shell
python example/offline_inference/target_time.py
```

使用 target_time 前，请先按照以下说明获取时间预测模型：

运行离线脚本：

```shell
python example/time_balance/offline_select_status_csv.py
```

它会用 `SarathiScheduler` 跑一批请求，并在模型执行阶段写出 `select_stats_rank0.csv`：

> 说明：
> - 脚本输出目录在 `offline_inference_output/<时间戳>/replica_0/`。
> - 关键产物是 `select_stats_rank0.csv`（包含 `decode_tokens/prefill_tokens/.../latency_ms` 等特征与标签）。
> - 若你改了脚本里的 `chunk_size/max_num_seqs/max_model_len` 等参数，建议在训练配置里保持一致（尤其是 `chunk_size`）。

接下来，打开 `sarathi/time_balance/config.py`，按你刚生成的数据修改：
- `CSV_PATH`：指向你最新目录下的 `select_stats_rank0.csv`
- `MODEL_CACHE_PATH`：模型保存路径（默认在 `sarathi/time_balance/time_predictor_mlp_v6.pt`）
- `BUCKET_SPLIT_CONFIG.chunk_size`：建议与数据采集时的 scheduler `chunk_size` 一致（默认 256）

执行：

```sh
python sarathi/time_balance/predict_time.py
```

训练完成后会输出 train/val/test 的 MAE，并将模型写入 `MODEL_CACHE_PATH`。

`OptSarathiScheduler` 会在初始化时从 `sarathi/time_balance/config.py` 的 `MODEL_CACHE_PATH` 加载模型；如果文件不存在会直接 `assert` 报错（确保线上不会静默退化）。

可用下面脚本快速验证模型能否正确加载并预测：

```sh
python sarathi/time_balance/load_model.py
```

#### 方式三: openai_entrypoints

```shell
python -m sarathi.entrypoints.openai.api_server \
    --model_config_model Qwen/Qwen-7B \
    --model_config_max_model_len 1024 \
    --worker_config_gpu_memory_utilization 0.9
```

以上的运行命令没有涵盖所有参数, 你可以运行下列 sh 脚本获取更多帮助，

```sh
python -m sarathi.entrypoints.api_server --help
```

### 调参脚本

如果你想要更高效的性能, 可能需要调整某些参数.

详见, [调整参数 README.md](example/README.md)

---

## 致谢

本仓库源自 [sarathi-serve](https://github.com/microsoft/sarathi-serve) 项目 的一个分支。 本项目仅为研究原型，我们仅保留了原项目最关键的功能，并对代码进行了精简，以便更快地进行研究迭代。