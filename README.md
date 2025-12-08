# Opt-Sarathi-Serve

Sarathi-Serve 是一个高吞吐、低延迟的大模型推理框架，技术细节详见的 [OSDI'24 paper](https://www.usenix.org/conference/osdi24/presentation/agrawal) 论文。
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
pip install -e .
```

### 如何使用本项目？

#### 方式一，离线测试

```shell
python examples/offline_inference.py
```

配置文件详见：`sarathi/config/config.py`

#### 方式二，openai_entrypoints

```shell
python -m sarathi.entrypoints.openai.api_server \
    --model_config_model Qwen/Qwen-7B \
    --model_config_max_model_len 1024 \
    --worker_config_gpu_memory_utilization 0.9
```

获取更多帮助，

```sh
python -m sarathi.entrypoints.api_server --help
```

---

## 致谢

本仓库最初源自 [sarathi-serve](https://github.com/microsoft/sarathi-serve) 项目 的一个分支。 本项目仅为研究原型，并未与开源版 sarathi-serve 保持功能完全对等。
我们仅保留了最关键的功能，并对代码进行了精简，以便更快地进行研究迭代。