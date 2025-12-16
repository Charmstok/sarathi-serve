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

### 如何使用本项目？

#### 方式一: 简单对话

```shell
python example/chat_only.py
```

#### 方式二: 离线测试

基准测试得到的各项指标可以在 offline_inference_output 目录下查看.

```shell
python example/offline_inference.py
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

详见, [调整参数 README.md](example/parameter_selection/README.md)

---

## 致谢

本仓库源自 [sarathi-serve](https://github.com/microsoft/sarathi-serve) 项目 的一个分支。 本项目仅为研究原型，我们仅保留了原项目最关键的功能，并对代码进行了精简，以便更快地进行研究迭代。