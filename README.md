# 基于联邦学习与集中式学习的机器学习模型性能对比框架

## 1. 项目概述

本项目是一个用于学术研究和实验的机器学习框架，旨在方便地比较和评估多种机器学习模型在**集中式学习（Centralized Learning）**和**联邦学习（Federated Learning）**两种不同范式下的性能表现。

框架通过自动化脚本，能够针对不同的模型和学习范式执行一系列实验，并自动收集、整理和可视化实验结果，如模型的准确率（Accuracy）、AUC值以及训练总耗时，极大地简化了实验流程和结果分析过程。

## 2. 主要特性

- **双模式支持**: 同时支持**集中式**和**联邦式**两种学习模式，方便直接对比。
- **多模型集成**: 集成了多种经典的机器学习和深度学习模型：
  - **深度学习模型**: `CNN`, `MLP`
  - **传统机器学习模型**: `KNN`, `Random Forest (RF)`, `SVC`, `Logistic Regression (LR)`
- **自动化实验**: 提供一键运行所有预设实验的脚本 (`run_all_experiments.py`)。
- **自动化报告**: 实验完成后，自动生成 `CSV` 格式的详细结果报告和可视化的性能对比柱状图。
- **灵活配置**: 所有实验参数（如学习率、迭代次数、客户端数量等）均可通过 `src/default.yaml` 文件进行灵活配置。
- **清晰的日志**: 为每次独立的实验生成详细的日志，方便追溯和调试。

## 3. 项目结构

```
CFLP_Revision/
├── data/                  # 数据集目录
│   ├── client1/           # 联邦学习客户端1的数据
│   ├── client2/           # 联邦学习客户端2的数据
│   ├── ...
│   └── complete/          # 集中式学习使用的完整数据集
├── out/                   # 实验输出目录 (自动生成)
│   ├── experiment_results.csv  # 实验结果汇总
│   ├── results.png        # 实验结果可视化图表
│   └── *.log              # 每次实验的详细日志
├── src/                   # 核心源代码
│   ├── clients/           # 客户端实现
│   ├── data_process/      # 数据处理脚本
│   ├── models/            # 模型定义
│   ├── servers/           # 服务端实现
│   ├── trainers/          # 训练器 (定义不同学习模式的训练逻辑)
│   ├── utils/             # 工具函数 (日志, 绘图等)
│   ├── default.yaml       # 默认配置文件
│   └── main.py            # 单次实验的主入口
├── README.md              # 项目说明
├── requirements.txt       # Python 依赖包
└── run_all_experiments.py # 自动化实验运行脚本
```

## 4. 安装与环境配置

1.  **克隆项目**
    ```bash
    git clone <your-repository-url>
    cd CFLP_Revision
    ```

2.  **创建虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Windows, use `venv\Scripts\activate`
    ```

3.  **安装依赖**
    项目所需的所有依赖库都已在 `requirements.txt` 文件中列出。运行以下命令进行安装：
    ```bash
    pip install -r requirements.txt
    ```
    *注意：`PyTorch` 的安装可能因您的 `CUDA` 版本而异。如果遇到问题，请参考 [PyTorch官网](https://pytorch.org/) 的指导进行安装。*

## 5. 使用说明

### 5.1. 数据准备

项目使用 MNIST 数据集。您可以运行 `src/data_process/generate_mnist_data.py` 来自动下载并切分数据，为集中式和联邦式学习准备所需的数据文件。
```bash
python src/data_process/generate_mnist_data.py
```
该脚本会创建 `data/complete`（完整数据）和 `data/clientN`（客户端数据）目录。

### 5.2. 一键运行所有实验

最简单的使用方式是运行根目录下的 `run_all_experiments.py` 脚本。它会自动遍历所有定义的模型和学习模式，执行全部实验，并将结果保存在 `out` 目录下。

```bash
python run_all_experiments.py
```
实验完成后，您可以在 `out` 目录查下 `experiment_results.csv` 和 `results.png` 文件来分析结果。

### 5.3. 运行单次指定实验

如果您想运行特定的实验（例如，只测试联邦学习模式下的CNN模型），可以手动修改 `src/default.yaml` 文件，然后直接运行 `src/main.py`。

1.  **修改配置**: 打开 `src/default.yaml`，根据需要修改 `mode` 和 `model.type` 等参数。
    ```yaml
    # 示例: 配置为联邦学习模式下的CNN模型
    mode: 'Federated'
    model:
      type: 'CNN'
      # ... 其他模型参数
    ```

2.  **运行脚本**:
    ```bash
    python src/main.py
    ```
    该次实验的日志会保存在 `logs` 目录下（运行 `run_all_experiments.py` 时会自动移动到 `out` 目录）。

## 6. 配置文件说明

项目的主要配置均在 `src/default.yaml` 中定义，主要参数包括：

- `mode`: 实验模式，可选值为 `'Centralized'` 或 `'Federated'`。
- `seed`: 全局随机种子，用于保证实验可复现。
- `data`: 数据路径配置。
- `federated`: 联邦学习相关参数，如客户端数量、每轮抽样的客户端比例等。
- `model`: 模型配置，包含模型类型 (`type`) 和各个模型的具体超参数。
- `training`: 训练过程的超参数，如 `batch_size`, `epochs` (用于集中式), `rounds` (用于联邦式) 和 `lr` (学习率)。

## 7. 实验输出

所有实验的产出都位于 `out` 目录中，方便集中查看：
- `experiment_results.csv`: 包含了每次实验的模式、模型、最终准确率、AUC和训练耗时。
- `results.png`: 柱状图，直观展示不同实验组合下的性能对比。
- `[mode]_[model].log`: 对应实验的完整日志文件，记录了详细的训练过程。
- `batch.log`: `run_all_experiments.py` 脚本本身的运行日志。