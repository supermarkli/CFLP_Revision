# CFLP 联邦模拟平台

## 1. 项目概述

本项目是机密联邦学习平台（Confidential Federated Learning Platform, CFLP）的联邦模拟平台部分，一个用于学术研究和实验的机器学习框架，旨在方便地比较和评估多种机器学习模型在**集中式学习（Centralized Learning）**和**联邦学习（Federated Learning）**两种不同范式下的性能表现。

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
│   ├── experiment_results_bar.png  # 实验结果可视化柱状图
│   ├── experiment_state.json  # 批量实验断点续跑状态文件
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

> **联邦数据分布可选值**  
> 通过在 `src/default.yaml` 的 `data.federated_dist` 字段中设置：  
> • `iid` – 客户端数据独立同分布（默认）  
> • `noniid_label_skew` – 按标签异质分布  
> • `noniid_quantity_skew` – 按样本数量异质分布

### 5.2. 一键运行所有实验

最简单的使用方式是运行根目录下的 `run_all_experiments.py` 脚本。它会自动遍历所有定义的模型和学习模式，执行全部实验，并将结果保存在 `out` 目录下。

```bash
python run_all_experiments.py
```
实验完成后，您可以在 `out` 目录查看 `experiment_results.csv` 与 `experiment_results_bar.png` 进行分析；
若需重新绘制柱状图，可直接执行：
```bash
python -m src.utils.draw
```

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

> **提示**：联邦学习模式目前仅支持深度学习模型 (`CNN`, `MLP`)。若在 `Federated` 模式下选择传统机器学习模型 (`KNN`, `RF`, `SVC`, `LR`)，脚本会自动跳过或报错提示。

## 6. 配置文件说明

项目的主要配置均在 `src/default.yaml` 中定义，常用字段示例如下：

- `mode`：`Centralized` / `Federated`
- `seed`：随机种子，保证结果可复现
- `data`：
  - `complete` / `clients`：数据目录
  - `federated_dist`：联邦数据分布，可选 `iid` / `noniid_label_skew` / `noniid_quantity_skew`
- `federated`：
  - `aggregation`：聚合算法（目前为 `fedavg`）
  - `num_clients`：客户端总数
  - `local_epochs`：客户端本地训练轮数
  - `rounds`：联邦总迭代轮数
- `model`：模型相关超参数（见文件内各子节点）
- `training`：
  - `batch_size`, `learning_rate`, `eval_batch_size`
  - `epochs`（集中式） / `rounds`（联邦）
  - `converge_threshold`：提前停止的收敛阈值

## 7. 实验输出

所有实验的产出都位于 `out` 目录中，方便集中查看：
- `experiment_results.csv`：每次实验的模式、模型、最终准确率、AUC 和训练耗时。
- `experiment_results_bar.png`：汇总柱状图，直观展示不同实验组合下的性能对比。
- `experiment_state.json`：批量实验进度记录，可在意外中断后自动续跑。
- `[mode]_[model][_分布]_run_N.log`：单次实验日志文件（联邦模式包含数据分布与运行序号）。
- `batch.log`：`run_all_experiments.py` 脚本本身的运行日志。

> **说明**：当测试集类别不完整时，AUC 可能无法计算，CSV 中会显示 `N/A`，日志会标记为 `pass`。

## 8. 预设实验组合

项目默认在 `run_all_experiments.py` 中一键运行 12 组实验，方便横向/纵向全面对比。

| 序号 | 类型 | 模型 | 数据分布 |
|-----|------|------|-----------|
| 1 | CL | CNN | 完整数据 |
| 2 | CL | MLP | 完整数据 |
| 3 | CL | KNN | 完整数据 |
| 4 | CL | Random&nbsp;Forest | 完整数据 |
| 5 | CL | SVC | 完整数据 |
| 6 | CL | Logistic&nbsp;Regression | 完整数据 |
| 7 | FL | CNN | iid |
| 8 | FL | CNN | noniid_label_skew |
| 9 | FL | CNN | noniid_quantity_skew |
| 10 | FL | MLP | iid |
| 11 | FL | MLP | noniid_label_skew |
| 12 | FL | MLP | noniid_quantity_skew |
 
 > 说明：
 > * CL = Centralized Learning；FL = Federated Learning（FedAvg 算法）。
 > * noniid_label_skew：客户端仅包含部分类别；noniid_quantity_skew：客户端样本量不均衡。
 > * 传统 ML 模型暂不在 FL 中运行，故 FL 仅包含深度模型 (CNN, MLP)。

### 8.1 集中式学习（CL）详解

| 序号 | 实验代号 | 主要参数 | 实验目的 |
|------|---------|---------|---------|
| 1 | **CL-CNN** | `FedAvgCNN`, Adam(lr=1e-3), Epochs=100 | 作为深度卷积网络在整数据集上的「上限」基线，后续对比 FL-CNN 准确率损失。 |
| 2 | **CL-MLP** | MLP(784-128-64-10), ReLU | 比较浅层全连接网络与 CNN 的差距，验证特征抽取能力影响。 |
| 3 | **CL-KNN** | k=5, uniform | KNN 属于惰性学习，无显式训练过程；用于观察实例-based 方法在高维向量化 MNIST 上的效果及推理耗时。 |
| 4 | **CL-RF** | n_estimators=100, max_depth=None | 评估基于树的集成方法对图像灰度特征的非线性拟合能力。 |
| 5 | **CL-SVC** | C=1.0, RBF 核 | 大 margin 分类器，考察其在 60k 样本+10 类任务上的准确率与训练耗时。 |
| 6 | **CL-LR** | LogisticRegression, lbfgs | 线性模型最低基线，帮助判断数据是否近似线性可分。 |

### 8.2 联邦学习（FL）详解

> 全局参数：客户端数 `num_clients=3`，本地训练轮 `local_epochs=1`，FedAvg 聚合，通信在每轮后进行。

| 序号 | 实验代号 | 数据分布 | 主要关注点 |
|------|---------|---------|-----------|
| 7 | **FL-CNN-iid** | iid | 在理想同分布情况下，FL-CNN 与 CL-CNN 的性能差距 → 衡量通信/分布式噪声的影响。 |
| 8 | **FL-CNN-label** | noniid_label_skew | 每客户端仅含子集类别，考察标签异质性对 FedAvg 收敛速度与最终准确率的影响。 |
| 9 | **FL-CNN-quantity** | noniid_quantity_skew | 客户端样本量差异大，研究客户端权重不均衡导致的性能波动。 |
| 10 | **FL-MLP-iid** | iid | 浅层网络在 FL 场景的基线，对比 7 了解网络容量对联邦性能的作用。 |
| 11 | **FL-MLP-label** | noniid_label_skew | 与 8 相同分布，换用 MLP；探究模型复杂度与标签异质性的交互效应。 |
| 12 | **FL-MLP-quantity** | noniid_quantity_skew | 与 9 相同分布，换用 MLP；评估数据量失衡对浅层网络的影响。 |

每组实验默认重复 **3 次**（不同随机种子 0/1/2），`run_all_experiments.py` 会统计均值±标准差并绘制柱状图，指标包括：

* **准确率 (Accuracy)**：主要衡量分类正确率。
* **AUC**：对多类别任务使用 `ovr`/`ovo` 策略，若类别不足则显示 *N/A*。
* **训练总耗时**：便于分析通信与算法复杂度差异。

通过上述 12 组组合，你可以：

1. 横向比较不同模型之间的效果差异；
2. 纵向比较同一模型在 CL vs. FL 的性能损失；
3. 分析 IID 与两种典型 Non-IID (标签偏斜 / 数量失衡) 对联邦训练的影响；
4. 评估模型复杂度（CNN vs. MLP）在异质分布下的稳健性。