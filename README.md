# CFLP 联邦学习仿真平台

## 1. 项目概述

本项目是一个面向学术研究的联邦学习仿真平台（Federated Learning Simulation Platform），旨在为研究人员提供一个便捷、可扩展的实验框架，用于对比和评估不同学习范式和数据分布条件下的模型性能。

### 1.1 设计目标

- **范式对比**：直接对比**集中式学习（Centralized Learning）**与**联邦学习（Federated Learning）**在相同数据集和模型架构下的性能差异
- **数据异质性研究**：支持多种数据分布策略（IID 与 Non-IID），帮助研究者分析数据异质性对联邦学习收敛性和模型精度的影响
- **自动化实验流程**：提供一键式批量实验脚本，自动完成实验执行、结果收集、统计分析和可视化
- **可复现性**：通过随机种子控制和配置文件管理，确保实验结果可复现

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| 多数据集支持 | MNIST、CIFAR-10（可扩展至其他数据集） |
| 多模型架构 | CNN（轻量级）、ResNet18（深度网络）、MLP、及传统 ML 模型 |
| 多数据分布 | IID、Label Skew、Quantity Skew、Dirichlet 分布 |
| 丰富的评估指标 | 准确率、AUC、收敛轮数、通信开销、Local-Global Gap 等 |

## 2. 平台架构

### 2.1 整体设计

平台采用模块化设计，将训练逻辑、模型定义、数据处理和实验管理解耦，便于扩展和维护：

```
CFLP_Revision/
├── data/                      # 数据集目录（自动生成）
│   ├── complete/              # 完整数据集（用于集中式训练和全局测试）
│   ├── client1/               # 客户端1的本地数据
│   ├── client2/               # 客户端2的本地数据
│   └── client3/               # 客户端3的本地数据
├── out/                       # 实验输出目录（自动生成）
│   ├── experiment_results.csv # 实验结果汇总表
│   ├── convergence_combined.png # 收敛曲线可视化
│   ├── experiment_state.json  # 断点续跑状态文件
│   └── *.log                  # 各实验的详细日志
├── src/                       # 核心源代码
│   ├── clients/               # 联邦学习客户端实现
│   ├── servers/               # 联邦学习服务端实现
│   ├── trainers/              # 训练器（集中式、联邦式、传统ML）
│   ├── models/                # 模型定义（深度学习 + 传统ML）
│   ├── data_process/          # 数据预处理和划分脚本
│   ├── utils/                 # 工具函数（日志、绘图等）
│   ├── default.yaml           # 默认配置文件
│   └── main.py                # 单次实验入口
├── run_all_experiments.py     # 批量实验脚本
├── requirements.txt           # Python 依赖
└── README.md                  # 项目说明
```

### 2.2 训练器层次结构

平台采用面向对象的训练器设计，通过抽象基类统一接口：

- **BaseTrainer**：抽象基类，定义训练和评估的通用接口，提供收敛检测、最佳模型保存等基础功能
- **CentralizedTrainer**：集中式训练器，在完整数据集上进行标准的深度学习训练
- **FederatedTrainer**：联邦训练器，协调客户端-服务器交互，实现 FedAvg 聚合流程
- **MLTrainer**：传统机器学习模型训练器，封装 scikit-learn 模型的训练和评估逻辑

### 2.3 联邦学习架构

联邦学习模块采用经典的客户端-服务器架构：

- **Client（客户端）**：持有本地数据，执行本地模型训练，支持数据增强和混合精度训练
- **Server（服务器）**：管理全局模型，负责参数分发和聚合，计算通信开销等指标

**训练流程**：
1. 服务器将全局模型参数分发给所有客户端
2. 各客户端在本地数据上进行多轮训练
3. 客户端将更新后的模型参数上传到服务器
4. 服务器使用 FedAvg 算法聚合模型参数
5. 重复以上步骤直到收敛或达到最大轮数

### 2.4 模型架构

平台内置多种模型，覆盖从轻量级到深度网络的不同复杂度：

| 模型 | 适用场景 | 说明 |
|------|----------|------|
| FedAvgCNN | MNIST 等简单图像 | 轻量级两层卷积网络，参数量小，训练快速 |
| ResNet18 | CIFAR-10 等复杂图像 | 适配小图像输入的 ResNet18 变体，具有残差连接 |
| MLP | 通用基线 | 可配置隐藏层的多层感知机 |
| 传统 ML | 集中式对比 | KNN、随机森林、SVC、逻辑回归（仅支持集中式训练） |

## 3. 数据分布策略

平台支持多种数据分布方式，用于模拟真实联邦学习场景中的数据异质性：

### 3.1 IID（独立同分布）

将训练数据随机均匀地分配给各客户端，每个客户端的数据分布与全局分布一致。这是最理想的场景，作为基准对照。

### 3.2 Non-IID Label Skew（标签倾斜）

不同客户端仅持有部分类别的数据。例如：
- 客户端1：类别 0、1、2
- 客户端2：类别 3、4、5
- 客户端3：类别 6、7、8、9

这种分布模拟了数据在不同用户间自然聚集的场景。

### 3.3 Non-IID Quantity Skew（数量倾斜）

各客户端持有所有类别的数据，但数据量差异显著。例如：
- 客户端1：60% 的数据
- 客户端2：30% 的数据
- 客户端3：10% 的数据

这种分布模拟了参与方数据规模差异大的场景。

### 3.4 Non-IID Dirichlet（狄利克雷分布）

使用 Dirichlet 分布控制每个类别在各客户端之间的分配比例。参数 α 控制异质程度：
- α → 0：数据极度倾斜，每个客户端主要持有少数类别
- α → ∞：接近 IID 分布

这是学术研究中最常用的 Non-IID 建模方式，可以灵活调整异质程度。

## 4. 实验指标

平台收集以下关键指标用于实验分析：

| 指标 | 说明 |
|------|------|
| **准确率（Accuracy）** | 模型在全局测试集上的分类正确率 |
| **AUC** | 多分类 ROC-AUC（OVR 策略），衡量模型的排序能力 |
| **收敛轮数** | 模型达到收敛所需的训练轮数（基于准确率波动阈值判定） |
| **通信体积** | 联邦学习中客户端与服务器之间的总数据传输量 |
| **Local-Global Gap** | 全局模型与本地模型在全局测试集上的准确率差距，衡量数据异质性影响 |
| **最后N轮准确率标准差** | 评估训练过程稳定性，标准差越小收敛越稳定 |

## 5. 安装与配置

### 5.1 环境要求

- **Python**: 3.8+（推荐 3.9 或 3.10）
- **PyTorch**: 2.0+（建议使用 GPU 版本以加速训练）
- **CUDA**: 11.8+ 或 12.1+（可选，用于 GPU 加速）
- **混合精度训练**: 需要支持 AMP 的 GPU（如 V100, A100, H100 等）

### 5.2 核心依赖

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

### 5.3 安装步骤

1. **克隆项目**
   ```bash
   git clone <your-repository-url>
   cd CFLP_Revision
   ```

2. **创建虚拟环境（推荐）**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **安装 PyTorch**
   
   根据您的系统和 CUDA 版本，参考 [PyTorch 官网](https://pytorch.org/) 安装：
   ```bash
   # 示例：CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # 或仅 CPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

## 6. 使用说明

### 6.1 数据准备

首次运行前需要生成实验数据。数据处理脚本会自动下载原始数据集并生成各种分布的客户端数据：

```bash
# 生成 MNIST 数据（包含 IID 和所有 Non-IID 划分）
python src/data_process/generate_mnist_data.py

# 生成 CIFAR-10 数据（包含 IID 和所有 Non-IID 划分）
python src/data_process/generate_cifar10_data.py
```

生成的数据结构：
```
data/
├── complete/              # 完整数据集
│   ├── mnist_train.npz    # 训练集（集中式训练用）
│   ├── mnist_test.npz     # 测试集（全局评估用）
│   └── ...
└── clientX/               # 各客户端数据
    ├── mnist_train.npz    # IID 训练数据
    ├── mnist_train_noniid_dirichlet.npz  # Dirichlet Non-IID
    └── ...
```

### 6.2 数据归一化与增强

**CIFAR-10 归一化参数**（标准值）：
```python
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
```

**数据增强**（仅在训练时应用）：
```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 随机裁剪
    transforms.RandomHorizontalFlip(),      # 随机水平翻转（概率 0.5）
])
```

> ⚠️ 数据增强可以显著提升模型泛化能力，缺失会导致准确率下降 10-15%

### 6.2 一键运行全部实验

使用批量实验脚本执行预设的实验组合：

```bash
python run_all_experiments.py
```

该脚本会：
1. 自动遍历所有预设实验配置
2. 为每个实验执行多次重复运行（默认5次）
3. 收集和解析日志中的实验指标
4. 计算均值和标准差
5. 生成结果汇总表和可视化图表
6. 支持断点续跑（意外中断后可恢复）

### 6.3 运行单次实验

如需运行特定实验，可修改配置文件后直接执行：

1. **修改配置** `src/default.yaml`：
   ```yaml
   mode: 'Federated'          # 或 'Centralized'
   data:
     dataset: cifar10         # 或 'mnist'
     federated_dist: noniid_dirichlet  # IID/Non-IID 分布类型
   model:
     type: ResNet18           # 模型类型
   ```

2. **运行实验**：
   ```bash
   python src/main.py
   ```

### 6.4 重新生成可视化图表

如需单独重新绘制收敛曲线图：

```bash
python -m src.utils.draw
```

## 7. 配置文件说明

`src/default.yaml` 是主要配置文件，关键参数说明：

```yaml
# ========== 基础配置 ==========
mode: 'Federated'              # 训练模式：'Centralized' 或 'Federated'
seed: 0                        # 随机种子（确保可复现）

# ========== 数据配置 ==========
data:
  dataset: cifar10             # 数据集：'mnist' 或 'cifar10'
  federated_dist: noniid_dirichlet  # 联邦数据分布类型
  # 可选值：'iid', 'noniid_label_skew', 'noniid_quantity_skew', 'noniid_dirichlet'

# ========== 模型配置 ==========
model:
  type: ResNet18               # 模型类型：'CNN', 'ResNet18', 'MLP', 'KNN', 'RF', 'SVC', 'LR'

# ========== 联邦学习配置 ==========
federated:
  num_clients: 3               # 客户端数量
  local_epochs: 5              # 每轮本地训练轮数
  rounds: 10000                # 最大通信轮数（依靠收敛检测自动停止）
  aggregation: fedavg          # 聚合算法

# ========== 训练配置 ==========
training:
  batch_size: 128              # 批次大小
  learning_rate: 0.01          # 学习率
  optimizer: sgd               # 优化器：'sgd' 或 'adam'
  momentum: 0.9                # 动量系数（SGD 配置）
  weight_decay: 0.0005         # L2 正则化系数
  scheduler: cosine            # 学习率调度器：'cosine', 'multistep' 或 null
  converge_threshold: 0.005    # 收敛判定阈值
  use_amp: true                # 是否启用混合精度训练（仅 GPU）
  use_augmentation: true       # 是否启用数据增强
```

> ⚠️ **重要**：联邦学习建议使用 **SGD** 而非 Adam。Adam 在联邦场景下由于状态管理问题效果较差。

## 8. 预设实验组合

平台默认配置了 6 组实验，覆盖轻量级（MNIST）和重量级（CIFAR-10）两种场景：

### 8.1 联邦学习实验

| 实验代号 | 数据集 | 模型 | 数据分布 | 研究目的 |
|----------|--------|------|----------|----------|
| Sim-1 | MNIST | CNN | IID | 轻量级任务的联邦学习基准 |
| Sim-2 | MNIST | CNN | Dirichlet Non-IID | 数据异质性对简单任务的影响 |
| Sim-3 | CIFAR-10 | ResNet18 | IID | 复杂任务的联邦学习基准 |
| Sim-4 | CIFAR-10 | ResNet18 | Dirichlet Non-IID | 数据异质性对复杂任务的影响 |

### 8.2 集中式对照实验

| 实验代号 | 数据集 | 模型 | 研究目的 |
|----------|--------|------|----------|
| Central-MNIST | MNIST | CNN | Sim-1/2 的集中式上限基准 |
| Central-CIFAR10 | CIFAR-10 | ResNet18 | Sim-3/4 的集中式上限基准 |

通过对比联邦实验与对应的集中式实验，可以量化联邦学习带来的性能损失。

## 9. 预期效果

使用默认配置，CIFAR-10 + ResNet18 联邦学习预期能达到：

| 指标 | 预期值 |
|------|--------|
| 准确率 | ~95%+ |
| AUC | ~0.99+ |

**训练曲线特征**：
- Round 1-10：快速上升期（准确率 40% → 80%）
- Round 10-50：稳定提升期（准确率 80% → 92%）
- Round 50+：精细优化期（准确率 92% → 95%+）

## 10. 实验输出

所有实验产出保存在 `out/` 目录：

| 文件 | 说明 |
|------|------|
| `experiment_results.csv` | 各实验的指标汇总（均值±标准差） |
| `convergence_combined.png` | MNIST 和 CIFAR-10 的收敛曲线对比图 |
| `experiment_state.json` | 实验进度状态（用于断点续跑） |
| `[Mode]_[Model]_[Dist]_run_N.log` | 单次实验的详细日志 |
| `batch.log` | 批量实验脚本运行日志 |

## 11. 扩展指南

### 11.1 添加新数据集

1. 在 `src/data_process/` 下创建数据生成脚本
2. 在 `src/main.py` 的 `get_dataset_info()` 中添加数据集配置
3. 更新 `default.yaml` 中的 `data.dataset` 可选值

### 11.2 添加新模型

1. 在 `src/models/models.py` 中定义模型类
2. 在 `src/main.py` 的 `create_model()` 中添加创建逻辑
3. 确保模型实现 `get_parameters()` 和 `set_parameters()` 方法（联邦学习需要）

### 11.3 添加新聚合算法

1. 在 `src/servers/server.py` 中实现新的聚合方法
2. 通过配置文件的 `federated.aggregation` 字段选择算法

## 12. 常见问题

### 为什么推荐 SGD 而非 Adam？
Adam 在联邦学习中由于状态管理问题，效果较差。SGD + Momentum 0.9 是联邦学习的标准配置。

### 数据增强为什么重要？
数据增强可以显著提升模型泛化能力。缺失数据增强会导致准确率下降 10-15%。

### 学习率调度器的作用？
CosineAnnealingLR 帮助模型在训练后期精细优化。固定学习率会导致后期收敛不佳。

### 如何调整客户端数量？
修改 `federated.num_clients`，同时需要重新生成对应数量的客户端数据。更多客户端通常需要更多轮数才能收敛。

## 13. 许可证

本项目仅供学术研究使用。

## 14. 引用

如果本项目对您的研究有帮助，请考虑引用相关论文或本仓库。
