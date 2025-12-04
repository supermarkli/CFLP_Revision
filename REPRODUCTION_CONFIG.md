# 联邦学习完整复现配置清单

本文档提供完整的配置信息，用于在其他平台复现当前的联邦学习效果（准确率 ~95.3%，AUC ~0.997）。

## 📋 目录

1. [环境配置](#环境配置)
2. [数据处理配置](#数据处理配置)
3. [模型配置](#模型配置)
4. [训练配置](#训练配置)
5. [联邦学习配置](#联邦学习配置)
6. [完整配置文件](#完整配置文件)

---

## 🔧 环境配置

### Python 版本
- **Python**: 3.8+ (推荐 3.9 或 3.10)

### 核心依赖包
```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

### GPU 配置（可选但推荐）
- **CUDA**: 11.8+ 或 12.1+
- **cuDNN**: 与 CUDA 版本匹配
- **混合精度训练**: 需要支持 AMP 的 GPU（如 V100, A100, H100 等）

---

## 📊 数据处理配置

### 数据集：CIFAR-10

#### 数据加载参数
```python
# 数据归一化参数（CIFAR-10 标准值）
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# 数据划分
test_size = 0.2  # 每个客户端测试集占比 20%
random_state = 42  # 随机种子，保证可复现
```

#### 数据分布：IID（独立同分布）
- **分布类型**: `iid`
- **客户端数量**: `3`
- **划分方式**: 随机打乱后均匀分配给 3 个客户端
- **每个客户端**: 训练集 80%，测试集 20%

#### 数据增强（训练时）
```python
# 仅在训练时应用，评估时不使用
transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪（32x32，padding=4）
    transforms.RandomHorizontalFlip(),      # 随机水平翻转（概率 0.5）
])
```

**注意**：
- 数据增强仅在客户端训练时应用
- 评估时使用原始数据（无增强）
- 仅对 32x32 图像（CIFAR-10/100）应用

---

## 🏗️ 模型配置

### 模型类型：ResNet18（适用于 CIFAR-10）

#### 模型结构
```python
class ResNet18(nn.Module):
    def __init__(self, in_features=3, num_classes=10):
        # 输入通道数: 3 (RGB)
        # 输出类别数: 10 (CIFAR-10)
        
        # 第一层卷积
        conv1: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1: BatchNorm2d(64)
        
        # 残差块层
        layer1: BasicBlock x 2 (64 channels)
        layer2: BasicBlock x 2 (128 channels, stride=2)
        layer3: BasicBlock x 2 (256 channels, stride=2)
        layer4: BasicBlock x 2 (512 channels, stride=2)
        
        # 全连接层
        linear: Linear(512, 10)
        
        # 激活函数: ReLU
        # 池化: AdaptiveAvgPool2d(1, 1)
```

#### 模型初始化
- **输入尺寸**: `(batch_size, 3, 32, 32)`
- **输出尺寸**: `(batch_size, 10)`
- **参数量**: ~11.2M

---

## 🎯 训练配置

### 优化器：SGD（随机梯度下降）

```yaml
optimizer: sgd
learning_rate: 0.1          # 初始学习率
momentum: 0.9              # 动量系数
weight_decay: 0.0005       # L2 正则化系数
```

**关键点**：
- **必须使用 SGD**，不要使用 Adam（在联邦学习中效果较差）
- 学习率 0.1 是 SGD 的推荐值（Adam 需要 0.001）
- Momentum 0.9 是标准配置

### 学习率调度器：CosineAnnealingLR

```yaml
scheduler: cosine
```

**调度器参数**：
```python
# 总训练步数 = federated_rounds * local_epochs
T_max = 100 * 3 = 300

# 学习率变化公式
lr(t) = lr_min + (lr_max - lr_min) * (1 + cos(π * t / T_max)) / 2

# 其中：
# lr_max = 0.1 (初始学习率)
# lr_min = 0 (最小学习率)
# t = 当前步数 (0 到 T_max)
```

**学习率变化示例**：
- Round 1: ~0.1
- Round 50: ~0.05
- Round 100: ~0.0

### 损失函数

```python
criterion = nn.CrossEntropyLoss()
```

### 数据加载配置

```yaml
batch_size: 128           # 训练批次大小
eval_batch_size: 512     # 评估批次大小（更大以加速评估）
num_workers: 4           # DataLoader 多进程数（GPU 模式）
prefetch_factor: 2       # 数据预取批次数
shuffle: true            # 训练时打乱数据
```

### 混合精度训练（AMP）

```yaml
use_amp: true  # 启用混合精度训练（仅 GPU 模式）
```

**说明**：
- 使用 `torch.amp.autocast` 进行前向传播
- 使用 `GradScaler` 进行梯度缩放
- 可提升训练速度约 1.5-2 倍，几乎不影响精度

### 数据增强

```yaml
use_augmentation: true
```

**增强策略**：
- `RandomCrop(32, padding=4)`: 随机裁剪，padding=4
- `RandomHorizontalFlip()`: 随机水平翻转（概率 0.5）

---

## 🌐 联邦学习配置

### 联邦学习参数

```yaml
federated:
  aggregation: fedavg           # 聚合算法：FedAvg（简单平均）
  rounds: 100                   # 联邦学习总轮数
  local_epochs: 3               # 每轮每个客户端本地训练轮数
  num_clients: 3                # 客户端数量
```

### 聚合算法：FedAvg

```python
# 简单平均聚合
def aggregate(client_parameters_list):
    for key in parameters.keys():
        stacked = torch.stack([params[key] for params in client_parameters_list], dim=0)
        aggregated[key] = torch.mean(stacked, dim=0)  # 简单平均
```

**关键点**：
- 所有客户端权重相等（简单平均）
- 每轮联邦学习后，服务器聚合所有客户端的模型参数
- 聚合后的全局模型分发给所有客户端

### 训练流程

```
For round = 1 to 100:
    1. 服务器分发全局模型给所有客户端
    2. 每个客户端：
       - 接收全局模型参数
       - 本地训练 3 个 epoch（使用本地数据 + 数据增强）
       - 更新学习率（如果使用调度器）
    3. 服务器收集所有客户端的模型参数
    4. 服务器聚合参数（FedAvg 平均）
    5. 评估聚合后的全局模型
```

### 收敛检测

```yaml
converge_threshold: 0.0001  # 收敛阈值
```

**检测逻辑**：
- 记录最近 4 轮的准确率
- 如果 `max(最近4轮) - min(最近4轮) < 0.0001`，则提前停止

---

## 📝 完整配置文件

### `default.yaml`（完整版）

```yaml
# ============================================
# 全局配置
# ============================================
mode: Federated
seed: 42
log_dir: logs/

# ============================================
# 数据配置
# ============================================
data:
  dataset: cifar10
  clients:
  - data/client1/
  - data/client2/
  - data/client3/
  complete: data/complete/
  federated_dist: iid  # IID 分布

# ============================================
# 模型配置
# ============================================
model:
  type: ResNet18

# ============================================
# 联邦学习配置
# ============================================
federated:
  aggregation: fedavg
  rounds: 100
  local_epochs: 3
  num_clients: 3

# ============================================
# 训练配置
# ============================================
training:
  # 优化器配置
  optimizer: sgd
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0005
  
  # 数据加载配置
  batch_size: 128
  eval_batch_size: 512
  num_workers: 4
  prefetch_factor: 2
  
  # 计算配置
  use_amp: true
  num_threads: null
  compute_auc: true
  
  # 学习率调度器
  scheduler: cosine
  milestones: [60, 120, 160]  # multistep 用（当前使用 cosine）
  gamma: 0.2                  # multistep 用
  
  # 数据增强
  use_augmentation: true
  
  # 收敛检测
  converge_threshold: 0.0001
```

---

## 🔑 关键配置总结

### 必须完全一致的配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **数据集** | CIFAR-10 | 32x32 RGB 图像，10 类 |
| **数据分布** | IID | 随机均匀分配 |
| **模型** | ResNet18 | 标准 ResNet18 结构 |
| **优化器** | SGD | **必须用 SGD，不能用 Adam** |
| **学习率** | 0.1 | SGD 的推荐值 |
| **Momentum** | 0.9 | 标准值 |
| **Weight Decay** | 0.0005 | L2 正则化 |
| **调度器** | CosineAnnealingLR | T_max = 300 |
| **数据增强** | RandomCrop + RandomFlip | **必须启用** |
| **联邦轮数** | 100 | 总训练轮数 |
| **本地轮数** | 3 | 每轮客户端训练 3 个 epoch |
| **客户端数** | 3 | 参与训练的客户端数量 |
| **Batch Size** | 128 | 训练批次大小 |
| **随机种子** | 42 | 保证可复现性 |

### 性能优化配置（可选）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **混合精度** | true | GPU 模式下启用，加速训练 |
| **num_workers** | 4 | GPU 模式下使用多进程加载数据 |
| **eval_batch_size** | 512 | 评估时使用更大的批次加速 |

---

## 📈 预期效果

使用上述配置，预期能达到：

- **准确率**: ~95.3% (Round 83-85)
- **AUC**: ~0.997
- **损失**: ~0.165-0.167

### 训练曲线特征

- **Round 1-10**: 快速上升期（准确率 40% → 80%）
- **Round 10-50**: 稳定提升期（准确率 80% → 92%）
- **Round 50-100**: 精细优化期（准确率 92% → 95%+）

---

## ⚠️ 常见问题

### 1. 为什么必须用 SGD？
- Adam 在联邦学习中由于状态管理问题，效果较差
- SGD + Momentum 0.9 是联邦学习的标准配置

### 2. 为什么学习率是 0.1？
- 这是 SGD 的推荐初始学习率
- 如果使用 Adam，必须改为 0.001

### 3. 数据增强为什么重要？
- 数据增强可以显著提升模型泛化能力
- 缺失数据增强会导致准确率下降 10-15%

### 4. 学习率调度器的作用？
- CosineAnnealingLR 帮助模型在训练后期精细优化
- 固定学习率会导致后期收敛不佳

### 5. 如何调整客户端数量？
- 修改 `federated.num_clients`
- 需要重新生成数据（使用对应的客户端数量）
- 更多客户端通常需要更多轮数才能收敛

---

## 🚀 快速开始

### 1. 数据准备

```bash
# 生成 CIFAR-10 数据（IID 分布，3 个客户端）
python src/data_process/generate_cifar10_data.py
```

### 2. 训练

```bash
# 使用配置文件运行联邦学习
python src/main.py
```

### 3. 验证

检查日志文件，确认：
- ✅ 数据增强已启用
- ✅ 优化器: SGD
- ✅ 学习率调度: CosineAnnealingLR
- ✅ 准确率逐步提升

---

## 📚 参考

- **FedAvg 论文**: Communication-Efficient Learning of Deep Networks from Decentralized Data
- **ResNet 论文**: Deep Residual Learning for Image Recognition
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html

---

**最后更新**: 2025-12-03  
**当前效果**: 准确率 95.3%, AUC 0.997

