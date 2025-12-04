import os
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torchvision.transforms as transforms
import copy
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.logging_config import get_logger

logger = get_logger()

class AugmentedDataset(Dataset):
    """支持数据增强的数据集包装类"""
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class Client:
    def __init__(self, client_id, data, model, config):
        self.client_id = client_id
        self.data = data  # 本地数据
        self.model = copy.deepcopy(model)  # 模型副本
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = self.device.type == 'cuda'
        self.model = self.model.to(self.device)
        
        # CPU/GPU 优化设置
        if not self.use_cuda:
            num_threads = config['training'].get('num_threads', None)
            if num_threads is not None:
                torch.set_num_threads(num_threads)
        else:
            torch.backends.cudnn.benchmark = True
        
        # 根据配置选择优化器
        optimizer_type = config['training'].get('optimizer', 'adam').lower()
        lr = config['training']['learning_rate']
        
        if optimizer_type == 'sgd':
            momentum = config['training'].get('momentum', 0.9)
            weight_decay = config['training'].get('weight_decay', 0.0005)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
            logger.info(f"[Client {client_id}] 优化器: SGD (lr={lr}, momentum={momentum}, weight_decay={weight_decay})")
        else:  # 默认使用 Adam
            weight_decay = config['training'].get('weight_decay', 0.0)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            logger.info(f"[Client {client_id}] 优化器: Adam (lr={lr}, weight_decay={weight_decay})")
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 混合精度训练设置
        self.use_amp = config['training'].get('use_amp', True) and self.use_cuda
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # 学习率调度器初始化
        self.scheduler_type = config['training'].get('scheduler', None)
        self.local_epochs = config['federated'].get('local_epochs', 1)
        # 使用预估收敛轮数（而非最大轮数）用于调度器，避免 T_max 过大导致学习率下降太慢
        estimated_rounds = config['training'].get('estimated_rounds', 200)
        
        # 初始化调度器（基于预估总训练步数：estimated_rounds * local_epochs）
        if self.scheduler_type == 'cosine':
            T_max = estimated_rounds * self.local_epochs
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)
            logger.info(f"[Client {client_id}] 学习率调度: CosineAnnealingLR (T_max={T_max})")
        elif self.scheduler_type == 'multistep':
            milestones = config['training'].get('milestones', [60, 120, 160])
            gamma = config['training'].get('gamma', 0.2)
            # 对于 multistep，milestones 需要转换为基于总步数
            # 这里假设 milestones 是基于集中式训练的 epoch，需要按比例转换
            # 简化处理：直接使用（用户需要根据实际情况调整）
            self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
            logger.info(f"[Client {client_id}] 学习率调度: MultiStepLR (milestones={milestones}, gamma={gamma})")
        else:
            self.scheduler = None
        
        # 数据准备（使用 from_numpy 避免复制）
        x_np = self.data.x.astype(np.float32) if self.data.x.dtype != np.float32 else self.data.x
        y_np = self.data.y.astype(np.int64) if self.data.y.dtype != np.int64 else self.data.y
        x_train = torch.from_numpy(x_np)
        y_train = torch.from_numpy(y_np)
        if x_train.ndim == 3:
            x_train = x_train.unsqueeze(1)
        
        # 数据增强（对 CIFAR-10 等图像数据集很重要）
        use_augmentation = config['training'].get('use_augmentation', False)
        if use_augmentation and x_train.shape[2] == 32:  # CIFAR-10/100
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
            self.train_dataset = AugmentedDataset(x_train, y_train, transform=train_transform)
            logger.info(f"[Client {client_id}] 数据增强已启用")
        else:
            self.train_dataset = TensorDataset(x_train, y_train)
        
        # DataLoader 优化：CPU 模式下 num_workers=0 更快（TensorDataset 数据已在内存）
        num_workers = config['training'].get('num_workers', 4) if self.use_cuda else 0
        prefetch_factor = config['training'].get('prefetch_factor', 2) if num_workers > 0 else None
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor
        )

    def set_parameters(self, parameters):
        """设置本地模型参数"""
        self.model.load_state_dict(parameters)

    def get_parameters(self):
        """获取本地模型参数"""
        return self.model.state_dict()

    def local_train(self, round_num=None):
        """本地训练一个 epoch（外层由 server 控制训练轮数）
        
        Args:
            round_num: 当前联邦学习轮数（可选，用于日志记录）
        """
        self.model.train()
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device, non_blocking=self.use_cuda)
            batch_y = batch_y.to(self.device, non_blocking=self.use_cuda)
            self.optimizer.zero_grad(set_to_none=True)  # 更快的梯度清零
            
            # 混合精度前向传播
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
            
            # 混合精度反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # 每个 epoch 后更新学习率（如果使用调度器）
        if self.scheduler is not None:
            self.scheduler.step()

    def evaluate(self, test_data):
        """在本地或全局测试集上评估模型，返回loss, acc, auc"""
        self.model.eval()
        
        # 使用 from_numpy 避免数据复制
        x_np = test_data.x.astype(np.float32) if test_data.x.dtype != np.float32 else test_data.x
        y_np = test_data.y.astype(np.int64) if test_data.y.dtype != np.int64 else test_data.y
        x_test = torch.from_numpy(x_np)
        y_test = torch.from_numpy(y_np)
        if x_test.ndim == 3:
            x_test = x_test.unsqueeze(1)
        test_dataset = TensorDataset(x_test, y_test)
        
        # DataLoader 优化：CPU 模式下 num_workers=0 更快
        num_workers = self.config['training'].get('num_workers', 4) if self.use_cuda else 0
        prefetch_factor = self.config['training'].get('prefetch_factor', 2) if num_workers > 0 else None
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training'].get('eval_batch_size', 256),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.use_cuda,
            prefetch_factor=prefetch_factor
        )
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        compute_auc = self.config['training'].get('compute_auc', True)
        
        with torch.inference_mode():  # 比 no_grad 更快
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=self.use_cuda)
                batch_y = batch_y.to(self.device, non_blocking=self.use_cuda)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                
                if compute_auc:
                    probs = F.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu())
                    all_labels.append(batch_y.cpu())

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
        
        avg_loss = total_loss / total
        acc = correct / total
        
        # 安全地计算AUC
        auc = None
        if compute_auc:
            try:
                y_true = torch.cat(all_labels).numpy()
                y_proba = torch.cat(all_probs).numpy().astype(np.float64)  # 转为 float64 提高精度
                # 修复 AMP 导致的概率舍入误差：重新归一化确保每行和为 1
                y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
                
                # 从概率矩阵的列数推断类别数（更可靠）
                n_classes = y_proba.shape[1]
                
                # 检查测试集中的类别是否足够计算AUC（至少需要2个类别）
                unique_classes = np.unique(y_true)
                if len(unique_classes) > 1 and len(unique_classes) == n_classes:
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
                elif len(unique_classes) > 1:
                    # 如果类别数不匹配，使用实际存在的类别
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', labels=unique_classes)
            except Exception as e:
                # 静默失败，返回 None 而不是 NaN
                pass

        return avg_loss, acc, auc
