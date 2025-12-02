import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from .base_trainer import BaseTrainer
from types import SimpleNamespace
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

class CentralizedTrainer(BaseTrainer):
    def train(self, train_data: SimpleNamespace, test_data: SimpleNamespace) -> None:
        """实现集中式训练流程"""
        # 检测是否使用 CUDA
        use_cuda = self.device.type == 'cuda'
        
        # CPU 优化：线程数设置
        if not use_cuda:
            num_threads = self.config['training'].get('num_threads', None)
            if num_threads is not None:
                # 仅当用户显式指定时才设置，否则使用 PyTorch 默认值
                torch.set_num_threads(num_threads)
                logger.info(f"CPU 训练模式：使用 {num_threads} 个计算线程")
            else:
                logger.info(f"CPU 训练模式：使用 PyTorch 默认线程数 ({torch.get_num_threads()})")
        
        # 启用 cuDNN benchmark 优化（仅 CUDA 且固定输入尺寸时有效）
        if use_cuda:
            torch.backends.cudnn.benchmark = True
        
        # 使用 from_numpy 避免数据复制（比 torch.tensor 更高效）
        x_np = train_data.x.astype(np.float32) if train_data.x.dtype != np.float32 else train_data.x
        y_np = train_data.y.astype(np.int64) if train_data.y.dtype != np.int64 else train_data.y
        x_train = torch.from_numpy(x_np)
        y_train = torch.from_numpy(y_np)
        if x_train.ndim == 3:
            x_train = x_train.unsqueeze(1)
        
        # 数据增强（对 CIFAR-10 等图像数据集很重要）
        use_augmentation = self.config['training'].get('use_augmentation', False)
        if use_augmentation and x_train.shape[2] == 32:  # CIFAR-10/100
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
            train_dataset = AugmentedDataset(x_train, y_train, transform=train_transform)
            logger.info("数据增强已启用: RandomCrop(32, padding=4) + RandomHorizontalFlip")
        else:
            train_dataset = TensorDataset(x_train, y_train)
            
        # 优化: num_workers 仅在 GPU 模式下有效
        num_workers = self.config['training'].get('num_workers', 4) if use_cuda else 0
        prefetch_factor = self.config['training'].get('prefetch_factor', 2) if num_workers > 0 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor
        )
        
        # 优化器选择
        lr = self.config['training']['learning_rate']
        optimizer_type = self.config['training'].get('optimizer', 'adam').lower()
        weight_decay = self.config['training'].get('weight_decay', 0)
        
        if optimizer_type == 'sgd':
            momentum = self.config['training'].get('momentum', 0.9)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            logger.info(f"优化器: SGD (lr={lr}, momentum={momentum}, weight_decay={weight_decay})")
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"优化器: Adam (lr={lr}, weight_decay={weight_decay})")
        
        criterion = torch.nn.CrossEntropyLoss()
        epochs = self.config['training']['epochs']
        threshold = self.config['training'].get('converge_threshold', 0.001)
        eval_interval = self.config['training'].get('eval_interval', 1)
        
        # 学习率调度器
        scheduler_type = self.config['training'].get('scheduler', None)
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
            logger.info(f"学习率调度: CosineAnnealingLR (T_max={epochs})")
        elif scheduler_type == 'multistep':
            milestones = self.config['training'].get('milestones', [60, 120, 160])
            gamma = self.config['training'].get('gamma', 0.2)
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            logger.info(f"学习率调度: MultiStepLR (milestones={milestones}, gamma={gamma})")
        
        # 混合精度训练 (仅 CUDA 时启用)
        use_amp = self.config['training'].get('use_amp', True) and use_cuda
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        self.model.train()
        acc_history = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                # non_blocking 仅在使用 pin_memory 时有效
                batch_x = batch_x.to(self.device, non_blocking=use_cuda)
                batch_y = batch_y.to(self.device, non_blocking=use_cuda)
                optimizer.zero_grad(set_to_none=True)  # 比 zero_grad() 更快
                
                # 混合精度前向传播 (use_amp=False 时不做任何转换)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                
                # 混合精度反向传播 (scaler.enabled=False 时直接透传)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * batch_x.size(0)
                
            avg_loss = total_loss / len(train_loader.dataset)
            
            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()

            # 按间隔评估（减少评估开销）
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                loss, acc, auc = self.evaluate(test_data)
                lr_info = f" LR: {current_lr:.6f}" if scheduler else ""
                logger.info(f"[Centralized][Epoch {epoch+1}/{epochs}] 准确率: {format(int(acc * 1000) / 1000, '.3f')}  AUC: {format(int(auc * 1000) / 1000, '.3f') if auc is not None else '计算失败'} 损失: {format(int(loss * 1000) / 1000, '.3f')}{lr_info}")
                acc_history.append(acc)
                
                # 保存最佳模型
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_auc = auc
                    self.best_state_dict = self.model.state_dict().copy()
                
                # 收敛检测
                if self._check_convergence(acc_history, threshold):
                    logger.info(f"[Centralized] 收敛检测：最近准确率波动未超过阈值({threshold})，提前停止训练。")
                    break
            
            self.model.train()  # 确保评估后恢复训练模式
                
        # 恢复最佳模型
        self.save_best_model() 
        logger.info(f"最终准确率: {format(int(self.best_acc * 1000) / 1000, '.3f')}")
        logger.info(f"最终AUC: {format(int(self.best_auc * 1000) / 1000, '.3f') if self.best_auc is not None else '计算失败'}")

    def evaluate(self, test_data: SimpleNamespace) -> tuple:
        """统一的模型评估方法，返回损失、准确率、AUC"""
        self.model.eval()
        use_cuda = self.device.type == 'cuda'
        
        # 使用 from_numpy 避免数据复制
        x_np = test_data.x.astype(np.float32) if test_data.x.dtype != np.float32 else test_data.x
        y_np = test_data.y.astype(np.int64) if test_data.y.dtype != np.int64 else test_data.y
        x_test = torch.from_numpy(x_np)
        y_test = torch.from_numpy(y_np)
        if x_test.ndim == 3:
            x_test = x_test.unsqueeze(1)
        test_dataset = TensorDataset(x_test, y_test)
        
        # 评估时：CPU 模式下 num_workers=0 更快（数据已在内存）
        num_workers = self.config['training'].get('num_workers', 4) if use_cuda else 0
        prefetch_factor = self.config['training'].get('prefetch_factor', 2) if num_workers > 0 else None
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['training'].get('eval_batch_size', 256),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            prefetch_factor=prefetch_factor
        )
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        criterion = torch.nn.CrossEntropyLoss()
        
        # 评估时使用 inference_mode（比 no_grad 更快）和混合精度 (仅 CUDA 时启用 AMP)
        use_amp = self.config['training'].get('use_amp', True) and use_cuda
        # 可选：减少 AUC 计算频率以加速（仅在需要时计算）
        compute_auc = self.config['training'].get('compute_auc', True)
        
        with torch.inference_mode():  # 比 no_grad 更快
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=use_cuda)
                batch_y = batch_y.to(self.device, non_blocking=use_cuda)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                # 只在需要 AUC 时计算概率
                if compute_auc:
                    probs = F.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu())
                    all_labels.append(batch_y.cpu())
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
        avg_loss = total_loss / total
        acc = correct / total
        
        # 只在需要时计算 AUC（AUC 计算较耗时）
        if compute_auc:
            from sklearn.metrics import roc_auc_score
            try:
                y_true = torch.cat(all_labels).numpy()
                y_proba = torch.cat(all_probs).numpy()
                n_classes = y_proba.shape[1]
                unique_classes = np.unique(y_true)
                if len(unique_classes) == n_classes:
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
                else:
                    logger.info(f"类别不全，无法计算AUC: 现有类别 {unique_classes}, 期望类别数 {n_classes}")
                    auc = None
            except Exception as e:
                logger.info(f'AUC计算异常: {e}')
                auc = None
        else:
            auc = None
        return avg_loss, acc, auc 