import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import copy
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.logging_config import get_logger

logger = get_logger()

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
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 混合精度训练设置
        self.use_amp = config['training'].get('use_amp', True) and self.use_cuda
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # 数据准备（使用 from_numpy 避免复制）
        x_np = self.data.x.astype(np.float32) if self.data.x.dtype != np.float32 else self.data.x
        y_np = self.data.y.astype(np.int64) if self.data.y.dtype != np.int64 else self.data.y
        x_train = torch.from_numpy(x_np)
        y_train = torch.from_numpy(y_np)
        if x_train.ndim == 3:
            x_train = x_train.unsqueeze(1)
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

    def local_train(self):
        """本地训练一轮"""
        self.model.train()
        epochs = self.config['federated']['local_epochs']
        for epoch in range(epochs):
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
                y_proba = torch.cat(all_probs).numpy()
                n_classes = self.config['model'][self.config['model']['type']]['num_classes']
                
                # 检查测试集中的类别是否足够计算AUC
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', labels=np.arange(n_classes))
            except Exception:
                pass

        return avg_loss, acc, auc
