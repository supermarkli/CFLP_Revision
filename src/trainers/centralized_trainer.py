import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .base_trainer import BaseTrainer
from types import SimpleNamespace
from src.utils.logging_config import get_logger

logger = get_logger()

class CentralizedTrainer(BaseTrainer):
    def train(self, train_data: SimpleNamespace, test_data: SimpleNamespace) -> None:
        """实现集中式训练流程"""
        x_train = torch.tensor(train_data.x, dtype=torch.float32)
        y_train = torch.tensor(train_data.y, dtype=torch.long)
        if x_train.ndim == 3:
            x_train = x_train.unsqueeze(1)
            
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()
        epochs = self.config['training']['epochs']
        threshold = self.config['training'].get('converge_threshold', 0.001)
        
        self.model.train()
        acc_history = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
                
            avg_loss = total_loss / len(train_loader.dataset)

            # 评估准确率、AUC、损失
            loss, acc, auc = self.evaluate(test_data)
            logger.info(f"[Centralized][Epoch {epoch+1}/{epochs}] 准确率: {format(int(acc * 1000) / 1000, '.3f')}  AUC: {format(int(auc * 1000) / 1000, '.3f') if auc is not None else '计算失败'} 损失: {format(int(loss * 1000) / 1000, '.3f')}")
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
                
        # 恢复最佳模型
        self.save_best_model() 
        logger.info(f"最终准确率: {format(int(self.best_acc * 1000) / 1000, '.3f')}")
        logger.info(f"最终AUC: {format(int(self.best_auc * 1000) / 1000, '.3f') if self.best_auc is not None else '计算失败'}")

    def evaluate(self, test_data: SimpleNamespace) -> tuple:
        """统一的模型评估方法，返回损失、准确率、AUC"""
        self.model.eval()
        x_test = torch.tensor(test_data.x, dtype=torch.float32)
        y_test = torch.tensor(test_data.y, dtype=torch.long)
        if x_test.ndim == 3:
            x_test = x_test.unsqueeze(1)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.config['training'].get('eval_batch_size', 128),
            shuffle=False
        )
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        criterion = torch.nn.CrossEntropyLoss()
        import torch.nn.functional as F
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.detach().cpu())
            all_labels.append(batch_y.detach().cpu())
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
        avg_loss = total_loss / total
        acc = correct / total
        import numpy as np
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
            try:
                logger.info(f'y_true unique: {np.unique(y_true)}')
            except Exception:
                pass
            try:
                logger.info(f'all_probs 长度: {len(all_probs)}')
            except Exception:
                pass
            auc = None
        return avg_loss, acc, auc 