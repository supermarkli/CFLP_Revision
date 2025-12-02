import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import torch
from types import SimpleNamespace
from src.utils.logging_config import get_logger
from .base_trainer import BaseTrainer
import numpy as np

logger = get_logger()

class MLTrainer(BaseTrainer):
    """传统机器学习模型的训练器"""
    
    def train(self, train_data: SimpleNamespace, test_data: SimpleNamespace) -> None:
        """训练传统机器学习模型（精简版）
        Args:
            train_data: SimpleNamespace, 包含x和y的训练数据
            test_data: SimpleNamespace, 包含x和y的测试数据
        """
        # 使用 from_numpy 避免数据复制
        x_np = train_data.x.astype(np.float32) if train_data.x.dtype != np.float32 else train_data.x
        y_np = train_data.y.astype(np.int64) if train_data.y.dtype != np.int64 else train_data.y
        x_train = torch.from_numpy(x_np)
        y_train = torch.from_numpy(y_np)
        if x_train.ndim > 2:
            x_train = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train, y_train)
        loss, acc, auc = self.evaluate(test_data)
        logger.info(f"最终准确率: {format(int(acc * 1000) / 1000, '.3f')}")
        logger.info(f"最终AUC: {format(int(auc * 1000) / 1000, '.3f') if auc is not None else '计算失败'}")


    def evaluate(self, test_data: SimpleNamespace):
        """评估ML模型，输出损失、准确率、AUC"""
        from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
        
        # 使用 from_numpy 避免数据复制
        x_np = test_data.x.astype(np.float32) if test_data.x.dtype != np.float32 else test_data.x
        x_test = torch.from_numpy(x_np)
        y_test = test_data.y
        if x_test.ndim > 2:
            x_test = x_test.reshape(x_test.shape[0], -1)
        
        with torch.inference_mode():  # 比 no_grad 更快
            y_proba = self.model(x_test).cpu().numpy()
            y_pred = np.argmax(y_proba, axis=1)
        try:
            loss = log_loss(y_test, y_proba)
        except Exception:
            loss = 0.0
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except Exception:
            auc = None
        return loss, acc, auc 