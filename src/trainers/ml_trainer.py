import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import torch
from types import SimpleNamespace
from src.utils.logging_config import get_logger
from .base_trainer import BaseTrainer
from tqdm import tqdm
import numpy as np
from sklearn.utils import parallel_backend
from sklearn.base import clone

logger = get_logger()

class MLTrainer(BaseTrainer):
    """传统机器学习模型的训练器"""
    
    def train(self, train_data: SimpleNamespace, test_data: SimpleNamespace) -> None:
        """训练传统机器学习模型（精简版）
        Args:
            train_data: SimpleNamespace, 包含x和y的训练数据
            test_data: SimpleNamespace, 包含x和y的测试数据
        """
        x_train = torch.tensor(train_data.x, dtype=torch.float32)
        y_train = torch.tensor(train_data.y, dtype=torch.long)
        if x_train.ndim > 2:
            x_train = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train, y_train)
        loss, acc, auc = self.evaluate(test_data)
        logger.info(f"最终准确率: {format(int(acc * 1000) / 1000, '.3f')}")
        logger.info(f"最终AUC: {format(int(auc * 1000) / 1000, '.3f') if auc is not None else '计算失败'}")


    def evaluate(self, test_data: SimpleNamespace):
        """评估ML模型，输出损失、准确率、AUC"""
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
        x_test = torch.tensor(test_data.x, dtype=torch.float32)
        y_test = test_data.y
        if x_test.ndim > 2:
            x_test = x_test.reshape(x_test.shape[0], -1)
        with torch.no_grad():
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