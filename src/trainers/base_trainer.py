from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Tuple, Optional

class BaseTrainer(ABC):
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.best_state_dict = None
        self.best_acc = 0
        
    @abstractmethod
    def train(self, train_data: SimpleNamespace, test_data: SimpleNamespace) -> None:
        """训练模型的抽象方法"""
        pass
        
    @abstractmethod
    def evaluate(self, test_data: SimpleNamespace):
        """模型评估的抽象方法，需子类实现"""
        pass
        
    def save_best_model(self) -> None:
        """保存最佳模型状态"""
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            
    def _check_convergence(self, acc_history: list, threshold: float) -> bool:
        """检查模型是否收敛"""
        if len(acc_history) >= 4:
            recent_accs = acc_history[-4:]
            return max(recent_accs) - min(recent_accs) < threshold
        return False 