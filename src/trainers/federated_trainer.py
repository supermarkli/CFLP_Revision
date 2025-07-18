import os
import sys
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import torch
from typing import List, Tuple
from types import SimpleNamespace
from .base_trainer import BaseTrainer
from src.utils.logging_config import get_logger
from src.clients.client import Client
from src.servers.server import Server

logger = get_logger()

class FederatedTrainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, config: dict):
        super().__init__(model, config)
        self.clients = []
        self.server = None
        
    def setup(self, clients_data: List[SimpleNamespace]):
        """初始化客户端和服务器"""
        self.clients = [
            Client(client_id=i, data=data, model=self.model, config=self.config)
            for i, data in enumerate(clients_data)
        ]
        self.server = Server(clients=self.clients, model=self.model, config=self.config)
        
    def train(self, train_data: List[SimpleNamespace], test_data: List[SimpleNamespace]) -> None:
        """实现联邦学习训练流程"""
        if not self.clients or not self.server:
            self.setup(train_data)
            
        rounds = self.config['federated']['rounds']
        local_epochs = self.config['federated']['local_epochs']
        threshold = self.config.get('converge_threshold', 0.001)
        
        acc_history = []
        auc_history = []
        loss_history = []
        
        for r in range(rounds):
            
            # 分发全局模型
            self.server.distribute()
            
            # 收集客户端训练结果
            client_params_list, client_accs, client_aucs, client_losses = \
                self.server.collect_and_evaluate(test_data, local_epochs)
            
            # 聚合模型
            self.server.aggregate(client_params_list)
            
            # 安全地计算平均指标
            # 过滤掉None值，避免计算错误
            valid_aucs = [auc for auc in client_aucs if auc is not None]
            
            avg_acc = sum(client_accs) / len(client_accs) if client_accs else 0
            avg_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else None
            avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0
            
            acc_history.append(avg_acc)
            if avg_auc is not None:
                auc_history.append(avg_auc)
            loss_history.append(avg_loss)
            
            log_message = (
                f"[Federated][Round {r+1}/{rounds}] "
                f"客户端平均准确率: {format(int(avg_acc * 1000) / 1000, '.3f')}, "
                f"平均AUC: {format(int(avg_auc * 1000) / 1000, '.3f') if avg_auc is not None and not math.isnan(avg_auc) else 'pass'} "
                f"平均损失: {format(int(avg_loss * 1000) / 1000, '.3f')}"
            )
            logger.info(log_message)
            
            # 保存最佳模型
            if avg_acc > self.best_acc:
                self.best_acc = avg_acc
                self.best_auc = avg_auc
                self.best_loss = avg_loss
                self.best_state_dict = self.model.state_dict().copy()
            
            # 收敛检测
            if self._check_convergence(acc_history, threshold):
                logger.info(
                    f"[Federated] 收敛检测：最近准确率波动未超过阈值({threshold})，"
                    "提前停止训练"
                )
                break
                
            
        # 恢复最佳模型
        self.save_best_model()
        logger.info(f"最终准确率: {format(int(self.best_acc * 1000) / 1000, '.3f')}")
        logger.info(f"最终AUC: {format(int(self.best_auc * 1000) / 1000, '.3f') if self.best_auc is not None and not math.isnan(self.best_auc) else 'pass'}")

    def evaluate(self, test_data):
        """联邦Trainer不直接评估全局模型，仅为抽象方法占位"""
        pass 