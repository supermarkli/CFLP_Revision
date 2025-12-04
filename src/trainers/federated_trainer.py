import os
import sys
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
from typing import List
from types import SimpleNamespace
from .base_trainer import BaseTrainer
from src.utils.logging_config import get_logger
from src.clients.client import Client
from src.servers.server import Server

logger = get_logger()

def safe_format(value, default='N/A'):
    """安全格式化数值，处理 NaN 和 None"""
    if value is None:
        return default
    try:
        if math.isnan(value):
            return default
        return format(int(value * 1000) / 1000, '.3f')
    except (ValueError, TypeError):
        return default

def format_bytes(num_bytes):
    """格式化字节数为人类可读的格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

class FederatedTrainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, config: dict):
        super().__init__(model, config)
        self.clients = []
        self.server = None
        # 实验指标
        self.convergence_round = 0  # 收敛轮数
        self.total_communication_volume = 0  # 总通信体积（字节）
        self.final_acc_std = None  # 最后阶段准确率标准差
        self.local_global_gap = None  # Local-Global Gap
        self.best_auc = None
        self.best_loss = None
        
    def setup(self, clients_data: List[SimpleNamespace]):
        """初始化客户端和服务器"""
        self.clients = [
            Client(client_id=i, data=data, model=self.model, config=self.config)
            for i, data in enumerate(clients_data)
        ]
        self.server = Server(clients=self.clients, model=self.model, config=self.config)
        
    def train(self, train_data: List[SimpleNamespace], test_data: SimpleNamespace) -> None:
        """
        实现联邦学习训练流程
        
        Args:
            train_data: 各客户端的训练数据列表
            test_data: 全局测试集（用于评估聚合后的全局模型）
        """
        if not self.clients or not self.server:
            self.setup(train_data)
        
        # 获取配置：使用非常大的默认值表示"无限制"
        max_rounds = self.config['federated'].get('rounds', 10000)
        local_epochs = self.config['federated'].get('local_epochs', 1)
        threshold = self.config['training'].get('converge_threshold', 0.0001)
        
        # 获取每轮通信体积
        comm_per_round = self.server.get_communication_volume_per_round()
        logger.info(f"每轮通信体积: {format_bytes(comm_per_round)}")
        
        acc_history = []
        auc_history = []
        loss_history = []
        
        for r in range(max_rounds):
            
            # 分发全局模型
            self.server.distribute()
            
            # 各客户端本地训练
            client_params_list = self.server.collect_parameters_after_training(
                local_epochs, round_num=r+1
            )
            
            # 聚合模型
            self.server.aggregate(client_params_list)
            
            # 累计通信体积
            self.total_communication_volume += comm_per_round
            
            # 使用全局测试集评估聚合后的全局模型
            global_loss, global_acc, global_auc = self.server.evaluate_global_model(test_data)
            
            acc_history.append(global_acc)
            if global_auc is not None:
                auc_history.append(global_auc)
            loss_history.append(global_loss)
            
            log_message = (
                f"[Federated][Round {r+1}] "
                f"全局模型准确率: {safe_format(global_acc)}, "
                f"全局模型AUC: {safe_format(global_auc, 'N/A')}, "
                f"全局模型损失: {safe_format(global_loss)}"
            )
            logger.info(log_message)
            
            # 保存最佳模型
            if global_acc > self.best_acc:
                self.best_acc = global_acc
                self.best_auc = global_auc
                self.best_loss = global_loss
                self.best_state_dict = self.model.state_dict().copy()
            
            # 收敛检测
            if self._check_convergence(acc_history, threshold):
                self.convergence_round = r + 1
                logger.info(
                    f"[Federated] 收敛检测：在第 {self.convergence_round} 轮收敛，"
                    f"最近准确率波动未超过阈值({threshold})"
                )
                break
        else:
            # 如果达到最大轮数仍未收敛
            self.convergence_round = max_rounds
            logger.info(f"[Federated] 达到最大轮数 {max_rounds}，训练结束")
        
        # ========== 计算最终指标 ==========
        
        # 1. 计算最后10轮（或所有轮次如果不足10轮）准确率的标准差
        final_rounds = min(10, len(acc_history))
        if final_rounds > 0:
            recent_accs = acc_history[-final_rounds:]
            self.final_acc_std = np.std(recent_accs)
            logger.info(f"最后 {final_rounds} 轮准确率标准差: {safe_format(self.final_acc_std)}")
        
        # 2. 计算 Local-Global Gap
        # 首先恢复到最佳全局模型
        self.save_best_model()
        
        # 评估各客户端本地模型在全局测试集上的表现
        local_accs, avg_local_acc = self.server.evaluate_local_models_on_global_test(test_data)
        self.local_global_gap = self.best_acc - avg_local_acc
        
        logger.info(f"全局模型准确率: {safe_format(self.best_acc)}")
        logger.info(f"本地模型平均准确率: {safe_format(avg_local_acc)}")
        logger.info(f"Local-Global Gap: {safe_format(self.local_global_gap)}")
        
        # 3. 输出通信相关指标
        logger.info(f"收敛轮数: {self.convergence_round}")
        logger.info(f"总通信体积: {format_bytes(self.total_communication_volume)}")
        
        # 输出最终结果
        logger.info(f"最终准确率: {safe_format(self.best_acc)}")
        logger.info(f"最终AUC: {safe_format(self.best_auc, 'N/A')}")

    def evaluate(self, test_data):
        """联邦Trainer不直接评估全局模型，仅为抽象方法占位"""
        pass
    
    def get_experiment_metrics(self):
        """
        获取实验指标，用于日志解析和结果汇总。
        
        Returns:
            dict: 包含所有实验指标的字典
        """
        return {
            'convergence_round': self.convergence_round,
            'total_communication_volume': self.total_communication_volume,
            'final_acc_std': self.final_acc_std,
            'local_global_gap': self.local_global_gap,
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_loss': self.best_loss,
        }
