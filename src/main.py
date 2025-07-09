# ========== 导入区 ==========
import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from types import SimpleNamespace
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils.logging_config import get_logger
from src.models.models import FedAvgCNN, MLP
from src.models.ml_models import create_ml_model
from src.trainers import CentralizedTrainer, FederatedTrainer, MLTrainer

logger = get_logger(create_file=True)


# ========== 配置与工具函数区 ==========
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model(config):
    """根据配置创建模型
    
    Args:
        config: dict, 配置字典
        
    Returns:
        model: torch.nn.Module, 创建的模型
    """
    model_type = config['model']['type']
    
    if model_type == 'CNN':
        CNN_cfg = config['model'].get('CNN', {})
        return FedAvgCNN(**CNN_cfg)
    elif model_type == 'MLP':
        MLP_cfg = config['model'].get('MLP', {})
        return MLP(**MLP_cfg)
    elif model_type in ['KNN', 'RF', 'SVC', 'LR']:
        # 获取对应模型的参数
        model_params = config['model']['ml'][model_type]
        return create_ml_model(model_type, **model_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def get_trainer(model, config):
    """根据模型类型和配置获取合适的训练器
    
    Args:
        model: torch.nn.Module, 模型实例
        config: dict, 配置字典
        
    Returns:
        trainer: BaseTrainer, 训练器实例
    """
    model_type = config['model']['type']
    
    if model_type in ['CNN', 'MLP']:
        if config['mode'] == 'Federated':
            return FederatedTrainer(model=model, config=config)
        else:
            return CentralizedTrainer(model=model, config=config)
    elif model_type in [ 'KNN', 'RF', 'SVC', 'LR']:
        return MLTrainer(model=model, config=config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# ========== 数据加载区 ==========
def load_complete_data():
    """加载完整数据集，返回(train_data, test_data)"""
    config = load_config(os.path.join(PROJECT_ROOT, 'src', 'default.yaml'))
    data_dir = os.path.join(PROJECT_ROOT, config['data']['complete'])
    train_npz = np.load(os.path.join(data_dir, 'mnist_train.npz'))
    test_npz = np.load(os.path.join(data_dir, 'mnist_test.npz'))
    train_data = SimpleNamespace(x=train_npz['X_train'], y=train_npz['y_train'])
    test_data = SimpleNamespace(x=test_npz['X_test'], y=test_npz['y_test'])
    return train_data, test_data

def load_federated_data():
    """加载联邦学习各客户端数据，返回([client1_data, ...], [client1_test, ...])"""
    config = load_config(os.path.join(PROJECT_ROOT, 'src', 'default.yaml'))
    client_dirs = [os.path.join(PROJECT_ROOT, d) for d in config['data']['clients']]
    clients_data = []
    clients_test_data = []
    for client_dir in client_dirs:
        train_npz = np.load(os.path.join(client_dir, 'mnist_train.npz'))
        test_npz = np.load(os.path.join(client_dir, 'mnist_test.npz'))
        client_data = SimpleNamespace(x=train_npz['X_train'], y=train_npz['y_train'])
        client_test = SimpleNamespace(x=test_npz['X_test'], y=test_npz['y_test'])
        clients_data.append(client_data)
        clients_test_data.append(client_test)
    return clients_data, clients_test_data

def run_training(model, train_data, test_data, config, mode):
    """统一训练入口，自动选择Trainer并输出日志"""
    trainer = get_trainer(model, config)
    logger.info(f"开始训练 {config['model']['type']} 模型")
    trainer.train(train_data, test_data)
    logger.info(f"{mode}训练完成")

# ========== 主流程入口 ==========
def main():
    config = load_config(os.path.join(PROJECT_ROOT, 'src', 'default.yaml'))
    mode = config.get('mode', 'Centralized')
    logger.info(f"实验模式: {mode}")
    set_seed(config.get('seed', 42))
    
    model = create_model(config)
    logger.info(f"模型初始化完成: {config['model']['type']}")
    

    if mode == 'Centralized':
        train_data, test_data = load_complete_data()
        logger.info("加载完整数据集")
        start_time = time.time()  # 计时开始
        run_training(model, train_data, test_data, config, mode)
    elif mode == 'Federated':
        if config['model']['type'] in [ 'KNN', 'RF', 'SVC', 'LR']:
            logger.error("传统机器学习模型暂不支持联邦学习模式")
            return
        clients_data, clients_test_data = load_federated_data()
        logger.info("加载联邦学习客户端数据")
        start_time = time.time()  # 计时开始
        run_training(model, clients_data, clients_test_data, config, mode)
    else:
        logger.error(f"未知模式: {mode}")
        return

    end_time = time.time()  # 计时结束
    logger.info(f"训练总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()

