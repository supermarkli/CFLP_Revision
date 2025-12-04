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
from src.models.models import FedAvgCNN, MLP, ResNet18
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

def get_dataset_info(dataset_name):
    """根据数据集名称返回维度信息
    
    Args:
        dataset_name: str, 数据集名称 (mnist, cifar10, etc.)
        
    Returns:
        dict: 包含 in_channels, img_size, num_classes, cnn_dim, mlp_input_dim
    """
    dataset_configs = {
        'mnist': {
            'in_channels': 1,
            'img_size': 28,
            'num_classes': 10,
        },
        'cifar10': {
            'in_channels': 3,
            'img_size': 32,
            'num_classes': 10,
        },
        'cifar100': {
            'in_channels': 3,
            'img_size': 32,
            'num_classes': 100,
        },
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"未知数据集: {dataset_name}，支持的数据集: {list(dataset_configs.keys())}")
    
    info = dataset_configs[dataset_name]
    
    # 计算 CNN (FedAvgCNN) 的 dim: 经过两个 5x5 conv + 2x2 maxpool
    # conv1: (img_size - 5 + 1) / 2 = (img_size - 4) / 2
    # conv2: ((img_size - 4) / 2 - 5 + 1) / 2 = ((img_size - 4) / 2 - 4) / 2
    after_conv1 = (info['img_size'] - 4) // 2
    after_conv2 = (after_conv1 - 4) // 2
    info['cnn_dim'] = 64 * after_conv2 * after_conv2
    
    # 计算 MLP 的 input_dim
    info['mlp_input_dim'] = info['in_channels'] * info['img_size'] * info['img_size']
    
    return info

def create_model(config):
    """根据配置创建模型，自动推断数据集相关参数
    
    Args:
        config: dict, 配置字典
        
    Returns:
        model: torch.nn.Module, 创建的模型
    """
    model_type = config['model']['type']
    dataset_name = config['data'].get('dataset', 'mnist')
    dataset_info = get_dataset_info(dataset_name)
    
    if model_type == 'CNN':
        # 使用数据集信息作为默认值，配置文件可以覆盖
        default_cfg = {
            'in_features': dataset_info['in_channels'],
            'num_classes': dataset_info['num_classes'],
            'dim': dataset_info['cnn_dim'],
        }
        user_cfg = config['model'].get('CNN', {})
        # 只有当用户没有在配置中指定时才使用默认值
        final_cfg = {**default_cfg, **{k: v for k, v in user_cfg.items() if k not in ['in_features', 'dim', 'num_classes'] or v != default_cfg.get(k)}}
        # 如果用户配置中显式指定了值，使用数据集推断的值覆盖
        final_cfg['in_features'] = dataset_info['in_channels']
        final_cfg['dim'] = dataset_info['cnn_dim']
        final_cfg['num_classes'] = dataset_info['num_classes']
        return FedAvgCNN(**final_cfg)
    elif model_type == 'ResNet18':
        default_cfg = {
            'in_features': dataset_info['in_channels'],
            'num_classes': dataset_info['num_classes'],
        }
        final_cfg = {**default_cfg}
        return ResNet18(**final_cfg)
    elif model_type == 'MLP':
        user_cfg = config['model'].get('MLP', {})
        default_cfg = {
            'input_dim': dataset_info['mlp_input_dim'],
            'num_classes': dataset_info['num_classes'],
            'hidden_dims': user_cfg.get('hidden_dims', [128, 64]),
            'activation': user_cfg.get('activation', 'relu'),
        }
        return MLP(**default_cfg)
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
    
    if model_type in ['CNN', 'MLP', 'ResNet18']:
        if config['mode'] == 'Federated':
            return FederatedTrainer(model=model, config=config)
        else:
            return CentralizedTrainer(model=model, config=config)
    elif model_type in [ 'KNN', 'RF', 'SVC', 'LR']:
        return MLTrainer(model=model, config=config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# ========== 数据加载区 ==========
def get_dataset_name(config):
    """获取数据集名称"""
    return config['data'].get('dataset', 'mnist')

def load_complete_data():
    """加载完整数据集，返回(train_data, test_data)"""
    logger.info("加载完整数据集")
    config = load_config(os.path.join(PROJECT_ROOT, 'src', 'default.yaml'))
    data_dir = os.path.join(PROJECT_ROOT, config['data']['complete'])
    dataset = get_dataset_name(config)
    
    train_npz = np.load(os.path.join(data_dir, f'{dataset}_train.npz'))
    test_npz = np.load(os.path.join(data_dir, f'{dataset}_test.npz'))
    train_data = SimpleNamespace(x=train_npz['X_train'], y=train_npz['y_train'])
    test_data = SimpleNamespace(x=test_npz['X_test'], y=test_npz['y_test'])
    logger.info(f"数据集: {dataset}, 训练集形状: {train_data.x.shape}, 测试集形状: {test_data.x.shape}")
    return train_data, test_data

def load_federated_data():
    """
    根据配置加载联邦学习各客户端数据（IID或Non-IID）。
    
    返回:
        clients_data: 各客户端的训练数据列表
        global_test_data: 全局测试集（用于评估全局模型）
    """
    config = load_config(os.path.join(PROJECT_ROOT, 'src', 'default.yaml'))
    
    # 获取数据集名称
    dataset = get_dataset_name(config)
    
    # 根据配置确定数据分布类型
    dist_type = config['data'].get('federated_dist', 'iid')
    logger.info(f"为联邦学习加载数据，数据集: {dataset}, 分布类型: {dist_type}")

    if dist_type == 'iid':
        train_file = f'{dataset}_train.npz'
    elif dist_type == 'noniid_label_skew':
        train_file = f'{dataset}_train_noniid_label_skew.npz'
    elif dist_type == 'noniid_quantity_skew':
        train_file = f'{dataset}_train_noniid_quantity_skew.npz'
    elif dist_type == 'noniid_dirichlet':
        train_file = f'{dataset}_train_noniid_dirichlet.npz'
    else:
        logger.error(f"不支持的数据分布类型: {dist_type}")
        raise ValueError(f"不支持的数据分布类型: {dist_type}")

    # 加载各客户端的训练数据
    client_dirs = [os.path.join(PROJECT_ROOT, d) for d in config['data']['clients']]
    clients_data = []
    for client_dir in client_dirs:
        train_path = os.path.join(client_dir, train_file)

        if not os.path.exists(train_path):
            logger.error(f"数据文件不存在: {train_path}")
            logger.error(f"请先运行 'python src/data_process/generate_{dataset}_data.py' 生成所需的数据文件。")
            sys.exit(1)

        train_npz = np.load(train_path)
        client_data = SimpleNamespace(x=train_npz['X_train'], y=train_npz['y_train'])
        clients_data.append(client_data)
        logger.info(f"客户端 {len(clients_data)} 训练数据加载完成，样本数: {len(client_data.x)}")
    
    # 加载全局测试集（官方测试集）
    global_test_path = os.path.join(PROJECT_ROOT, config['data']['complete'], f'{dataset}_test.npz')
    if not os.path.exists(global_test_path):
        logger.error(f"全局测试集不存在: {global_test_path}")
        logger.error(f"请先运行 'python src/data_process/generate_{dataset}_data.py' 生成所需的数据文件。")
        sys.exit(1)
    
    test_npz = np.load(global_test_path)
    global_test_data = SimpleNamespace(x=test_npz['X_test'], y=test_npz['y_test'])
    logger.info(f"全局测试集加载完成，样本数: {len(global_test_data.x)}")
    
    return clients_data, global_test_data

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
    
    # 设备信息日志
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🚀 使用 GPU 训练: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = torch.device('cpu')
        logger.info(f"⚠️ 未检测到 GPU，使用 CPU 训练")
    
    model = create_model(config)
    logger.info(f"模型初始化完成: {config['model']['type']}")
    

    if mode == 'Centralized':
        train_data, test_data = load_complete_data()
        start_time = time.time()  # 计时开始
        run_training(model, train_data, test_data, config, mode)
    elif mode == 'Federated':
        if config['model']['type'] in ['KNN', 'RF', 'SVC', 'LR']:
            logger.error("传统机器学习模型暂不支持联邦学习模式")
            return
        clients_data, global_test_data = load_federated_data()
        start_time = time.time()  # 计时开始
        run_training(model, clients_data, global_test_data, config, mode)
    else:
        logger.error(f"未知模式: {mode}")
        return

    end_time = time.time()  # 计时结束
    logger.info(f"训练总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()

