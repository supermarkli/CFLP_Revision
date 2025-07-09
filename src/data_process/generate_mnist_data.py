import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
from torchvision import datasets, transforms
from src.utils.logging_config import get_logger
from sklearn.model_selection import train_test_split
import shutil

# ===== 全局参数 =====
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, 'data', 'mnist_raw')
OUTPUT_NPZ = os.path.join(PROJECT_ROOT, 'data', 'mnist_raw', 'mnist_data.npz')
NORMALIZE = True  
TEST_SIZE = 0.2

logger = get_logger()

def load_mnist_data(download_dir, normalize=True):
    """加载MNIST数据集并返回numpy数组"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
    ])
    train_dataset = datasets.MNIST(root=download_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=download_dir, train=False, download=True, transform=transform)

    def dataset_to_numpy(dataset):
        images = []
        labels = []
        for img, label in dataset:
            img_np = img.numpy()  # 保留原始 shape: [1, 28, 28]
            images.append(img_np)
            labels.append(label)
        return np.array(images), np.array(labels)

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    logger.info(f"MNIST加载完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
    return X_train, y_train, X_test, y_test

def save_npz(X_train, y_train, X_test, y_test, output_path):
    """保存为npz格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    logger.info(f"已保存训练、测试集到: {output_path}")

def split_data_for_federation(X, y, num_clients=3):
    """将数据集随机分为num_clients份 (IID),每个客户端再划分训练集和测试集"""
    logger.info(f"开始为 {num_clients} 个客户端分割 IID 数据集...")
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    
    base_data_dir = os.path.join(PROJECT_ROOT, 'data')
    for i, idx in enumerate(split_indices):
        client_id = i + 1   
        client_dir = os.path.join(base_data_dir, f'client{client_id}')
        os.makedirs(client_dir, exist_ok=True)
        
        # 获取该客户端的数据
        X_part = X[idx]
        y_part = y[idx]
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y_part, test_size=TEST_SIZE, random_state=42
        )
        
        # 保存训练集
        train_path = os.path.join(client_dir, 'mnist_train.npz')
        np.savez_compressed(train_path, X_train=X_train, y_train=y_train)
        
        # 保存测试集
        test_path = os.path.join(client_dir, 'mnist_test.npz')
        np.savez_compressed(test_path, X_test=X_test, y_test=y_test)
        
        logger.info(f"客户端{client_id} (IID) 数据已保存:")
        logger.info(f"  - 训练集: {train_path}, 样本数: {len(X_train)}")
        logger.info(f"  - 测试集: {test_path}, 样本数: {len(X_test)}")
    
    logger.info(f"为 {num_clients} 个客户端分割 IID 数据集完成。")

def split_data_non_iid_label_skew(X, y, num_clients=3, label_map=None):
    """
    根据标签倾斜 (Label Skew) 的方式为客户端分割 Non-IID 数据集。
    
    Args:
        X (np.array): 特征数据。
        y (np.array): 标签数据。
        num_clients (int): 客户端数量。
        label_map (dict, optional): 指定每个客户端分配的标签。默认为None,
                                    如果为None, 则使用预设的分配规则。
    """
    if label_map is None:
        # 预设的标签分配规则
        if num_clients == 3:
            label_map = {1: [0, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8, 9]}
        else:
            # 如果客户端数量不是3, 可以自定义其他分配规则或抛出错误
            raise ValueError(f"当前预设只支持3个客户端, 请为 {num_clients} 个客户端提供 label_map")
            
    logger.info(f"开始为 {num_clients} 个客户端分割 Non-IID (Label Skew) 数据集...")
    base_data_dir = os.path.join(PROJECT_ROOT, 'data')

    for client_id, labels in label_map.items():
        client_dir = os.path.join(base_data_dir, f'client{client_id}')
        os.makedirs(client_dir, exist_ok=True)

        # 筛选出包含指定标签的数据
        mask = np.isin(y, labels)
        X_part = X[mask]
        y_part = y[mask]

        if len(X_part) == 0:
            logger.warning(f"客户端 {client_id} 没有分配到任何数据，标签为 {labels}")
            continue

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y_part, test_size=TEST_SIZE, random_state=42, stratify=y_part
        )

        # 保存训练集
        train_path = os.path.join(client_dir, 'mnist_train_noniid_label_skew.npz')
        np.savez_compressed(train_path, X_train=X_train, y_train=y_train)

        # 保存测试集
        test_path = os.path.join(client_dir, 'mnist_test_noniid_label_skew.npz')
        np.savez_compressed(test_path, X_test=X_test, y_test=y_test)

        logger.info(f"客户端{client_id} (Non-IID) 数据已保存 (标签: {labels}):")
        logger.info(f"  - 训练集: {train_path}, 样本数: {len(X_train)}")
        logger.info(f"  - 测试集: {test_path}, 样本数: {len(X_test)}")

    logger.info(f"为 {num_clients} 个客户端分割 Non-IID 数据集完成。")


def split_data_non_iid_quantity_skew(X, y, num_clients=3, proportions=None):
    """
    根据数量倾斜 (Quantity Skew) 的方式为客户端分割 Non-IID 数据集。

    Args:
        X (np.array): 特征数据。
        y (np.array): 标签数据。
        num_clients (int): 客户端数量。
        proportions (list, optional): 指定每个客户端的数据量比例。默认为None,
                                      如果为None, 则使用预设的分配规则。
    """
    if proportions is None:
        if num_clients == 3:
            proportions = [0.6, 0.3, 0.1]  # 客户端1: 60%, 客户端2: 30%, 客户端3: 10%
        else:
            raise ValueError(f"请为 {num_clients} 个客户端提供 proportions 分配比例")

    if abs(sum(proportions) - 1.0) > 1e-8:
        raise ValueError("所有客户端的比例之和必须为1")
    if len(proportions) != num_clients:
        raise ValueError("proportions 列表的长度必须等于客户端数量")

    logger.info(f"开始为 {num_clients} 个客户端分割 Non-IID (Quantity Skew) 数据集...")
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 根据比例切分索引
    split_points = np.cumsum([int(p * n_samples) for p in proportions[:-1]])
    client_indices = np.split(indices, split_points)

    base_data_dir = os.path.join(PROJECT_ROOT, 'data')
    for i, idx in enumerate(client_indices):
        client_id = i + 1
        client_dir = os.path.join(base_data_dir, f'client{client_id}')
        os.makedirs(client_dir, exist_ok=True)

        X_part, y_part = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y_part, test_size=TEST_SIZE, random_state=42
        )

        train_path = os.path.join(client_dir, 'mnist_train_noniid_quantity_skew.npz')
        np.savez_compressed(train_path, X_train=X_train, y_train=y_train)

        test_path = os.path.join(client_dir, 'mnist_test_noniid_quantity_skew.npz')
        np.savez_compressed(test_path, X_test=X_test, y_test=y_test)

        logger.info(f"客户端{client_id} (Non-IID, Quantity Skew) 数据已保存 (比例: {proportions[i]*100:.0f}%):")
        logger.info(f"  - 训练集: {train_path}, 样本数: {len(X_train)}")
        logger.info(f"  - 测试集: {test_path}, 样本数: {len(X_test)}")

    logger.info(f"为 {num_clients} 个客户端分割 Non-IID (Quantity Skew) 数据集完成。")


def cleanup_raw_data(directory):
    """清理原始数据目录"""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            logger.info(f"已成功清理原始数据目录: {directory}")
        except OSError as e:
            logger.error(f"清理目录 {directory} 时出错: {e.strerror}")


def generate_mnist_data():
    logger.info("开始加载MNIST数据集...")
    X_train, y_train, X_test, y_test = load_mnist_data(DOWNLOAD_DIR, normalize=NORMALIZE)
    
    # 保存完整的训练集和测试集
    complete_data_dir = os.path.join(PROJECT_ROOT, 'data', 'complete')
    os.makedirs(complete_data_dir, exist_ok=True)
    
    # 保存完整训练集
    train_path = os.path.join(complete_data_dir, 'mnist_train.npz')
    np.savez_compressed(train_path, X_train=X_train, y_train=y_train)
    logger.info(f"完整训练集已保存: {train_path}, 样本数: {len(X_train)}")
    
    # 保存完整测试集
    test_path = os.path.join(complete_data_dir, 'mnist_test.npz')
    np.savez_compressed(test_path, X_test=X_test, y_test=y_test)
    logger.info(f"完整测试集已保存: {test_path}, 样本数: {len(X_test)}")
    
    # 合并训练集和测试集用于联邦学习划分
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    
    # 1. 为联邦学习划分 IID 数据
    split_data_for_federation(X, y, num_clients=3)
    
    # 2. 为联邦学习划分 Non-IID (Label Skew) 数据
    split_data_non_iid_label_skew(X, y, num_clients=3)
    
    # 3. 为联邦学习划分 Non-IID (Quantity Skew) 数据
    split_data_non_iid_quantity_skew(X, y, num_clients=3)

    logger.info("MNIST数据处理完成！")

if __name__ == '__main__':
    logger.info("开始执行MNIST数据处理脚本...")
    generate_mnist_data()
    
    logger.info("开始清理原始下载文件...")
    cleanup_raw_data(DOWNLOAD_DIR)
    
    logger.info("脚本执行完毕。") 