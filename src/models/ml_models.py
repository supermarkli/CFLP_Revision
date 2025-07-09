import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sys
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.logging_config import get_logger
logger = get_logger()

def preprocess_data(x):
    """预处理输入数据
    
    Args:
        x: numpy.ndarray 或 torch.Tensor, 输入数据
        
    Returns:
        numpy.ndarray: 预处理后的数据
    """
    # 转换为NumPy数组
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    
    # 保存原始形状
    original_shape = x.shape
    
    # 展平图像数据
    if x.ndim > 2:
        if x.ndim == 4:  # (N, C, H, W) 格式
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1)  # 转换为 (N, H, W, C)
            x = x.reshape(N, -1)  # 展平为 (N, H*W*C)
        else:  # (N, H, W) 格式
            N, H, W = x.shape
            x = x.reshape(N, -1)  # 展平为 (N, H*W)
    
    return x

class SVCWrapper(nn.Module):
    """SVC模型包装类，使其接口与PyTorch模型一致"""
    def __init__(self, C=1.0, class_weight=None, max_iter=1000, tol=0.001, **kwargs):
        super().__init__()
        self.svm = SVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            tol=tol,
            **kwargs
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = None

    def fit(self, x, y):
        """训练SVM模型
        
        Args:
            x: torch.Tensor 或 numpy.ndarray, 训练数据
            y: torch.Tensor 或 numpy.ndarray, 训练标签
        """
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        x = preprocess_data(x)
        x = self.scaler.fit_transform(x)
        self.svm.fit(x, y)
        self.is_fitted = True
        self.n_classes = len(np.unique(y))
        logger.info(f"SVM 训练完成，迭代次数: [{self.svm.n_iter_}]")

    def forward(self, x):
        """前向传播，返回预测概率
        
        Args:
            x: torch.Tensor, shape (batch_size, channels, height, width) 或 (batch_size, features)
            
        Returns:
            torch.Tensor: shape (batch_size, n_classes) 的预测概率
        """
        if not self.is_fitted:
            raise RuntimeError("需要先调用fit方法训练模型")
        x = preprocess_data(x)
        x = self.scaler.transform(x)
        scores = self.svm.decision_function(x)
        import scipy.special
        probs = scipy.special.softmax(scores, axis=1)
        return torch.from_numpy(probs).float().to(self.device)

    def state_dict(self):
        """返回模型状态，用于保存模型"""
        import pickle
        return {
            'svm_state': pickle.dumps(self.svm),
            'scaler_state': pickle.dumps(self.scaler),
            'is_fitted': self.is_fitted,
            'n_classes': self.n_classes
        }
        
    def load_state_dict(self, state_dict):
        """加载模型状态
        
        Args:
            state_dict: 由state_dict()方法返回的状态字典
        """
        import pickle
        self.svm = pickle.loads(state_dict['svm_state'])
        self.scaler = pickle.loads(state_dict['scaler_state'])
        self.is_fitted = state_dict['is_fitted']
        self.n_classes = state_dict['n_classes']
        
    def to(self, device):
        """兼容PyTorch的device转换接口"""
        self.device = device
        return self
        
    def train(self, mode=True):
        """兼容PyTorch的训练模式切换接口"""
        return self
        
    def eval(self):
        """兼容PyTorch的评估模式切换接口"""
        return self

class LRWrapper(nn.Module):
    """逻辑回归模型包装类，使其接口与PyTorch模型一致"""
    def __init__(self, C=1.0, class_weight=None, max_iter=1000, tol=0.001, **kwargs):
        super().__init__()
        self.lr = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            tol=tol,
            solver='lbfgs',
            **kwargs
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = None

    def fit(self, x, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        x = preprocess_data(x)
        x = self.scaler.fit_transform(x)
        self.lr.fit(x, y)
        self.is_fitted = True
        self.n_classes = len(np.unique(y))
        logger.info(f"LR 训练完成，迭代次数: {self.lr.n_iter_}")

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("需要先调用fit方法训练模型")
        x = preprocess_data(x)
        x = self.scaler.transform(x)
        probs = self.lr.predict_proba(x)
        return torch.from_numpy(probs).float().to(self.device)

    def state_dict(self):
        import pickle
        return {
            'lr_state': pickle.dumps(self.lr),
            'scaler_state': pickle.dumps(self.scaler),
            'is_fitted': self.is_fitted,
            'n_classes': self.n_classes
        }
    def load_state_dict(self, state_dict):
        import pickle
        self.lr = pickle.loads(state_dict['lr_state'])
        self.scaler = pickle.loads(state_dict['scaler_state'])
        self.is_fitted = state_dict['is_fitted']
        self.n_classes = state_dict['n_classes']
    def to(self, device):
        self.device = device
        return self
    def train(self, mode=True):
        return self
    def eval(self):
        return self

class KNNWrapper(nn.Module):
    """KNN模型包装类，使其接口与PyTorch模型一致"""
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', **kwargs):
        super().__init__()
        self.KNN = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            **kwargs
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = None

    def fit(self, x, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        x = preprocess_data(x)
        x = self.scaler.fit_transform(x)
        self.KNN.fit(x, y)
        self.is_fitted = True
        self.n_classes = len(np.unique(y))
        logger.info(f"KNN 训练完成，样本数: {x.shape[0]}")

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("需要先调用fit方法训练模型")
        x = preprocess_data(x)
        x = self.scaler.transform(x)
        probs = self.KNN.predict_proba(x)
        return torch.from_numpy(probs).float().to(self.device)

    def state_dict(self):
        import pickle
        return {
            'KNN_state': pickle.dumps(self.KNN),
            'scaler_state': pickle.dumps(self.scaler),
            'is_fitted': self.is_fitted,
            'n_classes': self.n_classes
        }
    def load_state_dict(self, state_dict):
        import pickle
        self.KNN = pickle.loads(state_dict['KNN_state'])
        self.scaler = pickle.loads(state_dict['scaler_state'])
        self.is_fitted = state_dict['is_fitted']
        self.n_classes = state_dict['n_classes']
    def to(self, device):
        self.device = device
        return self
    def train(self, mode=True):
        return self
    def eval(self):
        return self

class RFWrapper(nn.Module):
    """随机森林模型包装类，使其接口与PyTorch模型一致"""
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', class_weight=None, **kwargs):
        super().__init__()
        self.RF = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            **kwargs
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = None

    def fit(self, x, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        x = preprocess_data(x)
        x = self.scaler.fit_transform(x)
        self.RF.fit(x, y)
        self.is_fitted = True
        self.n_classes = len(np.unique(y))
        logger.info(f"RF 训练完成，树数: {self.RF.n_estimators}")

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("需要先调用fit方法训练模型")
        x = preprocess_data(x)
        x = self.scaler.transform(x)
        probs = self.RF.predict_proba(x)
        return torch.from_numpy(probs).float().to(self.device)

    def state_dict(self):
        import pickle
        return {
            'RF_state': pickle.dumps(self.RF),
            'scaler_state': pickle.dumps(self.scaler),
            'is_fitted': self.is_fitted,
            'n_classes': self.n_classes
        }
    def load_state_dict(self, state_dict):
        import pickle
        self.RF = pickle.loads(state_dict['RF_state'])
        self.scaler = pickle.loads(state_dict['scaler_state'])
        self.is_fitted = state_dict['is_fitted']
        self.n_classes = state_dict['n_classes']
    def to(self, device):
        self.device = device
        return self
    def train(self, mode=True):
        return self
    def eval(self):
        return self

def create_ml_model(model_type='SVC', **kwargs):
    """创建传统机器学习模型的工厂函数
    
    Args:
        model_type: str, 模型类型 ('svm', 'KNN', 'RF' 等)
        **kwargs: 模型的具体参数
        
    Returns:
        nn.Module: 包装后的模型实例
    """
    if model_type == 'SVC':
        return SVCWrapper(**kwargs)
    elif model_type == 'LR':
        return LRWrapper(**kwargs)
    elif model_type == 'KNN':
        return KNNWrapper(**kwargs)
    elif model_type == 'RF':
        return RFWrapper(**kwargs)
    else:
        raise ValueError(f"未支持的模型类型: {model_type}") 