import torch
import torch.nn.functional as F
from torch import nn

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

    def get_parameters(self):
        """
        获取模型参数，全部转为numpy数组
        """
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}

    def set_parameters(self, parameters):
        """
        用numpy数组字典设置模型参数
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param.data = torch.from_numpy(parameters[name].copy()).to(param.data.device).type(param.data.dtype)

class MLP(nn.Module):
    """多层感知机（MLP）模型，支持自定义隐藏层"""
    def __init__(self, input_dim=784, hidden_dims=[128, 64], num_classes=10, activation='relu'):
        super().__init__()
        layers = []
        last_dim = input_dim
        act_layer = nn.ReLU if activation == 'relu' else nn.Tanh
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(act_layer())
            last_dim = h
        layers.append(nn.Linear(last_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)