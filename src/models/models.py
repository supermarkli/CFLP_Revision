import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    """ResNet 基本块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """适用于 CIFAR-10 的 ResNet18 模型"""
    def __init__(self, in_features=3, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_features, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_parameters(self):
        """获取模型参数，全部转为numpy数组"""
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}

    def set_parameters(self, parameters):
        """用numpy数组字典设置模型参数"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param.data = torch.from_numpy(parameters[name].copy()).to(param.data.device).type(param.data.dtype)


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