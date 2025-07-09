import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class Client:
    def __init__(self, client_id, data, model, config):
        self.client_id = client_id
        self.data = data  # 本地数据
        self.model = model  # 模型副本
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()
        # 数据准备
        x_train = torch.tensor(self.data.x, dtype=torch.float32)
        y_train = torch.tensor(self.data.y, dtype=torch.long)
        if x_train.ndim == 3:
            x_train = x_train.unsqueeze(1)
        self.train_dataset = TensorDataset(x_train, y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)

    def set_parameters(self, parameters):
        """设置本地模型参数"""
        self.model.load_state_dict(parameters)

    def get_parameters(self):
        """获取本地模型参数"""
        return self.model.state_dict()

    def local_train(self):
        """本地训练一轮"""
        self.model.train()
        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, test_data):
        """在本地或全局测试集上评估模型，返回loss和acc"""
        self.model.eval()
        x_test = torch.tensor(test_data.x, dtype=torch.float32)
        y_test = torch.tensor(test_data.y, dtype=torch.long)
        if x_test.ndim == 3:
            x_test = x_test.unsqueeze(1)
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config.get('eval_batch_size', 128), shuffle=False)
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
        avg_loss = total_loss / total
        acc = correct / total
        return avg_loss, acc
