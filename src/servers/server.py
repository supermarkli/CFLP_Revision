import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

class Server:
    def __init__(self, clients, model, config):
        self.clients = clients  # 客户端列表
        self.model = model      # 全局模型
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = self.device.type == 'cuda'
        self.model = self.model.to(self.device)
        
        # 混合精度训练设置
        self.use_amp = config['training'].get('use_amp', True) and self.use_cuda
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 计算模型参数大小（用于通信体积计算）
        self.model_size_bytes = self._calculate_model_size()

    def _calculate_model_size(self):
        """计算模型参数的字节大小"""
        total_bytes = 0
        for param in self.model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes

    def get_communication_volume_per_round(self):
        """
        计算每轮通信体积（字节）。
        每轮通信包括：
        - 服务器 -> 客户端：广播全局模型（1次模型大小 × 客户端数量）
        - 客户端 -> 服务器：上传本地模型（1次模型大小 × 客户端数量）
        
        Returns:
            total_bytes: 每轮总通信体积（字节）
        """
        num_clients = len(self.clients)
        # 下行 + 上行
        return 2 * self.model_size_bytes * num_clients

    def aggregate(self, client_parameters_list):
        """FedAvg参数聚合（简单平均）"""
        new_state_dict = {}
        for key in client_parameters_list[0].keys():
            stacked = torch.stack([params[key].float() for params in client_parameters_list], dim=0)
            new_state_dict[key] = torch.mean(stacked, dim=0)
        self.model.load_state_dict(new_state_dict)

    def distribute(self):
        """将全局模型参数分发给所有客户端"""
        global_params = self.get_global_parameters()
        for client in self.clients:
            client.set_parameters(global_params)

    def set_global_parameters(self, parameters):
        """设置全局模型参数"""
        self.model.load_state_dict(parameters)

    def get_global_parameters(self):
        """获取全局模型参数"""
        return self.model.state_dict()

    def collect_parameters_after_training(self, local_epochs, round_num=None):
        """
        让每个客户端进行本地训练，然后收集训练后的模型参数。
        
        Args:
            local_epochs: 每个客户端本地训练轮数
            round_num: 当前联邦学习轮数（用于学习率调度器）
            
        Returns:
            client_params_list: 各客户端训练后的模型参数列表
        """
        client_params_list = []
        for client in self.clients:
            # 本地训练
            for _ in range(local_epochs):
                client.local_train(round_num=round_num)
            # 收集参数
            client_params_list.append(client.get_parameters())
        return client_params_list

    def evaluate_global_model(self, test_data):
        """
        使用全局测试集评估聚合后的全局模型。
        
        Args:
            test_data: SimpleNamespace，包含 x 和 y 属性的全局测试集
            
        Returns:
            avg_loss: 平均损失
            acc: 准确率
            auc: AUC值（如果无法计算则为None）
        """
        self.model.eval()
        
        # 使用 from_numpy 避免数据复制
        x_np = test_data.x.astype(np.float32) if test_data.x.dtype != np.float32 else test_data.x
        y_np = test_data.y.astype(np.int64) if test_data.y.dtype != np.int64 else test_data.y
        x_test = torch.from_numpy(x_np)
        y_test = torch.from_numpy(y_np)
        if x_test.ndim == 3:
            x_test = x_test.unsqueeze(1)
        test_dataset = TensorDataset(x_test, y_test)
        
        # DataLoader 优化
        num_workers = self.config['training'].get('num_workers', 4) if self.use_cuda else 0
        prefetch_factor = self.config['training'].get('prefetch_factor', 2) if num_workers > 0 else None
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training'].get('eval_batch_size', 256),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.use_cuda,
            prefetch_factor=prefetch_factor
        )
        
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        compute_auc = self.config['training'].get('compute_auc', True)
        
        with torch.inference_mode():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=self.use_cuda)
                batch_y = batch_y.to(self.device, non_blocking=self.use_cuda)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                
                if compute_auc:
                    probs = F.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu())
                    all_labels.append(batch_y.cpu())

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
        
        avg_loss = total_loss / total
        acc = correct / total
        
        # 安全地计算AUC
        auc = None
        if compute_auc:
            try:
                y_true = torch.cat(all_labels).numpy()
                y_proba = torch.cat(all_probs).numpy().astype(np.float64)
                y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
                
                n_classes = y_proba.shape[1]
                unique_classes = np.unique(y_true)
                if len(unique_classes) > 1 and len(unique_classes) == n_classes:
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
                elif len(unique_classes) > 1:
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', labels=unique_classes)
            except Exception:
                pass

        return avg_loss, acc, auc

    def evaluate_local_models_on_global_test(self, test_data):
        """
        使用全局测试集评估各客户端的本地模型，用于计算 Local-Global Gap。
        
        Args:
            test_data: SimpleNamespace，包含 x 和 y 属性的全局测试集
            
        Returns:
            local_accs: 各客户端本地模型在全局测试集上的准确率列表
            avg_local_acc: 本地模型平均准确率
        """
        local_accs = []
        for client in self.clients:
            _, acc, _ = client.evaluate(test_data)
            local_accs.append(acc)
        avg_local_acc = sum(local_accs) / len(local_accs) if local_accs else 0.0
        return local_accs, avg_local_acc
