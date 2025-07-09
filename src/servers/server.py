import torch
from sklearn.metrics import roc_auc_score

class Server:
    def __init__(self, clients, model, config):
        self.clients = clients  # 客户端列表
        self.model = model      # 全局模型
        self.config = config

    def aggregate(self, client_parameters_list):
        """FedAvg参数聚合（简单平均）"""
        # 假设所有参数结构一致
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

    def collect_and_evaluate(self, clients_test_data, local_epochs):
        """让每个客户端本地训练并评估，收集参数、准确率、AUC和损失"""
        client_params_list = []
        client_accs = []
        client_aucs = []
        client_losses = []
        for client, client_test in zip(self.clients, clients_test_data):
            for _ in range(local_epochs):
                client.local_train()
            # 评估
            loss, acc, auc = self._evaluate_with_auc(client, client_test)
            client_accs.append(acc)
            client_aucs.append(auc)
            client_losses.append(loss)
            client_params_list.append(client.get_parameters())
        return client_params_list, client_accs, client_aucs, client_losses

    def _evaluate_with_auc(self, client, test_data):
        """评估loss, acc, auc"""
        import torch.nn.functional as F
        model = client.model
        model.eval()
        device = client.device
        x_test = torch.tensor(test_data.x, dtype=torch.float32)
        y_test = torch.tensor(test_data.y, dtype=torch.long)
        if x_test.ndim == 3:
            x_test = x_test.unsqueeze(1)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test),
            batch_size=client.config.get('eval_batch_size', 128), shuffle=False)
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
                # 取概率用于AUC
                probs = F.softmax(outputs, dim=1)
                if probs.shape[1] == 2:
                    all_probs.extend(probs[:, 1].cpu().numpy())
                else:
                    # 多分类AUC用macro
                    all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        avg_loss = total_loss / total
        acc = correct / total
        try:
            if probs.shape[1] == 2:
                auc = roc_auc_score(all_labels, all_probs)
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
        except Exception:
            auc = float('nan')
        return avg_loss, acc, auc

    def federated_train_loop(self, clients_test_data, rounds, local_epochs, threshold, logger):
        """联邦训练主循环，自动收敛和最佳模型维护，统计平均AUC和损失"""
        acc_history = []
        auc_history = []
        loss_history = []
        best_acc = 0
        best_state_dict = None
        for r in range(rounds):
            logger.info(f"[Federated][Round {r+1}/{rounds}] 开始")
            self.distribute()
            client_params_list, client_accs, client_aucs, client_losses = self.collect_and_evaluate(clients_test_data, local_epochs)
            self.aggregate(client_params_list)
            avg_acc = sum(client_accs) / len(client_accs)
            avg_auc = sum(client_aucs) / len(client_aucs)
            avg_loss = sum(client_losses) / len(client_losses)
            acc_history.append(avg_acc)
            auc_history.append(avg_auc)
            loss_history.append(avg_loss)
            logger.info(f"[Federated][Round {r+1}/{rounds}] 客户端平均准确率: {avg_acc:.4f} 平均AUC: {avg_auc:.4f} 平均损失: {avg_loss:.4f}")
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_auc = avg_auc
                best_loss = avg_loss
                best_state_dict = self.model.state_dict()
            if len(acc_history) >= 4:
                recent_accs = acc_history[-4:]
                if max(recent_accs) - min(recent_accs) < threshold:
                    logger.info(f"[Federated] 收敛检测：最近三轮平均准确率波动({max(recent_accs) - min(recent_accs):.6f})未超过阈值({threshold})，提前停止训练。")
                    break
            logger.info(f"[Federated][Round {r+1}/{rounds}] 聚合完成")
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        logger.info(f"[Federated] 训练完成，平均准确率: {best_acc:.4f} 平均AUC: {best_auc:.4f} 平均损失: {best_loss:.4f}")
