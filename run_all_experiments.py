import os
import shutil
import subprocess
import yaml
import time
import logging
import glob
import re
import json
import ast

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.join(PROJECT_ROOT, 'src', 'default.yaml')
MAIN_PY = os.path.join(PROJECT_ROOT, 'src', 'main.py')
OUT_DIR = os.path.join(PROJECT_ROOT, 'out')
STATE_FILE = os.path.join(OUT_DIR, 'experiment_state.json')

import sys
sys.path.append(PROJECT_ROOT)
from src.utils.draw import plot_experiment_results_bar

# 所有模型类型（保留完整列表，方便后续扩展）
MODEL_TYPES = ['CNN', 'ResNet18', 'KNN', 'RF', 'SVC', 'LR']
# 支持的实验模式
MODES = ['Centralized', 'Federated']
# 传统ML模型
ML_MODELS = ['KNN', 'RF', 'SVC', 'LR']

# 本次论文实验中使用的 4 组联邦实验（Sim-1 ~ Sim-4）
FEDERATED_EXPERIMENTS = [
    # Sim-1: 轻量级基准：MNIST + SimpleCNN (这里对应 CNN) + IID
    {'name': 'Sim-1', 'dataset': 'mnist', 'model': 'CNN', 'dist': 'iid'},
    # Sim-2: 轻量级异质：MNIST + SimpleCNN + Non-IID (Label Skew)
    {'name': 'Sim-2', 'dataset': 'mnist', 'model': 'CNN', 'dist': 'noniid_dirichlet'},
    # Sim-3: 重量级基准：CIFAR-10 + ResNet18 + IID
    {'name': 'Sim-3', 'dataset': 'cifar10', 'model': 'ResNet18', 'dist': 'iid'},
    # Sim-4: 重量级异质：CIFAR-10 + ResNet18 + Non-IID (Label Skew)
    {'name': 'Sim-4', 'dataset': 'cifar10', 'model': 'ResNet18', 'dist': 'noniid_dirichlet'},
]

CENTRALIZED_EXPERIMENTS = [
    # MNIST + CNN（对应 Sim-1/2 的集中式对照）
    {'name': 'Central-MNIST', 'dataset': 'mnist', 'model': 'CNN'},
    # CIFAR-10 + ResNet18（对应 Sim-3/4 的集中式对照）
    {'name': 'Central-CIFAR10', 'dataset': 'cifar10', 'model': 'ResNet18'},
]

# 每个实验的重复次数
NUM_RUNS = 5

batch_log = os.path.join(OUT_DIR, 'batch.log')
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=batch_log,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8'
)


def run_experiment(mode, model_type, dist_type=None, run_id=None, seed=None, dataset=None):
    """运行单次实验"""
    # 1. 读取配置
    with open(DEFAULT_YAML, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 修改配置
    config['mode'] = mode
    config['model']['type'] = model_type
    
    # 根据实验设计修改数据集
    if dataset is not None:
        if 'data' not in config:
            config['data'] = {}
        config['data']['dataset'] = dataset
        
        # 根据数据集设置学习率
        if 'training' not in config:
            config['training'] = {}
        if dataset == 'mnist':
            config['training']['learning_rate'] = 0.1
        elif dataset == 'cifar10':
            config['training']['learning_rate'] = 0.01
    
    # 联邦学习下设置数据分布类型
    if mode == 'Federated' and dist_type:
        if 'data' not in config:
            config['data'] = {}
        config['data']['federated_dist'] = dist_type
    
    if seed is not None:
        config['seed'] = seed
    
    # 3. 保存配置
    with open(DEFAULT_YAML, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # 4. 运行实验
    subprocess.run(['python', MAIN_PY], cwd=PROJECT_ROOT)
    
    # 只复制最新的日志文件并重命名
    logs_dir = os.path.join(PROJECT_ROOT, 'logs')
    if os.path.exists(logs_dir):
        log_files = glob.glob(os.path.join(logs_dir, '*.log'))
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            run_suffix = f'_run_{run_id}' if run_id is not None else ''
            if mode == 'Federated':
                new_log_name = f'{mode}_{model_type}_{dist_type}{run_suffix}.log'
            else:
                new_log_name = f'{mode}_{model_type}{run_suffix}.log'
            new_log_path = os.path.join(OUT_DIR, new_log_name)
            shutil.move(latest_log, new_log_path)
            logging.info(f'实验 {new_log_name} 日志文件: {new_log_path}')
    
    # 清理 logs 目录
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)


def parse_log_file(log_path):
    """
    解析日志文件，提取实验指标。
    
    Returns:
        dict: 包含所有实验指标的字典
    """
    metrics = {
        'mode': None,
        'model': None,
        'dist': None,
        'acc': None,
        'auc': None,
        'time_cost': None,
        'convergence_round': None,
        'communication_volume': None,
        'final_acc_std': None,
        'local_global_gap': None,
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 基本信息
            if '实验模式:' in line:
                match = re.search(r'实验模式:\s*(\w+)', line)
                if match:
                    metrics['mode'] = match.group(1)
            
            elif '为联邦学习加载数据' in line and '分布类型:' in line:
                match = re.search(r'分布类型:\s*(\w+)', line)
                if match:
                    metrics['dist'] = match.group(1)
            
            elif '模型初始化完成:' in line:
                match = re.search(r'模型初始化完成:\s*(\w+)', line)
                if match:
                    metrics['model'] = match.group(1)
            
            # 最终结果
            elif '最终准确率:' in line:
                match = re.search(r'最终准确率:\s*([0-9.]+)', line)
                if match:
                    metrics['acc'] = float(match.group(1))
            
            elif '最终AUC:' in line:
                match = re.search(r'最终AUC:\s*([0-9.]+)', line)
                if match:
                    metrics['auc'] = float(match.group(1))
                elif '计算失败' in line or 'N/A' in line:
                    metrics['auc'] = None
            
            elif '训练总耗时:' in line:
                match = re.search(r'训练总耗时:\s*([0-9.]+)', line)
                if match:
                    metrics['time_cost'] = float(match.group(1))
            
            # 新增指标
            elif '收敛轮数:' in line:
                match = re.search(r'收敛轮数:\s*(\d+)', line)
                if match:
                    metrics['convergence_round'] = int(match.group(1))
            
            elif '总通信体积:' in line:
                # 解析如 "总通信体积: 123.45 MB"
                match = re.search(r'总通信体积:\s*([0-9.]+)\s*(B|KB|MB|GB|TB)', line)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    # 转换为字节
                    unit_multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
                    metrics['communication_volume'] = value * unit_multipliers.get(unit, 1)
            
            elif '最后' in line and '轮准确率标准差:' in line:
                match = re.search(r'准确率标准差:\s*([0-9.]+)', line)
                if match:
                    metrics['final_acc_std'] = float(match.group(1))
            
            elif 'Local-Global Gap:' in line:
                match = re.search(r'Local-Global Gap:\s*([0-9.]+)', line)
                if match:
                    metrics['local_global_gap'] = float(match.group(1))
    
    return metrics


def format_bytes_display(num_bytes):
    """格式化字节数为人类可读的格式"""
    if num_bytes is None:
        return 'N/A'
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def main():
    import numpy as np
    
    # 备份 default.yaml
    backup_yaml = DEFAULT_YAML + '.bak'
    shutil.copy(DEFAULT_YAML, backup_yaml)
    
    # 加载或初始化实验状态
    if os.path.exists(STATE_FILE):
        logging.info(f'从 {STATE_FILE} 加载之前的实验状态')
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                loaded_dict = json.load(f)
                experiment_runs = {ast.literal_eval(k): v for k, v in loaded_dict.items()}
        except (json.JSONDecodeError, SyntaxError) as e:
            logging.warning(f'无法解析状态文件 {STATE_FILE}: {e}，将重新开始所有实验')
            experiment_runs = {}
    else:
        experiment_runs = {}

    try:
        for mode in MODES:
            if mode == 'Federated':
                # 只运行论文中设计的 4 组联邦实验（Sim-1 ~ Sim-4）
                for exp in FEDERATED_EXPERIMENTS:
                    model_type = exp['model']
                    dist_type = exp['dist']
                    dataset = exp['dataset']
                    exp_key = (mode, model_type, dist_type, dataset)
                    experiment_runs.setdefault(exp_key, [])
                    
                    for i in range(NUM_RUNS):
                        if i < len(experiment_runs[exp_key]):
                            logging.info(
                                f'跳过已完成的实验: name={exp["name"]}, mode={mode}, model={model_type}, '
                                f'dataset={dataset}, dist={dist_type}, run={i+1}/{NUM_RUNS}'
                            )
                            continue

                        logging.info(
                            f'运行实验: name={exp["name"]}, mode={mode}, model={model_type}, '
                            f'dataset={dataset}, dist={dist_type}, run={i+1}/{NUM_RUNS}'
                        )
                        run_experiment(mode, model_type, dist_type=dist_type, run_id=i, seed=i, dataset=dataset)
                        time.sleep(1)
                        
                        log_name = f'{mode}_{model_type}_{dist_type}_run_{i}.log'
                        log_path = os.path.join(OUT_DIR, log_name)
                        if os.path.exists(log_path):
                            parsed = parse_log_file(log_path)
                            # 检查解析是否成功（至少需要有准确率和时间）
                            if parsed['acc'] is not None and parsed['time_cost'] is not None:
                                experiment_runs[exp_key].append(parsed)
                                with open(STATE_FILE, 'w', encoding='utf-8') as f:
                                    json.dump({str(k): v for k, v in experiment_runs.items()}, f, ensure_ascii=False, indent=4)
                                logging.info(f'实验状态已更新并保存到 {STATE_FILE}')

            elif mode == 'Centralized':
                # 只运行与上述 4 组联邦实验对应的 2 组集中式实验
                for exp in CENTRALIZED_EXPERIMENTS:
                    model_type = exp['model']
                    dataset = exp['dataset']
                    exp_key = (mode, model_type, 'N/A', dataset)
                    experiment_runs.setdefault(exp_key, [])
                    
                    for i in range(NUM_RUNS):
                        if i < len(experiment_runs[exp_key]):
                            logging.info(
                                f'跳过已完成的实验: name={exp["name"]}, mode={mode}, model={model_type}, '
                                f'dataset={dataset}, run={i+1}/{NUM_RUNS}'
                            )
                            continue

                        logging.info(
                            f'运行实验: name={exp["name"]}, mode={mode}, model={model_type}, '
                            f'dataset={dataset}, run={i+1}/{NUM_RUNS}'
                        )
                        run_experiment(mode, model_type, run_id=i, seed=i, dataset=dataset)
                        time.sleep(1)
                        
                        log_name = f'{mode}_{model_type}_run_{i}.log'
                        log_path = os.path.join(OUT_DIR, log_name)
                        if os.path.exists(log_path):
                            parsed = parse_log_file(log_path)
                            if parsed['acc'] is not None and parsed['time_cost'] is not None:
                                experiment_runs[exp_key].append(parsed)
                                with open(STATE_FILE, 'w', encoding='utf-8') as f:
                                    json.dump({str(k): v for k, v in experiment_runs.items()}, f, ensure_ascii=False, indent=4)
                                logging.info(f'实验状态已更新并保存到 {STATE_FILE}')
                                
    finally:
        # 恢复 default.yaml
        if os.path.exists(backup_yaml):
            shutil.move(backup_yaml, DEFAULT_YAML)
            logging.info('所有实验完成，default.yaml 已恢复')
        else:
            logging.warning(f'备份文件 {backup_yaml} 不存在，跳过恢复步骤')

        # ========== 计算均值和标准差 ==========
        results_summary = []
        
        for exp_key, runs in experiment_runs.items():
            if not runs:
                continue
            
            mode, model, dist, dataset = exp_key
            
            # 提取每次运行的指标
            accs = [r['acc'] for r in runs if r.get('acc') is not None]
            aucs = [r['auc'] for r in runs if r.get('auc') is not None]
            times = [r['time_cost'] for r in runs if r.get('time_cost') is not None]
            conv_rounds = [r['convergence_round'] for r in runs if r.get('convergence_round') is not None]
            comm_volumes = [r['communication_volume'] for r in runs if r.get('communication_volume') is not None]
            acc_stds = [r['final_acc_std'] for r in runs if r.get('final_acc_std') is not None]
            lg_gaps = [r['local_global_gap'] for r in runs if r.get('local_global_gap') is not None]

            # 计算均值和标准差
            result = {
                'mode': mode,
                'model': model,
                'dist': dist,
                'dataset': dataset,
                'acc_mean': np.mean(accs) if accs else 0,
                'acc_std': np.std(accs) if accs else 0,
                'auc_mean': np.mean(aucs) if aucs else None,
                'auc_std': np.std(aucs) if aucs else None,
                'time_mean': np.mean(times) if times else 0,
                'time_std': np.std(times) if times else 0,
                'conv_round_mean': np.mean(conv_rounds) if conv_rounds else None,
                'conv_round_std': np.std(conv_rounds) if conv_rounds else None,
                'comm_volume_mean': np.mean(comm_volumes) if comm_volumes else None,
                'comm_volume_std': np.std(comm_volumes) if comm_volumes else None,
                'final_acc_std_mean': np.mean(acc_stds) if acc_stds else None,
                'final_acc_std_std': np.std(acc_stds) if acc_stds else None,
                'lg_gap_mean': np.mean(lg_gaps) if lg_gaps else None,
                'lg_gap_std': np.std(lg_gaps) if lg_gaps else None,
            }
            results_summary.append(result)

        # ========== 写入实验结果CSV ==========
        import csv
        result_csv = os.path.join(OUT_DIR, 'experiment_results.csv')
        
        with open(result_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                '模式', '模型', '数据分布', '数据集',
                '准确率 (均值±标准差)', 
                'AUC (均值±标准差)', 
                '收敛轮数 (均值±标准差)',
                '通信体积 (均值)',
                '最后10轮准确率标准差 (均值)',
                'Local-Global Gap (均值±标准差)',
                '训练总耗时 (均值±标准差)'
            ])
            
            for r in results_summary:
                # 格式化各个指标
                acc_str = f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}"
                auc_str = f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}" if r['auc_mean'] is not None else "N/A"
                time_str = f"{r['time_mean']:.2f}±{r['time_std']:.2f}"
                
                conv_str = f"{r['conv_round_mean']:.1f}±{r['conv_round_std']:.1f}" if r['conv_round_mean'] is not None else "N/A"
                comm_str = format_bytes_display(r['comm_volume_mean']) if r['comm_volume_mean'] is not None else "N/A"
                acc_std_str = f"{r['final_acc_std_mean']:.6f}" if r['final_acc_std_mean'] is not None else "N/A"
                lg_gap_str = f"{r['lg_gap_mean']:.4f}±{r['lg_gap_std']:.4f}" if r['lg_gap_mean'] is not None else "N/A"
                
                writer.writerow([
                    r['mode'], r['model'], r['dist'], r['dataset'],
                    acc_str, auc_str, conv_str, comm_str, acc_std_str, lg_gap_str, time_str
                ])
        
        logging.info(f'实验结果已保存到 {result_csv}')
        
        # ========== 准备数据用于绘图 ==========
        plot_data = []
        for r in results_summary:
            plot_data.append([
                r['mode'], r['model'], r['dist'],
                r['acc_mean'], r['acc_std'],
                r['auc_mean'] if r['auc_mean'] is not None else np.nan,
                r['auc_std'] if r['auc_std'] is not None else np.nan,
                r['time_mean'], r['time_std']
            ])

        # 绘制实验结果柱状图
        if plot_data:
            plot_experiment_results_bar(OUT_DIR, plot_data)
        else:
            plot_experiment_results_bar(OUT_DIR)
        
        # ========== 打印详细结果表格 ==========
        print("\n" + "=" * 120)
        print("实验结果汇总")
        print("=" * 120)
        print(f"{'实验名称':<20} {'准确率':<18} {'收敛轮数':<12} {'通信体积':<15} {'最后10轮Acc Std':<18} {'L-G Gap':<15}")
        print("-" * 120)
        
        for r in results_summary:
            exp_name = f"{r['mode']}-{r['model']}-{r['dist']}"
            acc_str = f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}"
            conv_str = f"{r['conv_round_mean']:.1f}" if r['conv_round_mean'] is not None else "N/A"
            comm_str = format_bytes_display(r['comm_volume_mean']) if r['comm_volume_mean'] is not None else "N/A"
            acc_std_str = f"{r['final_acc_std_mean']:.6f}" if r['final_acc_std_mean'] is not None else "N/A"
            lg_gap_str = f"{r['lg_gap_mean']:.4f}" if r['lg_gap_mean'] is not None else "N/A"
            
            print(f"{exp_name:<20} {acc_str:<18} {conv_str:<12} {comm_str:<15} {acc_std_str:<18} {lg_gap_str:<15}")
        
        print("=" * 120)


if __name__ == '__main__':
    main()
