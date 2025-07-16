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
# 所有模型类型
MODEL_TYPES = ['CNN', 'MLP', 'KNN', 'RF', 'SVC', 'LR']
# 支持的实验模式
MODES = ['Centralized', 'Federated']
# 联邦学习数据分布
FEDERATED_DISTS = ['iid', 'noniid_label_skew', 'noniid_quantity_skew']
# 传统ML模型
ML_MODELS = ['KNN', 'RF', 'SVC', 'LR']
# 每个实验的重复次数
NUM_RUNS = 3

batch_log = os.path.join(OUT_DIR, 'batch.log')
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=batch_log,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8'
)

def run_experiment(mode, model_type, dist_type=None, run_id=None, seed=None):
    # 1. 读取配置
    with open(DEFAULT_YAML, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 2. 修改配置
    config['mode'] = mode
    config['model']['type'] = model_type
    if mode == 'Federated' and dist_type:
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
    mode = model = acc = auc = time_cost = dist = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '实验模式:' in line:
                mode_match = re.search(r'实验模式:\s*(\w+)', line)
                if mode_match:
                    mode = mode_match.group(1)
            elif '为联邦学习加载数据，分布类型:' in line:
                dist_match = re.search(r'分布类型:\s*(\w+)', line)
                if dist_match:
                    dist = dist_match.group(1)
            elif '模型初始化完成:' in line:
                model_match = re.search(r'模型初始化完成:\s*(\w+)', line)
                if model_match:
                    model = model_match.group(1)
            elif '最终准确率:' in line:
                acc_match = re.search(r'最终准确率:\s*([0-9.]+)', line)
                if acc_match:
                    acc = acc_match.group(1)
            elif '最终AUC:' in line:
                auc_match = re.search(r'最终AUC:\s*([0-9.]+)', line)
                if auc_match:
                    auc = auc_match.group(1)
                elif '计算失败' in line:
                    auc = '计算失败'
            elif '训练总耗时:' in line:
                time_match = re.search(r'训练总耗时:\s*([0-9.]+)', line)
                if time_match:
                    time_cost = time_match.group(1)
    return mode, model, dist, acc, auc, time_cost

def main():
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
                for model_type in MODEL_TYPES:
                    if model_type in ML_MODELS:
                        continue
                    for dist_type in FEDERATED_DISTS:
                        exp_key = (mode, model_type, dist_type)
                        experiment_runs.setdefault(exp_key, [])
                        for i in range(NUM_RUNS):
                            if i < len(experiment_runs[exp_key]):
                                logging.info(f'跳过已完成的实验: mode={mode}, model={model_type}, dist={dist_type}, run={i+1}/{NUM_RUNS}')
                                continue

                            logging.info(f'运行实验: mode={mode}, model={model_type}, dist={dist_type}, run={i+1}/{NUM_RUNS}')
                            run_experiment(mode, model_type, dist_type=dist_type, run_id=i, seed=i)
                            time.sleep(1)
                            log_name = f'{mode}_{model_type}_{dist_type}_run_{i}.log'
                            log_path = os.path.join(OUT_DIR, log_name)
                            if os.path.exists(log_path):
                                parsed = parse_log_file(log_path)
                                # 检查解析是否成功
                                if all(p is not None for p in [parsed[3], parsed[5]]): # acc, time_cost
                                    experiment_runs[exp_key].append(parsed)
                                    with open(STATE_FILE, 'w', encoding='utf-8') as f:
                                        json.dump({str(k): v for k, v in experiment_runs.items()}, f, ensure_ascii=False, indent=4)
                                    logging.info(f'实验状态已更新并保存到 {STATE_FILE}')

            elif mode == 'Centralized':
                for model_type in MODEL_TYPES:
                    exp_key = (mode, model_type, 'N/A')
                    experiment_runs.setdefault(exp_key, [])
                    for i in range(NUM_RUNS):
                        if i < len(experiment_runs[exp_key]):
                            logging.info(f'跳过已完成的实验: mode={mode}, model={model_type}, run={i+1}/{NUM_RUNS}')
                            continue

                        logging.info(f'运行实验: mode={mode}, model={model_type}, run={i+1}/{NUM_RUNS}')
                        run_experiment(mode, model_type, run_id=i, seed=i)
                        time.sleep(1)
                        log_name = f'{mode}_{model_type}_run_{i}.log'
                        log_path = os.path.join(OUT_DIR, log_name)
                        if os.path.exists(log_path):
                            parsed = parse_log_file(log_path)
                            if all(p is not None for p in [parsed[3], parsed[5]]):
                                experiment_runs[exp_key].append(parsed)
                                with open(STATE_FILE, 'w', encoding='utf-8') as f:
                                    json.dump({str(k): v for k, v in experiment_runs.items()}, f, ensure_ascii=False, indent=4)
                                logging.info(f'实验状态已更新并保存到 {STATE_FILE}')
    finally:
        # 恢复 default.yaml
        shutil.move(backup_yaml, DEFAULT_YAML)
        logging.info('所有实验完成，default.yaml 已恢复')

        # 计算均值和标准差
        import numpy as np
        results_summary = []
        for exp_key, runs in experiment_runs.items():
            if not runs: continue
            
            mode, model, dist = exp_key
            
            # 提取每次运行的指标，转换为浮点数
            accs = [float(r[3]) for r in runs if r[3] is not None]
            aucs = [float(r[4]) for r in runs if r[4] is not None and r[4] != 'pass']
            times = [float(r[5]) for r in runs if r[5] is not None]

            # 计算均值和标准差
            acc_mean = np.mean(accs) if accs else 0
            acc_std = np.std(accs) if accs else 0
            auc_mean = np.mean(aucs) if aucs else 0
            auc_std = np.std(aucs) if aucs else 0
            time_mean = np.mean(times) if times else 0
            time_std = np.std(times) if times else 0
            
            results_summary.append([
                mode, model, dist, 
                f"{acc_mean:.4f}±{acc_std:.4f}",
                f"{auc_mean:.4f}±{auc_std:.4f}" if aucs else "N/A",
                f"{time_mean:.2f}±{time_std:.2f}"
            ])

        # 写入实验结果CSV
        import csv
        result_csv = os.path.join(OUT_DIR, 'experiment_results.csv')
        with open(result_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['模式', '模型', '数据分布', '准确率 (均值±标准差)', 'AUC (均值±标准差)', '训练总耗时 (均值±标准差)'])
            for row in results_summary:
                writer.writerow(row)
        
        # 准备数据用于绘图
        plot_data = []
        for row in results_summary:
            mode, model, dist, acc, auc, time_cost = row
            acc_mean, acc_std = map(float, acc.split('±'))
            if auc != 'N/A':
                auc_mean, auc_std = map(float, auc.split('±'))
            else:
                auc_mean, auc_std = np.nan, np.nan
            time_mean, time_std = map(float, time_cost.split('±'))
            plot_data.append([mode, model, dist, acc_mean, acc_std, auc_mean, auc_std, time_mean, time_std])

        # 绘制实验结果柱状图
        if plot_data:
            plot_experiment_results_bar(OUT_DIR, plot_data)
        else:
            plot_experiment_results_bar(OUT_DIR)

if __name__ == '__main__':
    main() 