import os
import shutil
import subprocess
import yaml
import time
import logging
import glob
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.join(PROJECT_ROOT, 'src', 'default.yaml')
MAIN_PY = os.path.join(PROJECT_ROOT, 'src', 'main.py')
OUT_DIR = os.path.join(PROJECT_ROOT, 'out')
import sys
sys.path.append(PROJECT_ROOT)
from src.utils.draw import plot_experiment_results_bar
# 所有模型类型
MODEL_TYPES = ['CNN', 'MLP', 'KNN', 'RF', 'SVC', 'LR']
# 支持的实验模式
MODES = ['Centralized', 'Federated']
# 传统ML模型
ML_MODELS = ['KNN', 'RF', 'SVC', 'LR']

batch_log = os.path.join(OUT_DIR, 'batch.log')
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=batch_log,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8'
)

def run_experiment(mode, model_type):
    # 1. 读取配置
    with open(DEFAULT_YAML, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 2. 修改配置
    config['mode'] = mode
    config['model']['type'] = model_type
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
            new_log_name = f'{mode}_{model_type}.log'
            new_log_path = os.path.join(OUT_DIR, new_log_name)
            shutil.move(latest_log, new_log_path)
            logging.info(f'实验 {mode}_{model_type} 日志文件: {new_log_path}')
    # 清理 logs 目录
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)

def parse_log_file(log_path):
    mode = model = acc = auc = time_cost = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '实验模式:' in line:
                mode_match = re.search(r'实验模式:\s*(\w+)', line)
                if mode_match:
                    mode = mode_match.group(1)
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
    return mode, model, acc, auc, time_cost

def main():
    # 备份 default.yaml
    backup_yaml = DEFAULT_YAML + '.bak'
    shutil.copy(DEFAULT_YAML, backup_yaml)
    results = []  # 用于存储所有实验结果
    try:
        for mode in MODES:
            for model_type in MODEL_TYPES:
                # 传统ML模型不支持 Federated
                if mode == 'Federated' and model_type in ML_MODELS:
                    continue
                logging.info(f'运行实验: mode={mode}, model={model_type}')
                run_experiment(mode, model_type)
                time.sleep(1)  # 防止IO冲突
                # 解析刚刚复制的日志
                log_name = f'{mode}_{model_type}.log'
                log_path = os.path.join(OUT_DIR, log_name)
                if os.path.exists(log_path):
                    parsed = parse_log_file(log_path)
                    if all(parsed):
                        results.append(parsed)
    finally:
        # 恢复 default.yaml
        shutil.move(backup_yaml, DEFAULT_YAML)
        logging.info('所有实验完成，default.yaml 已恢复')
        # 写入实验结果CSV
        import csv
        result_csv = os.path.join(OUT_DIR, 'experiment_results.csv')
        with open(result_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['模式', '模型', '准确率', 'AUC', '训练总耗时（秒）'])
            for mode, model, acc, auc, time_cost in results:
                writer.writerow([mode, model, acc, auc, time_cost])
        # 绘制实验结果柱状图，优先用内存中的results
        if results:
            plot_experiment_results_bar(OUT_DIR, results)
        else:
            plot_experiment_results_bar(OUT_DIR)

if __name__ == '__main__':
    main() 