"""
绘制收敛过程图：带有阴影误差带的折线图
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import defaultdict


def parse_log_file(log_path: str) -> list[tuple[int, float]]:
    """
    解析日志文件，提取每轮的准确率
    返回: [(round_num, accuracy), ...]
    """
    results = []
    
    # 匹配 Centralized 日志格式
    centralized_pattern = re.compile(
        r'\[Centralized\]\[Round (\d+)\] 准确率: ([\d.]+)'
    )
    # 匹配 Federated 日志格式
    federated_pattern = re.compile(
        r'\[Federated\]\[Round (\d+)\] 全局模型准确率: ([\d.]+)'
    )
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 尝试匹配 Centralized
            match = centralized_pattern.search(line)
            if match:
                round_num = int(match.group(1))
                accuracy = float(match.group(2))
                results.append((round_num, accuracy))
                continue
            
            # 尝试匹配 Federated
            match = federated_pattern.search(line)
            if match:
                round_num = int(match.group(1))
                accuracy = float(match.group(2))
                results.append((round_num, accuracy))
    
    return results


def load_experiment_data(out_dir: str) -> dict:
    """
    加载所有实验数据
    返回结构: {
        'mnist': {
            'Centralized': [[run0_data], [run1_data], ...],
            'Federated_IID': [...],
            'Federated_NonIID': [...]
        },
        'cifar10': {...}
    }
    """
    data = {
        'mnist': {
            'Centralized': [],
            'Federated_IID': [],
            'Federated_NonIID': []
        },
        'cifar10': {
            'Centralized': [],
            'Federated_IID': [],
            'Federated_NonIID': []
        }
    }
    
    # Centralized_CNN -> MNIST
    for i in range(5):
        log_path = os.path.join(out_dir, f'Centralized_CNN_run_{i}.log')
        if os.path.exists(log_path):
            data['mnist']['Centralized'].append(parse_log_file(log_path))
    
    # Centralized_ResNet18 -> CIFAR10
    for i in range(5):
        log_path = os.path.join(out_dir, f'Centralized_ResNet18_run_{i}.log')
        if os.path.exists(log_path):
            data['cifar10']['Centralized'].append(parse_log_file(log_path))
    
    # Federated_CNN_iid -> MNIST IID
    for i in range(5):
        log_path = os.path.join(out_dir, f'Federated_CNN_iid_run_{i}.log')
        if os.path.exists(log_path):
            data['mnist']['Federated_IID'].append(parse_log_file(log_path))
    
    # Federated_CNN_noniid_dirichlet -> MNIST NonIID
    for i in range(5):
        log_path = os.path.join(out_dir, f'Federated_CNN_noniid_dirichlet_run_{i}.log')
        if os.path.exists(log_path):
            data['mnist']['Federated_NonIID'].append(parse_log_file(log_path))
    
    # Federated_ResNet18_iid -> CIFAR10 IID
    for i in range(5):
        log_path = os.path.join(out_dir, f'Federated_ResNet18_iid_run_{i}.log')
        if os.path.exists(log_path):
            data['cifar10']['Federated_IID'].append(parse_log_file(log_path))
    
    # Federated_ResNet18_noniid_dirichlet -> CIFAR10 NonIID
    for i in range(5):
        log_path = os.path.join(out_dir, f'Federated_ResNet18_noniid_dirichlet_run_{i}.log')
        if os.path.exists(log_path):
            data['cifar10']['Federated_NonIID'].append(parse_log_file(log_path))
    
    return data


def process_runs_data(runs_data: list[list[tuple[int, float]]], 
                      is_federated: bool = False,
                      epochs_per_round: int = 5,
                      target_length: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    处理多次运行的数据，计算均值和标准差
    
    Args:
        runs_data: 多次运行的数据 [[(round, acc), ...], ...]
        is_federated: 是否是联邦学习（需要乘以epochs_per_round）
        epochs_per_round: 联邦学习每轮的epoch数
        target_length: 目标长度，如果指定则填充到该长度
    
    Returns:
        epochs: epoch数组
        mean_acc: 平均准确率
        std_acc: 标准差
    """
    if not runs_data:
        return np.array([]), np.array([]), np.array([])
    
    # 找到最短的运行长度（以保证所有运行都有数据）
    min_length = min(len(run) for run in runs_data)
    
    if min_length == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 先确定统一的epoch序列
    rounds = [r[0] for r in runs_data[0][:min_length]]
    if is_federated:
        # 联邦学习：round * epochs_per_round = 实际epoch数
        base_epochs = [r * epochs_per_round for r in rounds]
        epoch_step = epochs_per_round
    else:
        # 集中式：round = epoch
        base_epochs = rounds
        epoch_step = 1
    
    # 在开头添加 Epoch=0
    epochs = [0] + base_epochs
    
    # 如果需要填充到目标长度
    if target_length is not None and len(epochs) < target_length:
        last_epoch = epochs[-1]
        while len(epochs) < target_length:
            epochs.append(epochs[-1] + epoch_step)
    
    epochs = np.array(epochs)
    
    # 提取所有运行的准确率数据
    all_accs = []
    for run in runs_data:
        accs = [r[1] for r in run[:min_length]]
        
        # 在开头添加 Accuracy=0.1 的点（转换为百分比）
        accs = [10.0] + [a * 100 for a in accs]
        
        # 如果需要填充到目标长度，用最后一个准确率填充
        if target_length is not None and len(accs) < target_length:
            last_acc = accs[-1]
            while len(accs) < target_length:
                accs.append(last_acc)
        
        all_accs.append(accs)
    
    # 转换为numpy数组
    accs = np.array(all_accs)
    
    mean_acc = np.mean(accs, axis=0)
    std_acc = np.std(accs, axis=0)
    
    return epochs, mean_acc, std_acc


def get_fonts():
    """
    获取可用的字体：宋体风格和Times New Roman风格
    返回: (song_font, times_font)
    """
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 查找宋体风格字体（优先级从高到低）
    song_candidates = [
        'Noto Serif CJK JP',  # Noto CJK 宋体风格
        'Noto Serif CJK SC',
        'Noto Sans CJK JP',
        'Noto Sans CJK SC', 
        'SimSun',
        'STSong',
        'AR PL SungtiL GB',
    ]
    song_font = None
    for font in song_candidates:
        if font in available_fonts:
            song_font = font
            break
    if song_font is None:
        song_font = 'DejaVu Sans'  # 最终回退
    
    # 查找 Times New Roman 风格字体（优先级从高到低）
    times_candidates = [
        'Times New Roman',
        'Nimbus Roman',       # Times New Roman 开源替代
        'PT Serif',           # 高质量衬线字体
        'DejaVu Serif',
        'FreeSerif',
        'Noto Serif',
    ]
    times_font = None
    for font in times_candidates:
        if font in available_fonts:
            times_font = font
            break
    if times_font is None:
        times_font = 'DejaVu Serif'  # 最终回退
    
    return song_font, times_font


def plot_single_subplot(ax, data: dict, dataset: str, subplot_label: str, 
                        song_font: str, times_font: str):
    """
    在给定的axes上绘制单个数据集的收敛图
    
    Args:
        ax: matplotlib axes对象
        data: 实验数据字典
        dataset: 数据集名称 ('mnist' 或 'cifar10')
        subplot_label: 子图标签 ('(a)' 或 '(b)')
        song_font: 宋体字体名称
        times_font: Times New Roman 风格字体名称
    """
    # 黑白印刷样式：线型 + 标记区分
    styles = {
        'Centralized': {
            'linestyle': '-',           # 实线
            'marker': 'o',              # 圆形
            'color': 'black'
        },
        'Federated_IID': {
            'linestyle': '--',          # 虚线
            'marker': 's',              # 方形
            'color': 'black'
        },
        'Federated_NonIID': {
            'linestyle': '-.',          # 点划线
            'marker': '^',              # 三角形
            'color': 'black'
        }
    }

    labels = {
        'Centralized': 'Centralized',
        'Federated_IID': 'Federated (IID)',
        'Federated_NonIID': 'Federated (Non-IID)'
    }
    
    dataset_data = data[dataset]
    
    # 先计算所有方法的原始长度，找到最长的
    max_length = 0
    method_lengths = {}
    for method in ['Centralized', 'Federated_IID', 'Federated_NonIID']:
        runs_data = dataset_data[method]
        if runs_data:
            # 找到最短的运行长度
            min_length = min(len(run) for run in runs_data)
            # +1 是因为我们要添加 Epoch=0 的点
            method_lengths[method] = min_length + 1
            max_length = max(max_length, min_length + 1)
    
    # 绘制每种方法的收敛曲线
    for method in ['Centralized', 'Federated_IID', 'Federated_NonIID']:
        runs_data = dataset_data[method]
        is_federated = method.startswith('Federated')
        
        epochs, mean_acc, std_acc = process_runs_data(
            runs_data, 
            is_federated=is_federated,
            epochs_per_round=5,
            target_length=max_length  # 填充到最长长度
        )
        
        if len(epochs) == 0:
            continue
        
        style = styles[method]
        label = labels[method]

        # 绘制均值线（黑白线型 + 标记）
        ax.plot(epochs, mean_acc,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2,
                label=label,
                marker=style['marker'],
                markersize=5,
                markerfacecolor='none',
                markeredgecolor='black',
                markeredgewidth=1.2,
                markevery=max(1, len(epochs) // 10))

        # 绘制阴影误差带（灰色半透明）
        ax.fill_between(epochs,
                        mean_acc - std_acc,
                        mean_acc + std_acc,
                        color='gray',
                        alpha=0.15)
    
    # 设置标签（不设置标题），使用 Times New Roman 风格字体
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', fontfamily=times_font)
    ax.set_ylabel('准确率/%', fontsize=14, fontweight='bold', fontfamily=song_font)
    
    # 设置图例，使用 Times New Roman 风格字体
    from matplotlib import font_manager as fm
    prop = fm.FontProperties(family=times_font, size=12)
    legend = ax.legend(loc='lower right', framealpha=0.9, prop=prop)
    # 确保图例中的所有文本都使用正确字体
    for text in legend.get_texts():
        text.set_fontfamily(times_font)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # 设置Y轴范围（百分比）
    ax.set_ylim([0, 105])
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # 设置Y轴刻度为百分比格式
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # 设置坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=12)
    # 设置刻度标签字体为 Times New Roman 风格
    for label in ax.get_xticklabels():
        label.set_fontfamily(times_font)
    for label in ax.get_yticklabels():
        label.set_fontfamily(times_font)
    
    # 设置子图标题（底部居中）
    # 格式: (a) MNIST 或 (b) CIFAR-10
    dataset_name = 'MNIST' if dataset == 'mnist' else 'CIFAR-10'
    subtitle = f'{subplot_label} {dataset_name}'
    
    # 在底部添加子图标题，使用 Times New Roman
    times_prop = fm.FontProperties(family=times_font, size=16, weight='bold')
    ax.text(0.5, -0.12, subtitle, 
            transform=ax.transAxes,
            fontproperties=times_prop,
            verticalalignment='top',
            horizontalalignment='center')


def plot_combined_convergence(data: dict, output_path: str):
    """
    绘制合并的收敛图（包含两个子图）
    """
    # 获取可用字体
    song_font, times_font = get_fonts()
    print(f"使用字体 - 宋体: {song_font}, Times New Roman: {times_font}")
    
    # 设置默认字体
    plt.rcParams['font.family'] = times_font
    plt.rcParams['font.serif'] = [times_font]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建包含两个子图的figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 绘制MNIST子图
    plot_single_subplot(ax1, data, 'mnist', '(a)', song_font, times_font)
    
    # 绘制CIFAR-10子图
    plot_single_subplot(ax2, data, 'cifar10', '(b)', song_font, times_font)
    
    # 调整布局，增加底部空间以容纳子标题
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"合并图表已保存: {output_path}")


def main():
    # 获取脚本所在目录的上两级作为项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    out_dir = os.path.join(project_root, 'out')
    
    print(f"读取日志目录: {out_dir}")
    
    # 加载所有实验数据
    data = load_experiment_data(out_dir)
    
    # 绘制合并的收敛图
    combined_output = os.path.join(out_dir, 'convergence_combined.png')
    plot_combined_convergence(data, combined_output)
    
    print("图表生成完成！")


if __name__ == '__main__':
    main()

