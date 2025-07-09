import matplotlib.pyplot as plt
import numpy as np
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from src.utils.logging_config import get_logger
import pandas as pd

logger=get_logger()


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'out'))
os.makedirs(save_dir, exist_ok=True)

def plot_experiment_results_bar(save_dir, data=None):
    """
    将实验结果以美观学术风格的柱状图形式可视化并保存为图片。
    data: 可选，若传入则使用该数据（list of list），否则自动读取save_dir下experiment_results.csv
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if data is None:
        csv_path = os.path.join(save_dir, 'experiment_results.csv')
        df = pd.read_csv(csv_path)
    else:
        # 只取前4列用于准确率和AUC
        data = [row[:4] for row in data]

    columns = ["模式", "模型", "准确率", "AUC"]
    df = pd.DataFrame(data, columns=columns)

    # 确保'准确率'和'AUC'为数值型，防止绘图警告
    df['准确率'] = pd.to_numeric(df['准确率'], errors='coerce')
    df['AUC'] = pd.to_numeric(df['AUC'], errors='coerce')

    # 按准确率升序排序
    df = df.sort_values(by='准确率', ascending=True).reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    color1 = '#5B9BD5'  # 柔和蓝
    color2 = '#70AD47'  # 柔和绿

    bars1 = ax.bar(x - width/2, df['准确率'], width, label='准确率', color=color1, edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, df['AUC'], width, label='AUC', color=color2, edgecolor='black', linewidth=0.7)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(str(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 1), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(str(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 1), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df['模式'] + '-' + df['模型'], rotation=25, fontsize=10)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_ylim(0.9, 1.02)
    ax.set_title('实验结果总览', fontsize=16, fontweight='bold', pad=15)
    ax.legend(frameon=False, fontsize=11, loc='upper left', bbox_to_anchor=(1.01, 1))

    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    plt.subplots_adjust(left=0.07, right=0.85, top=0.88, bottom=0.18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experiment_results_bar.png'))
    plt.close()


if __name__ == "__main__":
    print("正在生成实验结果柱状图图片……")
    plot_experiment_results_bar(save_dir)
    print("图片已保存到 out/experiment_results_bar.png")





