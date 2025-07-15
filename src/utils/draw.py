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
        columns = ['模式', '模型', '数据分布', '准确率', 'AUC', '训练总耗时（秒）']
        df = pd.DataFrame(data, columns=columns)

    # 确保'准确率'和'AUC'为数值型，对于缺失值不进行删除，而是保留为NaN
    df['准确率'] = pd.to_numeric(df['准确率'], errors='coerce')
    df['AUC'] = pd.to_numeric(df['AUC'], errors='coerce')
    # 只删除准确率也缺失的行
    df.dropna(subset=['准确率'], inplace=True)

    # 创建更美观的标签
    def create_label(row):
        if row['模式'] == 'Federated' and pd.notna(row['数据分布']):
            # 缩短标签
            dist_map = {
                'noniid_label_skew': 'label',
                'noniid_quantity_skew': 'quantity',
                'iid': 'iid'
            }
            dist = dist_map.get(row['数据分布'], row['数据分布'])
            return f"FL {row['模型']}\n({dist})"
        else:
            return f"CL\n{row['模型']}"
    df['标签'] = df.apply(create_label, axis=1)

    # 按准确率升序排序
    df = df.sort_values(by='准确率', ascending=True).reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(20, 9), dpi=300)
    color1 = '#5B9BD5'
    color2 = '#70AD47'

    # 绘制准确率柱子
    bars1 = ax.bar(x - width/2, df['准确率'], width, label='准确率', color=color1, edgecolor='black', linewidth=0.7)

    # 仅为有AUC值的行绘制柱子
    auc_df = df.dropna(subset=['AUC'])
    auc_indices = auc_df.index.to_numpy()
    bars2 = ax.bar(x[auc_indices] + width/2, auc_df['AUC'], width, label='AUC', color=color2, edgecolor='black', linewidth=0.7)


    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height):
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(df['标签'], rotation=0, ha="center", fontsize=10)
    ax.set_ylabel('分数', fontsize=14)
    ax.set_ylim(0.9, 1.01)
    ax.set_title('实验结果总览', fontsize=18, fontweight='bold', pad=20)
    ax.legend(frameon=False, fontsize=12, loc='upper left')

    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(save_dir, 'experiment_results_bar.png'))
    plt.close()


if __name__ == "__main__":
    print("正在生成实验结果柱状图图片……")
    plot_experiment_results_bar(save_dir)
    print("图片已保存到 out/experiment_results_bar.png")





