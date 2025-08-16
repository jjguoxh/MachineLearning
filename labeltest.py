import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def find_top_peak_valley_pairs(df, top_k=2):
    """
    从 index_value 中找出峰谷对，并筛选出振幅差值最大的前 top_k 对
    """
    values = df['index_value'].values
    x_values = df['x'].values

    # 找峰和谷
    peaks, _ = find_peaks(values)
    valleys, _ = find_peaks(-values)

    # 所有峰谷组合
    pairs = []
    for p in peaks:
        for v in valleys:
            diff = abs(values[p] - values[v])
            pairs.append((diff, p, v))

    # 按振幅差值降序
    pairs.sort(key=lambda x: x[0], reverse=True)

    # 取前 top_k
    selected_pairs = pairs[:top_k]

    print(f"选中的前{top_k}组峰谷差：")
    for diff, p, v in selected_pairs:
        print(f"峰({x_values[p]}, {values[p]:.4f}), 谷({x_values[v]}, {values[v]:.4f}), 差值={diff:.4f}")

    return selected_pairs, peaks, valleys

def plot_selected_peak_valley(df, selected_pairs, save_path=None):
    """
    绘制数据曲线，并用三角形标记选中的峰谷，用紫色显示峰谷对之间的曲线段
    """
    plt.figure(figsize=(15, 6))
    # 绘制完整的黑色曲线
    plt.plot(df['x'], df['index_value'], label='Index Value', color='black')

    # 为每一对峰谷用紫色显示它们之间的曲线段
    for diff, p, v in selected_pairs:
        # 确定起点和终点（确保起点索引小于终点索引）
        start_idx = min(p, v)
        end_idx = max(p, v)
        
        # 用紫色显示峰谷对之间的曲线段
        plt.plot(df['x'][start_idx:end_idx+1], df['index_value'][start_idx:end_idx+1], 
                 color='purple', linewidth=2, zorder=4)

    # 标记峰和谷
    for diff, p, v in selected_pairs:
        # 蓝色 ▲ 峰
        plt.scatter(df['x'][p], df['index_value'][p], color='blue', marker='^', s=120, zorder=5)
        # 红色 ▼ 谷
        plt.scatter(df['x'][v], df['index_value'][v], color='red', marker='v', s=120, zorder=5)

    plt.xlabel('Time / Index')
    plt.ylabel('Index Value')
    plt.title('Top Peak-Valley Pairs by Amplitude Difference')
    plt.grid(True)
    plt.legend(['Index Value', 'Selected Peak-Valley Segments', 'Peak', 'Valley'])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存: {save_path}")
    else:
        plt.show()

def process_csv_file(file_path, top_k=2, output_dir="output_plots"):
    """
    主流程：读 CSV → 找峰谷 → 画图显示
    """
    # 读取 CSV
    df = pd.read_csv(file_path)

    # 检查必要列
    if not {'x', 'index_value'}.issubset(df.columns):
        raise ValueError("CSV 文件必须包含 'x' 和 'index_value' 两列")

    # 找峰谷对
    selected_pairs, peaks, valleys = find_top_peak_valley_pairs(df, top_k=top_k)

    # 直接显示图像，不保存
    plot_selected_peak_valley(df, selected_pairs, save_path=None)

if __name__ == "__main__":
    # 示例：处理单个 CSV
    csv_path = "240112.csv"  # 你的 CSV 文件路径
    process_csv_file(csv_path, top_k=2)