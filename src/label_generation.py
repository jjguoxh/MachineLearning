# -*- coding: utf-8 -*-
"""
基于最大区间搜索 + 动态 50% 回撤规则生成监督学习标签
- 从每个起点分别延申上涨/下跌两种趋势：只要中途回撤未超过阈值，就持续延长；
  一旦超过阈值，趋势立刻在前一个极值处终止。
- 将所有候选段按振幅绝对值排序，选前 TOP_N 个互不重叠的区间。
- 为每个数据点生成标签：1(上升)、-1(下降)、0(未知/无趋势)
"""

import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import find_peaks

# ========= 可调参数 =========
INPUT_DIR = "../data/"  # 输入CSV文件目录
OUTPUT_DIR = "../label/"  # 输出标签文件目录
TOP_N = 5        # 每天取前 N 个最大振幅段（互不重叠）
RETRACE_FRAC = 0.50  # 动态回撤阈值（例如 0.50 即 50%）
MIN_LEN = 5      # 最小段长度（点数）
MIN_AMP = 0.0    # 最小振幅门槛（可设 >0 过滤噪声）

# ========= 工具函数 =========
def choose_time_axis(df: pd.DataFrame) -> np.ndarray:
    """优先使用 ['time','timestamp','datetime','x']，否则用顺序索引。"""
    for col in ["time", "timestamp", "datetime", "x"]:
        if col in df.columns:
            return df[col].values
    return np.arange(len(df))

def detect_data_format(df: pd.DataFrame):
    """
    检测数据格式：
    1. 原始格式：包含 x, a, b, c, d, index_value 列
    2. 趋势格式：包含 start_value, end_value 等列
    """
    original_format_cols = {"x", "a", "b", "c", "d", "index_value"}
    trend_format_cols = {"start_value", "end_value", "start_idx", "end_idx"}
    
    if original_format_cols.issubset(df.columns):
        return "original"
    elif trend_format_cols.issubset(df.columns):
        return "trend"
    else:
        raise ValueError(f"无法识别数据格式。原始格式需要列：{original_format_cols}，趋势格式需要列：{trend_format_cols}")

def convert_trend_to_original(df):
    """
    将趋势格式数据转换为原始格式数据
    这里我们简单地创建一个最小的原始格式数据框用于演示
    """
    # 对于趋势数据，我们创建一个简化版本的原始数据
    # 实际应用中，您可能需要从其他地方获取原始数据
    print("警告：使用趋势数据生成标签，可能不是最优结果")
    
    # 使用start_idx到end_idx范围内的数据点数来估计数据长度
    max_idx = max(df['end_idx'].max(), df['start_idx'].max()) if not df.empty else 0
    min_idx = min(df['start_idx'].min(), df['end_idx'].min()) if not df.empty else 0
    
    # 创建基本的原始格式数据框
    length = max_idx - min_idx + 1
    result_df = pd.DataFrame({
        'x': range(length),
        'a': np.random.random(length),  # 占位符数据
        'b': np.random.random(length),  # 占位符数据
        'c': np.random.random(length),  # 占位符数据
        'd': np.random.random(length),  # 占位符数据
        'index_value': np.random.random(length)  # 占位符数据
    })
    
    return result_df

def get_price_data(df, data_format):
    """
    从数据框中提取价格数据
    """
    if data_format == "original":
        return (-df["index_value"].values).astype(float)
    elif data_format == "trend":
        # 对于趋势数据，我们需要重建价格序列
        # 这里我们使用一个简化的方法
        if not df.empty:
            max_idx = max(df['end_idx'].max(), df['start_idx'].max())
            min_idx = min(df['start_idx'].min(), df['end_idx'].min())
            length = max_idx - min_idx + 1
            
            # 创建模拟的价格数据
            prices = np.zeros(length)
            # 简单地使用start_value和end_value来构造价格序列
            for _, row in df.iterrows():
                start_idx = int(row['start_idx'])
                end_idx = int(row['end_idx'])
                start_value = row['start_value']
                end_value = row['end_value']
                
                if end_idx >= len(prices):
                    continue
                    
                # 线性插值
                if end_idx > start_idx:
                    prices[start_idx:end_idx+1] = np.linspace(start_value, end_value, end_idx - start_idx + 1)
            
            return (-prices).astype(float)
        else:
            return np.array([])

def extend_up(values: np.ndarray, start: int, retrace_frac: float):
    """
    从 start 开始延申上涨趋势，逐步检查动态回撤：
    drawdown_t <= retrace_frac * profit_t（对所有 t 成立）
    违背时在最后一个最高点结束。
    返回：None 或 dict(i1,i2,amp,dir,max_retrace_ratio)
    """
    if len(values) == 0:
        return None
        
    n = len(values)
    if start >= n - 1:
        return None
        
    s = start
    run_max = values[s]
    last_high_idx = s
    max_ratio_seen = 0.0

    # 没有上行就不构成上涨段
    any_up = False

    for t in range(s + 1, n):
        if values[t] > run_max:
            run_max = values[t]
            last_high_idx = t
            any_up = True

        profit = run_max - values[s]
        if profit <= 0:
            # 还没形成盈利，继续观察
            continue

        drawdown = run_max - values[t]
        ratio = drawdown / (profit + 1e-12)
        max_ratio_seen = max(max_ratio_seen, ratio)

        if ratio > retrace_frac:
            # 触发阈值，段在上一个最高点结束
            if last_high_idx == s:
                return None
            amp = values[last_high_idx] - values[s]
            return {
                "i1": s, "i2": last_high_idx,
                "amp": float(amp), "dir": 1,
                "max_retrace": float(max_ratio_seen)
            }

    # 到末尾也未触发，段在最后一个最高点结束
    if any_up and last_high_idx > s:
        amp = values[last_high_idx] - values[s]
        return {
            "i1": s, "i2": last_high_idx,
            "amp": float(amp), "dir": 1,
            "max_retrace": float(max_ratio_seen)
        }
    return None

def extend_down(values: np.ndarray, start: int, retrace_frac: float):
    """
    从 start 开始延申下跌趋势，逐步检查动态回撤（反弹）：
    drawup_t <= retrace_frac * profit_t，其中 profit_t = values[s] - run_min
    违背时在最后一个最低点结束。
    返回：None 或 dict(i1,i2,amp,dir,max_retrace_ratio)
    """
    if len(values) == 0:
        return None
        
    n = len(values)
    if start >= n - 1:
        return None
        
    s = start
    run_min = values[s]
    last_low_idx = s
    max_ratio_seen = 0.0

    any_down = False

    for t in range(s + 1, n):
        if values[t] < run_min:
            run_min = values[t]
            last_low_idx = t
            any_down = True

        profit = values[s] - run_min
        if profit <= 0:
            continue

        drawup = values[t] - run_min
        ratio = drawup / (profit + 1e-12)
        max_ratio_seen = max(max_ratio_seen, ratio)

        if ratio > retrace_frac:
            if last_low_idx == s:
                return None
            amp = values[s] - values[last_low_idx]
            return {
                "i1": s, "i2": last_low_idx,
                "amp": float(amp), "dir": -1,
                "max_retrace": float(max_ratio_seen)
            }

    if any_down and last_low_idx > s:
        amp = values[s] - values[last_low_idx]
        return {
            "i1": s, "i2": last_low_idx,
            "amp": float(amp), "dir": -1,
            "max_retrace": float(max_ratio_seen)
        }
    return None

def build_candidates(values: np.ndarray, retrace_frac: float):
    """从每个起点生成上涨/下跌两个候选段（若存在）。"""
    if len(values) == 0:
        return []
        
    n = len(values)
    cands = []
    for s in range(n - 1):
        up = extend_up(values, s, retrace_frac)
        if up is not None:
            cands.append(up)
        down = extend_down(values, s, retrace_frac)
        if down is not None:
            cands.append(down)
    return cands

def select_non_overlapping_topN(cands, top_n: int, min_len: int, min_amp: float):
    """
    先按振幅绝对值降序，再筛互不重叠，且满足最小长度/振幅。
    """
    cands_sorted = sorted(cands, key=lambda d: abs(d["amp"]), reverse=True)
    selected = []
    used_ranges = []

    def overlap(a1, a2, b1, b2):
        return not (a2 < b1 or b2 < a1)

    for seg in cands_sorted:
        if len(selected) >= top_n:
            break
        i1, i2 = seg["i1"], seg["i2"]
        if (i2 - i1 + 1) < min_len:
            continue
        if abs(seg["amp"]) < min_amp:
            continue
        if any(overlap(i1, i2, u1, u2) for (u1, u2) in used_ranges):
            continue
        selected.append(seg)
        used_ranges.append((i1, i2))
    return selected

def find_trend_segments(true_prices, min_amplitude=20, retrace_frac=0.5):
    """
    根据新的规则查找趋势段：
    1. 从每个点开始寻找趋势（上涨或下跌）
    2. 累计盈利点数
    3. 如果回调不超过盈利的50%，且继续原方向，则继续扩展
    4. 直到不满足条件为止，在最大盈利点平仓
    """
    n = len(true_prices)
    if n < 2:
        return []
    
    segments = []
    
    # 从每个点开始尝试寻找趋势
    for start_idx in range(n - 1):
        # 尝试寻找上涨趋势
        up_segment = find_up_trend(true_prices, start_idx, min_amplitude, retrace_frac)
        if up_segment:
            segments.append(up_segment)
            
        # 尝试寻找下跌趋势
        down_segment = find_down_trend(true_prices, start_idx, min_amplitude, retrace_frac)
        if down_segment:
            segments.append(down_segment)
            
    return segments

def find_up_trend(prices, start_idx, min_amplitude, retrace_frac):
    """
    从start_idx开始寻找上涨趋势
    """
    n = len(prices)
    if start_idx >= n - 1:
        return None
        
    # 起始价格
    start_price = prices[start_idx]
    
    # 跟踪最高点和最大回撤
    highest_price = start_price
    highest_idx = start_idx
    max_profit = 0  # 最大盈利点数
    end_idx = start_idx
    
    # 遍历后续数据点
    for i in range(start_idx + 1, n):
        current_price = prices[i]
        
        # 更新最高点
        if current_price > highest_price:
            highest_price = current_price
            highest_idx = i
            max_profit = highest_price - start_price
            
        # 计算当前回撤
        current_drawdown = highest_price - current_price
        
        # 如果最大盈利为0，继续寻找
        if max_profit <= 0:
            continue
            
        # 计算回撤比例
        retrace_ratio = current_drawdown / max_profit if max_profit > 0 else 0
        
        # 如果回撤超过阈值，趋势结束
        if retrace_ratio > retrace_frac:
            break
            
        # 更新结束点
        end_idx = i
    
    # 检查是否满足最小振幅要求
    final_amplitude = highest_price - start_price
    if final_amplitude >= min_amplitude and highest_idx > start_idx:
        return {
            "i1": start_idx,
            "i2": highest_idx,  # 在最高点平仓
            "amp": float(final_amplitude),
            "dir": 1,  # 上涨
            "max_retrace": 0.0  # 可以根据需要计算实际最大回撤
        }
        
    return None

def find_down_trend(prices, start_idx, min_amplitude, retrace_frac):
    """
    从start_idx开始寻找下跌趋势
    """
    n = len(prices)
    if start_idx >= n - 1:
        return None
        
    # 起始价格
    start_price = prices[start_idx]
    
    # 跟踪最低点和最大盈利
    lowest_price = start_price
    lowest_idx = start_idx
    max_profit = 0  # 最大盈利点数
    end_idx = start_idx
    
    # 遍历后续数据点
    for i in range(start_idx + 1, n):
        current_price = prices[i]
        
        # 更新最低点
        if current_price < lowest_price:
            lowest_price = current_price
            lowest_idx = i
            max_profit = start_price - lowest_price
            
        # 计算当前反弹
        current_rebound = current_price - lowest_price
        
        # 如果最大盈利为0，继续寻找
        if max_profit <= 0:
            continue
            
        # 计算反弹比例
        rebound_ratio = current_rebound / max_profit if max_profit > 0 else 0
        
        # 如果反弹超过阈值，趋势结束
        if rebound_ratio > retrace_frac:
            break
            
        # 更新结束点
        end_idx = i
    
    # 检查是否满足最小振幅要求
    final_amplitude = start_price - lowest_price
    if final_amplitude >= min_amplitude and lowest_idx > start_idx:
        return {
            "i1": start_idx,
            "i2": lowest_idx,  # 在最低点平仓
            "amp": float(final_amplitude),
            "dir": -1,  # 下跌
            "max_retrace": 0.0  # 可以根据需要计算实际最大反弹
        }
        
    return None

def merge_trend_segments(segments):
    """
    合并相邻的同向趋势段
    """
    if not segments:
        return []
        
    # 按起始点排序
    sorted_segments = sorted(segments, key=lambda x: x["i1"])
    merged_segments = []
    
    i = 0
    while i < len(sorted_segments):
        current_segment = sorted_segments[i]
        
        # 查看是否可以与后续同向段合并
        j = i + 1
        while j < len(sorted_segments):
            next_segment = sorted_segments[j]
            
            # 检查是否同向且相邻
            if (current_segment["dir"] == next_segment["dir"] and 
                next_segment["i1"] <= current_segment["i2"] + 5):  # 允许一定间隔
                
                # 合并段
                if current_segment["dir"] == 1:  # 上涨
                    # 检查合并后是否仍然有效
                    new_start = current_segment["i1"]
                    new_end = next_segment["i2"]
                    i += 1
                    continue
                else:  # 下跌
                    # 检查合并后是否仍然有效
                    new_start = current_segment["i1"]
                    new_end = next_segment["i2"]
                    i += 1
                    continue
            else:
                break
                
        merged_segments.append(current_segment)
        i += 1
        
    return merged_segments

def generate_labels_for_file(csv_file_path, output_dir):
    """为单个CSV文件生成标签 - 动作标签版本"""
    # 读取数据
    df = pd.read_csv(csv_file_path)
    
    # 检查是否有true_index_value列
    if 'true_index_value' not in df.columns:
        print(f"[Warning] 文件 {csv_file_path} 中没有 'true_index_value' 列")
        return
    
    # 使用true_index_value进行趋势检测
    true_prices = df['true_index_value'].values
    
    # 查找趋势段
    segments = find_trend_segments(true_prices, min_amplitude=20, retrace_frac=0.5)
    
    # 合并相邻的同向趋势段
    # merged_segments = merge_trend_segments(segments)
    merged_segments = segments  # 暂时不合并
    
    # 选择互不重叠的区间，按振幅排序取前N个
    selected = select_non_overlapping_topN(merged_segments, TOP_N, MIN_LEN, MIN_AMP)
    
    print(f"[Info] 文件 {csv_file_path} - 候选段数：{len(merged_segments)}；选中段数：{len(selected)}")
    
    # 初始化标签列，全部设为0（无操作）
    # 标签定义：0-无操作, 1-做多开仓, 2-做多平仓, 3-做空开仓, 4-做空平仓
    labels = np.zeros(len(df), dtype=int)
    
    # 只标记动作点（开仓和平仓点），不标记持仓期间的点
    action_points = set()  # 记录已标记的动作点
    
    for seg in selected:
        i1, i2 = seg["i1"], seg["i2"]
        # 确保索引在有效范围内
        i1 = max(0, min(i1, len(labels) - 1))
        i2 = max(0, min(i2, len(labels) - 1))
        
        # 避免动作点重叠
        if i1 not in action_points and i2 not in action_points:
            if seg["dir"] == 1:  # 上涨趋势
                # 趋势开始点：做多开仓
                labels[i1] = 1
                action_points.add(i1)
                # 趋势结束点：做多平仓
                labels[i2] = 2
                action_points.add(i2)
            elif seg["dir"] == -1:  # 下跌趋势
                # 趋势开始点：做空开仓
                labels[i1] = 3
                action_points.add(i1)
                # 趋势结束点：做空平仓
                labels[i2] = 4
                action_points.add(i2)
    
    # 创建结果DataFrame
    result_df = df.copy()
    result_df["label"] = labels  # 动作标签
    
    # 保存结果
    filename = os.path.basename(csv_file_path)
    output_path = os.path.join(output_dir, filename)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[Info] 已保存标签文件: {output_path}")
    print(f"[Info] 标签分布: {Counter(labels)}")
    
    # 打印详细统计信息
    action_labels = {
        0: "无操作",
        1: "做多开仓",
        2: "做多平仓", 
        3: "做空开仓",
        4: "做空平仓"
    }
    
    for label_val, label_name in action_labels.items():
        count = np.sum(labels == label_val)
        print(f"  {label_name}({label_val}): {count} 个")
    
    # 使用matplotlib显示index_value曲线和标签结果
    visualize_labels(df, labels, filename)
    
def visualize_labels(df, labels, filename):
    """
    可视化index_value曲线和标签结果
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制index_value曲线
    ax.plot(df['index_value'], label='Index Value', color='blue', linewidth=1)
    
    # 提取各类标签点
    long_entry_points = []   # 做多开仓点 (label=1)
    long_exit_points = []    # 做多平仓点 (label=2)
    short_entry_points = []  # 做空开仓点 (label=3)
    short_exit_points = []   # 做空平仓点 (label=4)
    
    for i, label in enumerate(labels):
        if label == 1:  # 做多开仓
            long_entry_points.append((i, df['index_value'].iloc[i]))
        elif label == 2:  # 做多平仓
            long_exit_points.append((i, df['index_value'].iloc[i]))
        elif label == 3:  # 做空开仓
            short_entry_points.append((i, df['index_value'].iloc[i]))
        elif label == 4:  # 做空平仓
            short_exit_points.append((i, df['index_value'].iloc[i]))
    
    # 绘制交易信号箭头
    if long_entry_points:
        x, y = zip(*long_entry_points)
        ax.scatter(x, y, color='red', marker='^', s=100, label='Long Entry', zorder=5)
    
    if long_exit_points:
        x, y = zip(*long_exit_points)
        ax.scatter(x, y, color='red', marker='v', s=100, label='Long Exit', zorder=5)
    
    if short_entry_points:
        x, y = zip(*short_entry_points)
        ax.scatter(x, y, color='green', marker='^', s=100, label='Short Entry', zorder=5)
    
    if short_exit_points:
        x, y = zip(*short_exit_points)
        ax.scatter(x, y, color='green', marker='v', s=100, label='Short Exit', zorder=5)
    
    # 添加图例
    legend_elements = [
        Patch(facecolor='blue', label='Index Value'),
        Patch(facecolor='red', label='Long Entry (^) / Exit (v)'),
        Patch(facecolor='green', label='Short Entry (^) / Exit (v)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # 设置标题和标签
    ax.set_title(f'Index Value with Trading Signals - {filename}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Index Value')
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表
    plt.show()

def process_all_files():
    """处理所有CSV文件"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个文件
    for csv_file in csv_files:
        try:
            csv_file_path = os.path.join(INPUT_DIR, csv_file)
            generate_labels_for_file(csv_file_path, OUTPUT_DIR)
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

# ========= 主流程 =========
if __name__ == "__main__":
    process_all_files()
    print("标签生成完成！")