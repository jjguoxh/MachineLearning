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

# ========= 可调参数 =========
INPUT_DIR = "E:/SupervisedLearning/data/"  # 输入CSV文件目录
OUTPUT_DIR = "E:/SupervisedLearning/label/"  # 输出标签文件目录
TOP_N = 3        # 每天取前 N 个最大振幅段（互不重叠）
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

def generate_labels_for_file(csv_file_path, output_dir):
    """为单个CSV文件生成标签 - 动作标签版本"""
    # 读取数据
    df = pd.read_csv(csv_file_path)
    
    # 检测数据格式和获取价格数据（保持原有逻辑）
    try:
        data_format = detect_data_format(df)
        print(f"[Info] 检测到数据格式: {data_format}")
    except ValueError as e:
        print(f"[Warning] {e}")
        if "index_value" in df.columns:
            data_format = "original"
        else:
            print("[Error] 无法处理该文件格式")
            return
    
    if data_format == "trend":
        print("[Info] 检测到趋势格式数据，转换为原始格式...")
        original_df = convert_trend_to_original(df)
    else:
        original_df = df.copy()
    
    prices = get_price_data(df, data_format)
    
    if len(prices) == 0:
        print(f"[Warning] 无法从文件 {csv_file_path} 提取价格数据")
        return
    
    # 基于动态 50% 回撤规则生成候选段
    candidates = build_candidates(prices, RETRACE_FRAC)
    
    # 选取前 TOP_N 个互不重叠的区间
    selected = select_non_overlapping_topN(candidates, TOP_N, MIN_LEN, MIN_AMP)
    
    print(f"[Info] 文件 {csv_file_path} - 候选段数：{len(candidates)}；选中段数：{len(selected)}")
    
    # 初始化标签列，全部设为0（无操作）
    # 标签定义：0-无操作, 1-做多开仓, 2-做多平仓, 3-做空开仓, 4-做空平仓
    labels = np.zeros(len(original_df), dtype=int)
    
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
    if data_format == "original":
        result_df = df[["x", "a", "b", "c", "d", "index_value"]].copy()
    else:
        length = len(labels)
        result_df = pd.DataFrame({
            'x': range(length),
            'a': np.random.random(length),
            'b': np.random.random(length),
            'c': np.random.random(length),
            'd': np.random.random(length),
            'index_value': np.random.random(length)
        })
    
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