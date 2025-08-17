# -*- coding: utf-8 -*-
"""
确定性开仓点识别 + 止盈策略
- 识别当日最确定的开仓点（买入后立即顺着交易方向运行或回撤不超过开仓点）
- 在至少盈利10个点后考虑止盈
- 保存交易点到文件并绘制图表
"""
# 用于标记日内最确定的开仓点和止盈点，用于生成标签给监督学习

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ========= 可调参数 =========
CSV_FILE = "../data/250103.csv"
MIN_PROFIT = 10.0         # 最小止盈点数
MAX_DRAWBACK = 0.0        # 最大回撤容忍度（0表示不能回撤）
MIN_DISTANCE = 10         # 开仓点之间的最小距离
PLOT_FILL_ALPHA = 0.2     # 阴影透明度
SAVE_FIG = True

# ========= 工具函数 =========
def choose_time_axis(df: pd.DataFrame) -> np.ndarray:
    """优先使用 ['time','timestamp','datetime','x']，否则用顺序索引。"""
    for col in ["time", "timestamp", "datetime", "x"]:
        if col in df.columns:
            return df[col].values
    return np.arange(len(df))

def find_long_entries(values: np.ndarray, min_distance: int):
    """
    寻找做多开仓点
    确定性开仓点定义：买入后价格立即上涨，或者即使有回撤也不会低于开仓价
    """
    n = len(values)
    entries = []
    
    i = 1
    while i < n - 1:
        # 寻找局部低点作为潜在开仓点
        is_low_point = (values[i] <= values[i-1]) and (values[i] < values[i+1])
        
        if is_low_point:
            entry_idx = i
            entry_price = values[i]
            
            # 检查后续走势是否符合确定性开仓条件
            # 条件1：买入后价格不能低于开仓价（回撤为0）
            # 条件2：至少有一次价格高于开仓价（上涨）
            valid_entry = False
            has_upward_movement = False
            
            j = i + 1
            while j < n:
                # 如果价格低于开仓价，且超过容忍度，则不是有效开仓点
                if values[j] < entry_price - MAX_DRAWBACK:
                    break
                
                # 如果价格高于开仓价，说明有上涨
                if values[j] > entry_price:
                    has_upward_movement = True
                
                # 如果价格上涨且达到最小盈利要求，则标记为有效开仓点
                if has_upward_movement and values[j] >= entry_price + MIN_PROFIT:
                    valid_entry = True
                    break
                    
                j += 1
            
            if valid_entry:
                entries.append({
                    "idx": entry_idx,
                    "price": entry_price,
                    "exit_idx": j,
                    "exit_price": values[j],
                    "profit": values[j] - entry_price
                })
                # 跳过最小距离，避免相邻信号
                i = j + min_distance
            else:
                i += 1
        else:
            i += 1
    
    return entries

def find_short_entries(values: np.ndarray, min_distance: int):
    """
    寻找做空开仓点
    确定性开仓点定义：卖出后价格立即下跌，或者即使有反弹也不会高于开仓价
    """
    n = len(values)
    entries = []
    
    i = 1
    while i < n - 1:
        # 寻找局部高点作为潜在开仓点
        is_high_point = (values[i] >= values[i-1]) and (values[i] > values[i+1])
        
        if is_high_point:
            entry_idx = i
            entry_price = values[i]
            
            # 检查后续走势是否符合确定性开仓条件
            # 条件1：卖出后价格不能高于开仓价（反弹为0）
            # 条件2：至少有一次价格低于开仓价（下跌）
            valid_entry = False
            has_downward_movement = False
            
            j = i + 1
            while j < n:
                # 如果价格高于开仓价，且超过容忍度，则不是有效开仓点
                if values[j] > entry_price + MAX_DRAWBACK:
                    break
                
                # 如果价格低于开仓价，说明有下跌
                if values[j] < entry_price:
                    has_downward_movement = True
                
                # 如果价格下跌且达到最小盈利要求，则标记为有效开仓点
                if has_downward_movement and values[j] <= entry_price - MIN_PROFIT:
                    valid_entry = True
                    break
                    
                j += 1
            
            if valid_entry:
                entries.append({
                    "idx": entry_idx,
                    "price": entry_price,
                    "exit_idx": j,
                    "exit_price": values[j],
                    "profit": entry_price - values[j]  # 做空盈利为正
                })
                # 跳过最小距离，避免相邻信号
                i = j + min_distance
            else:
                i += 1
        else:
            i += 1
    
    return entries

def filter_and_rank_entries(long_entries, short_entries, max_positions=10):
    """
    过滤并排序开仓点，按盈利大小排序，限制最大持仓数
    """
    # 合并所有开仓点
    all_entries = []
    
    # 标记做多开仓点
    for entry in long_entries:
        entry["direction"] = "long"
        all_entries.append(entry)
    
    # 标记做空开仓点
    for entry in short_entries:
        entry["direction"] = "short"
        all_entries.append(entry)
    
    # 按盈利从高到低排序
    all_entries.sort(key=lambda x: x["profit"], reverse=True)
    
    # 限制最大持仓数
    selected_entries = all_entries[:max_positions]
    
    # 按时间顺序排序
    selected_entries.sort(key=lambda x: x["idx"])
    
    return selected_entries

# ========= 主流程 =========
def main():
    df = pd.read_csv(CSV_FILE)

    # 必要列检查
    required_cols = {"index_value", "a", "b", "c", "d"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列：{required_cols}")

    times = choose_time_axis(df)
    prices = df["index_value"].values.astype(float)

    # 1) 寻找做多和做空的确定性开仓点
    long_entries = find_long_entries(prices, MIN_DISTANCE)
    short_entries = find_short_entries(prices, MIN_DISTANCE)
    
    print(f"[Info] 做多开仓点数：{len(long_entries)}")
    print(f"[Info] 做空开仓点数：{len(short_entries)}")
    
    # 2) 过滤并排序开仓点
    selected_entries = filter_and_rank_entries(long_entries, short_entries)
    
    print(f"[Info] 选中交易点数：{len(selected_entries)}")
    for k, entry in enumerate(selected_entries, 1):
        direction = "做多" if entry["direction"] == "long" else "做空"
        print(f"  #{k}: [{entry['idx']}] {direction} 开仓价={entry['price']:.2f}, "
              f"止盈点=[{entry['exit_idx']}] 止盈价={entry['exit_price']:.2f}, "
              f"盈利={entry['profit']:.2f}")

    # 3) 输出 CSV（交易点信息）
    rows = []
    for entry in selected_entries:
        i1, i2 = entry["idx"], entry["exit_idx"]
        row = {
            "entry_idx": i1,
            "exit_idx": i2,
            "direction": entry["direction"],
            "entry_time": times[i1],
            "exit_time": times[i2],
            "entry_price": entry["price"],
            "exit_price": entry["exit_price"],
            "profit": entry["profit"],
            "length": int(i2 - i1 + 1),
        }
        for col in ["a", "b", "c", "d"]:
            vals = df[col].values[i1:i2+1].astype(float)
            row[f"{col}_entry"] = float(df[col].values[i1])
            row[f"{col}_exit"] = float(df[col].values[i2])
            row[f"{col}_mean"] = float(np.mean(vals))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv("certain_trading_points.csv", index=False, encoding="utf-8-sig")
    print("已保存 certain_trading_points.csv")

    # 4) 绘图：价格曲线 + 开仓/止盈点
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # 主图：价格曲线
    ax1.plot(times, prices, color="black", linewidth=1.2, label="index_value")

    # 标记做多交易点
    long_entries_selected = [e for e in selected_entries if e["direction"] == "long"]
    for entry in long_entries_selected:
        entry_idx, exit_idx = entry["idx"], entry["exit_idx"]
        entry_price, exit_price = entry["price"], entry["exit_price"]
        
        # 开仓点（绿色三角）
        ax1.scatter(times[entry_idx], entry_price, color="green", marker="^", s=100, zorder=5)
        # 止盈点（红色三角）
        ax1.scatter(times[exit_idx], exit_price, color="red", marker="v", s=100, zorder=5)
        # 连线
        ax1.plot([times[entry_idx], times[exit_idx]], [entry_price, exit_price], 
                color="green", linewidth=1.5, alpha=0.7)
        
        # 盈利区域阴影
        ax1.add_patch(Rectangle((times[entry_idx], entry_price),
                                times[exit_idx] - times[entry_idx],
                                exit_price - entry_price,
                                color="green", alpha=PLOT_FILL_ALPHA, zorder=1))

    # 标记做空交易点
    short_entries_selected = [e for e in selected_entries if e["direction"] == "short"]
    for entry in short_entries_selected:
        entry_idx, exit_idx = entry["idx"], entry["exit_idx"]
        entry_price, exit_price = entry["price"], entry["exit_price"]
        
        # 开仓点（红色三角）
        ax1.scatter(times[entry_idx], entry_price, color="red", marker="v", s=100, zorder=5)
        # 止盈点（绿色三角）
        ax1.scatter(times[exit_idx], exit_price, color="green", marker="^", s=100, zorder=5)
        # 连线
        ax1.plot([times[entry_idx], times[exit_idx]], [entry_price, exit_price], 
                color="red", linewidth=1.5, alpha=0.7)
        
        # 盈利区域阴影
        ax1.add_patch(Rectangle((times[entry_idx], exit_price),
                                times[exit_idx] - times[entry_idx],
                                entry_price - exit_price,
                                color="red", alpha=PLOT_FILL_ALPHA, zorder=1))

    ax1.set_ylabel("index_value")
    ax1.set_title(f"确定性开仓点识别 (最小止盈点数: {MIN_PROFIT})")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 副图：因子 a,b,c,d
    colors = {"a": "tab:red", "b": "tab:blue", "c": "tab:green", "d": "tab:orange"}
    for col in ["a", "b", "c", "d"]:
        ax2.plot(times, df[col].values, label=col, color=colors[col])
    ax2.set_ylabel("factors")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig("certain_trading_points.png", dpi=150)
        print("已保存 certain_trading_points.png")
    plt.show()

if __name__ == "__main__":
    main()