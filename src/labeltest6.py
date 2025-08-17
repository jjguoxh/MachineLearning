# -*- coding: utf-8 -*-
"""
最大区间搜索 + 动态 50% 回撤规则（逐步生效）
- 从每个起点分别延申上涨/下跌两种趋势：只要中途回撤未超过阈值，就持续延长；
  一旦超过阈值，趋势立刻在前一个极值处终止。
- 将所有候选段按综合评分排序，选前 TOP_N 个互不重叠的区间。
- 保存 top_trends.csv；绘制 index_value + a,b,c,d（不同颜色）并用红/蓝标记趋势。
"""
# 一个比较满意的实现，用于标记日内最大的涨跌幅趋势段，用于生成标签给监督学习

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ========= 可调参数 =========
CSV_FILE = "../data/250103.csv"
TOP_N = 4                 # 取前 N 个最大振幅段（互不重叠）
RETRACE_FRAC = 0.50       # 动态回撤阈值（例如 0.50 即 50%）
MIN_LEN = 5               # 最小段长度（点数）
MIN_AMP = 0.0             # 最小振幅门槛（可设 >0 过滤噪声）
PLOT_FILL_ALPHA = 0.18    # 阴影透明度
SAVE_FIG = True
# 新增参数：评分权重
AMP_WEIGHT = 0.6         # 振幅权重
TIME_PENALTY_WEIGHT = 0.4 # 时间惩罚权重

# ========= 工具函数 =========
def choose_time_axis(df: pd.DataFrame) -> np.ndarray:
    """优先使用 ['time','timestamp','datetime','x']，否则用顺序索引。"""
    for col in ["time", "timestamp", "datetime", "x"]:
        if col in df.columns:
            return df[col].values
    return np.arange(len(df))

def extend_up(values: np.ndarray, start: int, retrace_frac: float):
    """
    从 start 开始延申上涨趋势，逐步检查动态回撤：
    drawdown_t <= retrace_frac * profit_t（对所有 t 成立）
    违背时在最后一个最高点结束。
    返回：None 或 dict(i1,i2,amp,dir,max_retrace_ratio)
    """
    n = len(values)
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
    n = len(values)
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

def calculate_score(segment, amp_weight=0.7, time_penalty_weight=0.3):
    """
    计算综合评分，平衡收益和时间因素
    评分 = 振幅权重 * 振幅 - 时间惩罚权重 * 长度
    这样既奖励高振幅，又惩罚长时间持仓
    """
    amplitude = abs(segment["amp"])
    length = segment["i2"] - segment["i1"] + 1
    # 综合评分：振幅贡献减去时间惩罚
    score = amp_weight * amplitude - time_penalty_weight * length
    return score

def select_non_overlapping_topN(cands, top_n: int, min_len: int, min_amp: float):
    """
    按综合评分降序排序，再筛互不重叠，且满足最小长度/振幅。
    综合评分平衡了振幅和时间因素
    """
    # 为每个候选段计算综合评分
    for seg in cands:
        seg["score"] = calculate_score(seg, AMP_WEIGHT, TIME_PENALTY_WEIGHT)
    
    # 按综合评分降序排序
    cands_sorted = sorted(cands, key=lambda d: d["score"], reverse=True)
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

# ========= 主流程 =========
def main():
    df = pd.read_csv(CSV_FILE)

    # 必要列检查
    required_cols = {"index_value", "a", "b", "c", "d"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列：{required_cols}")

    times = choose_time_axis(df)
    # 上下颠倒 index_value
    prices = (-df["index_value"].values).astype(float)

    # 1) 基于动态 50% 回撤规则生成候选段
    candidates = build_candidates(prices, RETRACE_FRAC)

    # 2) 选取前 TOP_N 个互不重叠的区间（按综合评分排序）
    selected = select_non_overlapping_topN(candidates, TOP_N, MIN_LEN, MIN_AMP)

    print(f"[Info] 候选段数：{len(candidates)}；选中段数：{len(selected)}")
    for k, seg in enumerate(selected, 1):
        print(f"  #{k}: [{seg['i1']}, {seg['i2']}], dir={seg['dir']}, "
              f"amp={seg['amp']:.6f}, length={seg['i2']-seg['i1']+1}, "
              f"score={seg['score']:.6f}, max_retrace_seen={seg['max_retrace']:.3f}")

    # 3) 输出 CSV（每段的起止点与因子起止值；也附带段内因子均值）
    rows = []
    for seg in selected:
        i1, i2 = seg["i1"], seg["i2"]
        row = {
            "start_idx": i1,
            "end_idx": i2,
            "direction": "up" if seg["dir"] == 1 else "down",
            "start_time": times[i1],
            "end_time": times[i2],
            "start_value": -prices[i1],  # 恢复原始值用于输出
            "end_value": -prices[i2],    # 恢复原始值用于输出
            "amplitude": float(abs(seg["amp"])),
            "score": float(seg["score"]),
            "max_retrace_seen": float(seg["max_retrace"]),
            "length": int(i2 - i1 + 1),
        }
        for col in ["a", "b", "c", "d"]:
            vals = df[col].values[i1:i2+1].astype(float)
            row[f"{col}_start"] = float(df[col].values[i1])
            row[f"{col}_end"] = float(df[col].values[i2])
            row[f"{col}_mean"] = float(np.mean(vals))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv("top_trends.csv", index=False, encoding="utf-8-sig")
    print("已保存 top_trends.csv")

    # 4) 绘图：主图价格 + 阴影 + 三角；副图 a,b,c,d
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # 主图：价格（使用颠倒后的值绘图）
    ax1.plot(times, prices, color="black", linewidth=1.2, label="index_value")

    for seg in selected:
        i1, i2 = seg["i1"], seg["i2"]
        x1, x2 = times[i1], times[i2]
        y1, y2 = prices[i1], prices[i2]
        color = "red" if seg["dir"] == 1 else "blue"

        # 区间阴影与边界线
        ax1.add_patch(Rectangle((x1, min(y1, y2)),
                                x2 - x1 if x2 != x1 else 1e-9,
                                abs(y2 - y1) if y2 != y1 else 1e-9,
                                color=color, alpha=PLOT_FILL_ALPHA, zorder=1))
        ax1.plot(times[i1:i2+1], prices[i1:i2+1], color=color, linewidth=2.0, zorder=2)

        # 起止三角标
        if seg["dir"] == 1:  # 涨：起点^ 终点v（红）
            ax1.scatter(x1, y1, color=color, marker="^", s=90, zorder=3)
            ax1.scatter(x2, y2, color=color, marker="v", s=90, zorder=3)
        else:                # 跌：起点v 终点^（蓝）
            ax1.scatter(x1, y1, color=color, marker="v", s=90, zorder=3)
            ax1.scatter(x2, y2, color=color, marker="^", s=90, zorder=3)

        # 文本注释（振幅与最大回撤比）
        ax1.text((x1 + x2) / 2, max(y1, y2),
                 f"A={abs(seg['amp']):.3f} S={seg['score']:.3f} L={seg['i2']-seg['i1']+1}",
                 ha="center", va="bottom", fontsize=9, alpha=0.85)

    ax1.set_ylabel("index_value")
    ax1.set_title(f"最大区间（非重叠）+ 动态回撤 ≤ {int(RETRACE_FRAC*100)}% 规则")
    ax1.legend(loc="upper left")

    # 副图：因子 a,b,c,d
    colors = {"a": "tab:red", "b": "tab:blue", "c": "tab:green", "d": "tab:orange"}
    for col in ["a", "b", "c", "d"]:
        ax2.plot(times, df[col].values, label=col, color=colors[col])
    ax2.set_ylabel("factors")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig("top_trends.png", dpi=150)
        print("已保存 top_trends.png")
    plt.show()

if __name__ == "__main__":
    main()