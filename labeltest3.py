# -*- coding: utf-8 -*-
"""
延申趋势 + 动态回撤比例自动搜索 版
- 读取 CSV: 需要列 x,index_value,a,b,c,d（若无 x 则自动用顺序索引）
- 自动搜索最佳回撤比例（默认 0.10~0.50）
- 延申趋势：允许回撤未超阈值则继续延申，超过则结束
- 评分：Score = (A/D) * max(R2,0) * (1 - max_retrace_seen) * M
- 选 Top3 非重叠段；绘图 + 导出 top_trends.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple

# ========== 可调参数 ==========
CSV_FILE = "240112.csv"     # 你的 CSV 文件路径
SMOOTH_WIN = 5            # EMA 平滑窗口（越大越平）
MIN_LEN = 5               # 趋势段最小时长（点数）
MIN_AMP = 0.5             # 趋势段最小振幅（平滑后的 index_value 单位）
TOP_K = 10                 # 选择非重叠的前 K 段
# 回撤比例搜索网格（含起止值），例如 0.10~0.50 每 0.05 一格
RETRACE_GRID = np.linspace(0.10, 0.50, 9)  # [0.10, 0.15, ..., 0.50]
# 评估一个回撤比例的指标：'sum'（TopK分数求和）或 'mean'（TopK均值）
MODEL_SELECT_METRIC = "sum"


# ========== 核心函数 ==========
def compute_segment_metrics(seg_y: np.ndarray) -> Tuple[float, float, float]:
    """
    给定一段 y，计算线性回归 R2、单调性 M、斜率符号 sign_k
    """
    xx = np.arange(len(seg_y))
    # 线性拟合
    k, b = np.polyfit(xx, seg_y, 1)
    y_hat = k * xx + b
    ss_res = float(np.sum((seg_y - y_hat) ** 2))
    ss_tot = float(np.sum((seg_y - np.mean(seg_y)) ** 2))
    R2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # 单调性：与回归斜率同向的步子占比
    steps = np.sign(np.diff(seg_y))
    sign_k = np.sign(k) if k != 0 else 0.0
    M = (np.sum(steps == sign_k) / len(steps)) if len(steps) > 0 else 0.0
    return R2, M, float(sign_k)


def extend_trends(y: np.ndarray,
                  max_retrace: float,
                  min_len: int,
                  min_amp: float) -> List[Dict]:
    """
    延申趋势扫描：允许回撤 <= max_retrace 时继续延申，超过则结束。
    返回所有合格的趋势段（不做非重叠筛选）。
    每个段字典包含：i1,i2,A,D,R2,retrace,M,score,dir
    """
    n = len(y)
    segs: List[Dict] = []
    i = 0

    while i < n - 1:
        j = i + 1
        # 确定初始方向
        if y[j] > y[i]:
            direction = 1  # 上涨
            start_val = y[i]
            # 用于回撤计算的“最高点”
            run_max = y[i]
            max_retrace_seen = 0.0
            while j < n:
                if y[j] > run_max:
                    run_max = y[j]
                denom = max(run_max - start_val, 1e-12)
                retrace_now = (run_max - y[j]) / denom
                max_retrace_seen = max(max_retrace_seen, retrace_now)
                if retrace_now > max_retrace:
                    break
                j += 1
            end_idx = j - 1
        elif y[j] < y[i]:
            direction = -1  # 下跌
            start_val = y[i]
            # 用于回撤计算的“最低点”
            run_min = y[i]
            max_retrace_seen = 0.0
            while j < n:
                if y[j] < run_min:
                    run_min = y[j]
                denom = max(start_val - run_min, 1e-12)
                retrace_now = (y[j] - run_min) / denom
                max_retrace_seen = max(max_retrace_seen, retrace_now)
                if retrace_now > max_retrace:
                    break
                j += 1
            end_idx = j - 1
        else:
            # 水平，跳过该点
            i += 1
            continue

        # 形成一段
        A = abs(y[end_idx] - y[i])
        D = end_idx - i

        if D >= min_len and A >= min_amp:
            seg_y = y[i:end_idx + 1]
            R2, M, _ = compute_segment_metrics(seg_y)
            score = (A / D) * max(R2, 0.0) * (1.0 - min(max_retrace_seen, 1.0)) * M
            segs.append({
                "i1": i, "i2": end_idx, "A": float(A), "D": int(D),
                "R2": float(R2), "retrace": float(max_retrace_seen),
                "M": float(M), "score": float(score), "dir": int(direction)
            })

        # 下一个起点从该段结束后开始
        i = max(end_idx + 1, i + 1)

    return segs


def select_non_overlapping_topk(segments: List[Dict], k: int) -> List[Dict]:
    """
    从候选段中按 score 降序选取 TopK，要求互不重叠
    """
    segs_sorted = sorted(segments, key=lambda s: s["score"], reverse=True)
    selected: List[Dict] = []
    used = np.zeros(0, dtype=bool)  # 惰性，改用区间判断
    used_ranges: List[Tuple[int, int]] = []

    def overlaps(a1: int, a2: int, b1: int, b2: int) -> bool:
        return not (a2 < b1 or b2 < a1)

    for seg in segs_sorted:
        if len(selected) >= k:
            break
        ok = True
        for (u1, u2) in used_ranges:
            if overlaps(seg["i1"], seg["i2"], u1, u2):
                ok = False
                break
        if ok:
            selected.append(seg)
            used_ranges.append((seg["i1"], seg["i2"]))
    return selected


def search_best_retrace_ratio(y: np.ndarray,
                              grid: np.ndarray,
                              min_len: int,
                              min_amp: float,
                              k: int,
                              metric: str = "sum") -> Tuple[float, List[Dict]]:
    """
    在给定回撤比例网格上搜索最优比例。
    - 对每个比例生成候选段 -> 选 TopK 非重叠 -> 计算指标
    - 指标：TopK 分数之和（或均值）
    返回：最佳回撤比例、该比例下的 TopK 段
    """
    best_ratio = None
    best_score = -np.inf
    best_segments: List[Dict] = []

    for r in grid:
        all_segs = extend_trends(y, max_retrace=float(r),
                                 min_len=min_len, min_amp=min_amp)
        topk = select_non_overlapping_topk(all_segs, k=k)
        if len(topk) == 0:
            agg = -np.inf
        else:
            scores = [s["score"] for s in topk]
            agg = np.sum(scores) if metric == "sum" else np.mean(scores)
        # 选择更优
        if agg > best_score:
            best_score = float(agg)
            best_ratio = float(r)
            best_segments = topk

    # 若完全找不到，回退到最大比例
    if best_ratio is None:
        best_ratio = float(grid[-1])
        best_segments = []

    return best_ratio, best_segments


# ========== 主流程 ==========
def main():
    # 读取数据
    df = pd.read_csv(CSV_FILE)
    required = {"index_value", "a", "b", "c", "d"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列: {required}")
    if "x" not in df.columns:
        df["x"] = np.arange(len(df))

    # 轻度平滑
    df["smooth"] = df["index_value"].ewm(span=SMOOTH_WIN, adjust=False).mean()
    y = df["smooth"].to_numpy()

    # 自动搜索最佳回撤比例
    best_r, top_segments = search_best_retrace_ratio(
        y, RETRACE_GRID, MIN_LEN, MIN_AMP, TOP_K, MODEL_SELECT_METRIC
    )
    print(f"[Info] 最佳回撤比例 = {best_r:.3f} ，选出 {len(top_segments)} 段")

    # 如果搜索到的段不足 TOP_K，提示但继续输出/绘图
    if len(top_segments) == 0:
        print("[Warn] 未找到满足条件的趋势段，请适当降低 MIN_AMP 或 MIN_LEN，或放宽回撤比例范围。")

    # 导出 top_trends.csv
    rows = []
    for seg in top_segments:
        r = dict(seg)
        r["x_start"] = df.loc[seg["i1"], "x"]
        r["x_end"] = df.loc[seg["i2"], "x"]
        r["index_value_start"] = df.loc[seg["i1"], "index_value"]
        r["index_value_end"] = df.loc[seg["i2"], "index_value"]
        for col in ["a", "b", "c", "d"]:
            r[f"{col}_start"] = df.loc[seg["i1"], col]
            r[f"{col}_end"] = df.loc[seg["i2"], col]
        rows.append(r)
    out_df = pd.DataFrame(rows)
    out_df.to_csv("top_trends.csv", index=False, encoding="utf-8-sig")
    print("已保存 top_trends.csv（含 a,b,c,d 与 index_value 起止值）")

    # 绘图：主图价格 + 阴影 + 三角；副图 a,b,c,d
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # 主图：原始 & 平滑
    ax1.plot(df["x"], df["index_value"], color="gray", alpha=0.55, label="原始")
    ax1.plot(df["x"], df["smooth"], color="black", linewidth=1.2, label="平滑")

    for seg in top_segments:
        x1, x2 = df.loc[seg["i1"], "x"], df.loc[seg["i2"], "x"]
        y1, y2 = df.loc[seg["i1"], "smooth"], df.loc[seg["i2"], "smooth"]

        # 阴影区（涨=红，跌=蓝）
        ax1.add_patch(
            Rectangle(
                (x1, min(y1, y2)),
                x2 - x1,
                abs(y2 - y1),
                color=("red" if seg["dir"] == 1 else "blue"),
                alpha=0.15,
                zorder=1,
            )
        )
        # 起止点三角
        if seg["dir"] == 1:  # 涨段
            ax1.scatter(x1, y1, color="red", marker="^", s=100, zorder=3)
            ax1.scatter(x2, y2, color="red", marker="v", s=100, zorder=3)
        else:  # 跌段
            ax1.scatter(x1, y1, color="blue", marker="v", s=100, zorder=3)
            ax1.scatter(x2, y2, color="blue", marker="^", s=100, zorder=3)

        # 文本标注（可选：显示 Score）
        ax1.text((x1 + x2) / 2, max(y1, y2),
                 f"S={seg['score']:.3f}",
                 ha="center", va="bottom", fontsize=9, alpha=0.8)

    ax1.set_ylabel("Index Value")
    ax1.set_title(f"延申趋势 Top{TOP_K}（自动回撤阈值搜索，best={best_r:.2f}）")
    ax1.legend(loc="upper left")

    # 副图：因子 a,b,c,d
    colors = {"a": "tab:red", "b": "tab:blue", "c": "tab:green", "d": "tab:orange"}
    for col in ["a", "b", "c", "d"]:
        ax2.plot(df["x"], df[col], label=col, color=colors[col])
    ax2.set_ylabel("因子值")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
