import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.patches import Rectangle

# ==== 参数 ====
csv_file = "240112.csv"   # 你的CSV文件
min_len = 5             # 最小时长（数据点数）
min_amp = 0.5           # 最小振幅
max_retrace = 0.5       # 最大允许回撤比例
min_prom = 0.2          # 峰谷检测的prominence
min_dist = 5            # 峰谷检测的最小间距
smooth_win = 5          # EMA平滑窗口
top_k = 3               # 选Top3

# ==== 1. 读数据 ====
df = pd.read_csv(csv_file)

# 检查必要列
required_cols = {"index_value", "a", "b", "c", "d"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV必须包含列: {required_cols}")
if "x" not in df.columns:
    df["x"] = np.arange(len(df))

# EMA平滑
df["smooth"] = df["index_value"].ewm(span=smooth_win, adjust=False).mean()

# ==== 2. 找峰谷 ====
y = df["smooth"].to_numpy()
peaks, _ = find_peaks(y, distance=min_dist, prominence=min_prom)
valleys, _ = find_peaks(-y, distance=min_dist, prominence=min_prom)

# 合并极值并按顺序
extrema = sorted(
    [(i, y[i], "peak") for i in peaks] + [(i, y[i], "valley") for i in valleys],
    key=lambda t: t[0]
)

# ==== 3. 构造候选趋势段并打分 ====
segments = []
for (i1, v1, t1), (i2, v2, t2) in zip(extrema[:-1], extrema[1:]):
    if i2 <= i1:
        continue
    A = abs(v2 - v1)
    D = i2 - i1
    if D < min_len or A < min_amp:
        continue

    seg_y = y[i1:i2+1]

    # 回撤比例
    if t1 == "valley" and t2 == "peak":  # 涨段
        retrace = (v1 - np.min(seg_y)) / A if A != 0 else 0
        direction = 1
    elif t1 == "peak" and t2 == "valley":  # 跌段
        retrace = (np.max(seg_y) - v1) / A if A != 0 else 0
        direction = -1
    else:
        continue
    if retrace > max_retrace:
        continue

    # R²
    xx = np.arange(len(seg_y))
    k, b = np.polyfit(xx, seg_y, 1)
    y_hat = k * xx + b
    ss_res = np.sum((seg_y - y_hat) ** 2)
    ss_tot = np.sum((seg_y - seg_y.mean()) ** 2)
    R2 = 1 - ss_res / (ss_tot + 1e-9)

    # 单调性
    mono_steps = np.sign(np.diff(seg_y))
    M = (np.sum(mono_steps == np.sign(k)) / len(mono_steps)) if len(mono_steps) > 0 else 0.0

    score = (A / D) * max(R2, 0) * (1 - retrace) * M

    segments.append({
        "i1": i1, "i2": i2, "A": A, "D": D, "R2": R2,
        "retrace": retrace, "M": M, "score": score, "dir": direction
    })

# ==== 4. 选Top3非重叠段 ====
segments.sort(key=lambda s: s["score"], reverse=True)
selected = []
used_idx = set()

for seg in segments:
    if len(selected) >= top_k:
        break
    if any(idx in used_idx for idx in range(seg["i1"], seg["i2"] + 1)):
        continue
    selected.append(seg)
    used_idx.update(range(seg["i1"], seg["i2"] + 1))

# ==== 5. 输出段信息 ====
output_rows = []
for seg in selected:
    row = seg.copy()
    row["x_start"] = df.loc[seg["i1"], "x"]
    row["x_end"] = df.loc[seg["i2"], "x"]
    # 因子值（起点和终点）
    for col in ["a", "b", "c", "d"]:
        row[f"{col}_start"] = df.loc[seg["i1"], col]
        row[f"{col}_end"] = df.loc[seg["i2"], col]
    output_rows.append(row)

out_df = pd.DataFrame(output_rows)
out_df.to_csv("top_trends.csv", index=False)
print("已保存 top_trends.csv，包含因子a,b,c,d的起止值")

# ==== 6. 绘图 ====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})

# 主图：价格曲线
ax1.plot(df["x"], df["index_value"], color="gray", alpha=0.6, label="原始")
ax1.plot(df["x"], df["smooth"], color="black", linewidth=1.2, label="平滑")

for seg in selected:
    x1, x2 = df.loc[seg["i1"], "x"], df.loc[seg["i2"], "x"]
    y1, y2 = df.loc[seg["i1"], "smooth"], df.loc[seg["i2"], "smooth"]

    # 阴影
    ax1.add_patch(Rectangle((x1, min(y1, y2)), x2 - x1, abs(y2 - y1),
                            color="red" if seg["dir"] == 1 else "blue", alpha=0.15))

    # 起止点三角
    if seg["dir"] == 1:  # 涨段红色
        ax1.scatter(x1, y1, color="red", marker="^", s=100)
        ax1.scatter(x2, y2, color="red", marker="v", s=100)
    else:  # 跌段蓝色
        ax1.scatter(x1, y1, color="blue", marker="v", s=100)
        ax1.scatter(x2, y2, color="blue", marker="^", s=100)

ax1.legend()
ax1.set_ylabel("Index Value")
ax1.set_title("日内趋势Top3标记（峰谷评分法） + 因子曲线")

# 副图：因子曲线
colors = {"a": "tab:red", "b": "tab:blue", "c": "tab:green", "d": "tab:orange"}
for col in ["a", "b", "c", "d"]:
    ax2.plot(df["x"], df[col], label=col, color=colors[col])
ax2.legend()
ax2.set_ylabel("因子值")

plt.tight_layout()
plt.show()
