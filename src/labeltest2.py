import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==== 参数 ====
csv_file = "240110.csv"   # 你的CSV文件
min_len = 5             # 最小时长（数据点数）
min_amp = 0.5           # 最小振幅
max_retrace = 0.5       # 最大允许回撤比例
smooth_win = 5          # EMA平滑窗口
top_k = 3               # 选Top3

# ==== 1. 读数据 ====
df = pd.read_csv(csv_file)
required_cols = {"index_value", "a", "b", "c", "d"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV必须包含列: {required_cols}")
if "x" not in df.columns:
    df["x"] = np.arange(len(df))

# 平滑
df["smooth"] = df["index_value"].ewm(span=smooth_win, adjust=False).mean()
y = df["smooth"].to_numpy()

# ==== 2. 延申趋势扫描 ====
segments = []
i = 0
n = len(y)

while i < n - 1:
    # 1) 确定趋势方向
    j = i + 1
    if y[j] > y[i]:
        direction = 1  # 上涨
        start_val = y[i]
        max_val = y[i]
        while j < n:
            if y[j] > max_val:
                max_val = y[j]
            retrace = (max_val - y[j]) / (max_val - start_val + 1e-9)
            if retrace > max_retrace:
                break
            j += 1
        end_idx = j - 1
    elif y[j] < y[i]:
        direction = -1  # 下跌
        start_val = y[i]
        min_val = y[i]
        while j < n:
            if y[j] < min_val:
                min_val = y[j]
            retrace = (y[j] - min_val) / (start_val - min_val + 1e-9)
            if retrace > max_retrace:
                break
            j += 1
        end_idx = j - 1
    else:
        i += 1
        continue

    # 2) 保存趋势段
    A = abs(y[end_idx] - y[i])
    D = end_idx - i
    if D >= min_len and A >= min_amp:
        seg_y = y[i:end_idx+1]
        xx = np.arange(len(seg_y))
        k, b = np.polyfit(xx, seg_y, 1)
        y_hat = k * xx + b
        ss_res = np.sum((seg_y - y_hat) ** 2)
        ss_tot = np.sum((seg_y - seg_y.mean()) ** 2)
        R2 = 1 - ss_res / (ss_tot + 1e-9)
        mono_steps = np.sign(np.diff(seg_y))
        M = (np.sum(mono_steps == np.sign(k)) / len(mono_steps)) if len(mono_steps) > 0 else 0.0
        score = (A / D) * max(R2, 0) * (1 - min(retrace, 1)) * M
        segments.append({
            "i1": i, "i2": end_idx, "A": A, "D": D, "R2": R2,
            "retrace": retrace, "M": M, "score": score, "dir": direction
        })

    i = end_idx + 1

# ==== 3. 选TopK ====
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

# ==== 4. 输出CSV ====
output_rows = []
for seg in selected:
    row = seg.copy()
    row["x_start"] = df.loc[seg["i1"], "x"]
    row["x_end"] = df.loc[seg["i2"], "x"]
    for col in ["a", "b", "c", "d"]:
        row[f"{col}_start"] = df.loc[seg["i1"], col]
        row[f"{col}_end"] = df.loc[seg["i2"], col]
    output_rows.append(row)

out_df = pd.DataFrame(output_rows)
out_df.to_csv("top_trends.csv", index=False)
print("已保存 top_trends.csv（延申趋势版）")

# ==== 5. 绘图 ====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
ax1.plot(df["x"], df["index_value"], color="gray", alpha=0.6, label="原始")
ax1.plot(df["x"], df["smooth"], color="black", linewidth=1.2, label="平滑")

for seg in selected:
    x1, x2 = df.loc[seg["i1"], "x"], df.loc[seg["i2"], "x"]
    y1, y2 = df.loc[seg["i1"], "smooth"], df.loc[seg["i2"], "smooth"]
    ax1.add_patch(Rectangle((x1, min(y1, y2)), x2 - x1, abs(y2 - y1),
                            color="red" if seg["dir"] == 1 else "blue", alpha=0.15))
    if seg["dir"] == 1:
        ax1.scatter(x1, y1, color="red", marker="^", s=100)
        ax1.scatter(x2, y2, color="red", marker="v", s=100)
    else:
        ax1.scatter(x1, y1, color="blue", marker="v", s=100)
        ax1.scatter(x2, y2, color="blue", marker="^", s=100)

ax1.legend()
ax1.set_ylabel("Index Value")
ax1.set_title("日内趋势Top3标记（延申趋势法） + 因子曲线")

colors = {"a": "tab:red", "b": "tab:blue", "c": "tab:green", "d": "tab:orange"}
for col in ["a", "b", "c", "d"]:
    ax2.plot(df["x"], df[col], label=col, color=colors[col])
ax2.legend()
ax2.set_ylabel("因子值")

plt.tight_layout()
plt.show()
