import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV
file_path = "250307.csv"
df = pd.read_csv(file_path)

# 自动时间列处理
if 'time' in df.columns:
    times = df['time'].values
else:
    times = np.arange(len(df))  # 用索引代替时间

# 取数据列
index_values = df['index_value'].values
a_values = df['a'].values
b_values = df['b'].values
c_values = df['c'].values
d_values = df['d'].values

# 最大区间搜索算法
# def find_top_amplitude_segments(values, top_n=5, min_length=5):
#     segments = []
#     used = np.zeros(len(values), dtype=bool)

#     for _ in range(top_n):
#         best_amp = 0
#         best_seg = None
#         for i in range(len(values) - min_length):
#             for j in range(i + min_length, len(values)):
#                 if np.any(used[i:j+1]):
#                     continue  # 跳过重叠
#                 amp = abs(values[j] - values[i])
#                 if amp > best_amp:
#                     best_amp = amp
#                     best_seg = (i, j)
#         if best_seg:
#             segments.append(best_seg)
#             used[best_seg[0]:best_seg[1]+1] = True
#     return segments
def find_top_amplitude_segments(values, top_n=5, min_length=5):
    segments = []
    used = np.zeros(len(values), dtype=bool)
    
    # 预计算所有可能的区间振幅
    amplitudes = []
    for i in range(len(values) - min_length):
        for j in range(i + min_length, len(values)):
            if not np.any(used[i:j+1]):
                amp = abs(values[j] - values[i])
                amplitudes.append((amp, i, j))
    
    # 按振幅排序
    amplitudes.sort(reverse=True, key=lambda x: x[0])
    
    # 选择前top_n个不重叠的区间
    for amp, i, j in amplitudes:
        if len(segments) >= top_n:
            break
        if not np.any(used[i:j+1]):
            segments.append((i, j))
            used[i:j+1] = True
    
    return segments
# 找出前5的振幅空间
segments = find_top_amplitude_segments(index_values, top_n=2, min_length=5)

# 过滤中间回调超过50%的波段
def has_large_pullback(values, start, end, threshold=0.5):
    segment = values[start:end+1]
    peak = np.max(segment)
    trough = np.min(segment)
    amplitude = peak - trough
    if amplitude == 0:
        return False
    # 检查趋势方向
    if values[end] > values[start]:  # 上涨趋势
        max_drawdown = np.max(peak - segment)
    else:  # 下跌趋势
        max_drawdown = np.max(segment - trough)
    return (max_drawdown / amplitude) >= threshold

filtered_segments = [
    seg for seg in segments if has_large_pullback(index_values, seg[0], seg[1], threshold=0.5)
]

# 保存结果
output_rows = []
for start, end in filtered_segments:
    output_rows.append({
        "start_time": times[start],
        "end_time": times[end],
        "start_value": index_values[start],
        "end_value": index_values[end],
        "a_start": a_values[start],
        "a_end": a_values[end],
        "b_start": b_values[start],
        "b_end": b_values[end],
        "c_start": c_values[start],
        "c_end": c_values[end],
        "d_start": d_values[start],
        "d_end": d_values[end]
    })

pd.DataFrame(output_rows).to_csv("top_trends.csv", index=False)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(times, index_values, label="Index Value", color='black')
plt.plot(times, a_values, label="a", color='red', alpha=0.6)
plt.plot(times, b_values, label="b", color='blue', alpha=0.6)
plt.plot(times, c_values, label="c", color='green', alpha=0.6)
plt.plot(times, d_values, label="d", color='orange', alpha=0.6)

for start, end in filtered_segments:
    plt.axvspan(times[start], times[end], color='yellow', alpha=0.3)
    plt.scatter(times[start], index_values[start], color='red', marker='^', s=100)
    plt.scatter(times[end], index_values[end], color='blue', marker='v', s=100)

plt.legend()
plt.title("Top Amplitude Segments with >50% Pullback")
plt.savefig("top_trends.png", dpi=150)
plt.show()
