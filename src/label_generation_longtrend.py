import numpy as np
from scipy.signal import find_peaks

def count_reversals(window_vals):
    peaks_pos, _ = find_peaks(window_vals)
    valleys_pos, _ = find_peaks(-window_vals)
    return len(peaks_pos) + len(valleys_pos)

def generate_label_with_reversals(df, window_size=300, change_threshold=0.01, max_reversals=5):
    values = df['index_value'].values
    labels = [0]*len(values)
    
    print(f"数据总长度: {len(values)}")
    print(f"窗口数量: {len(values) - window_size}")

    valid_windows = 0
    up_trends = 0
    down_trends = 0
    
    returns_list = []

    for i in range(len(values) - window_size):
        window_vals = values[i:i+window_size]
        # 计算窗口内收益率
        ret = (window_vals[-1] - window_vals[0]) / window_vals[0]
        returns_list.append(ret)

        # 计算反转次数
        rev_count = count_reversals(window_vals)
        
        # 放宽条件，只要收益率满足阈值就标记
        if abs(ret) >= change_threshold:
            valid_windows += 1
            center = i + window_size // 2
            if ret > change_threshold:
                labels[center] = 1  # 上涨趋势
                up_trends += 1
            elif ret < -change_threshold:
                labels[center] = 2  # 下跌趋势
                down_trends += 1

    print(f"有效窗口数: {valid_windows}")
    print(f"上涨趋势标签数: {up_trends}")
    print(f"下跌趋势标签数: {down_trends}")
    
    if returns_list:
        print(f"收益率统计:")
        print(f"  最大收益率: {np.max(returns_list):.4f}")
        print(f"  最小收益率: {np.min(returns_list):.4f}")
        print(f"  平均收益率: {np.mean(returns_list):.4f}")
        print(f"  收益率标准差: {np.std(returns_list):.4f}")
        
        positive_returns = [r for r in returns_list if r > 0]
        negative_returns = [r for r in returns_list if r < 0]
        print(f"正收益窗口比例: {len(positive_returns)/len(returns_list):.2%}")
        print(f"负收益窗口比例: {len(negative_returns)/len(returns_list):.2%}")

    df['label'] = labels
    return df

# 使用分位数方法生成更平衡的标签（推荐使用）
def generate_label_balanced(df, window_size=300, percentile=70):
    """
    使用分位数方法生成更平衡的标签
    """
    values = df['index_value'].values
    labels = [0]*len(values)
    
    returns_list = []
    indices_list = []
    
    # 先计算所有窗口的收益率
    for i in range(len(values) - window_size):
        window_vals = values[i:i+window_size]
        ret = (window_vals[-1] - window_vals[0]) / window_vals[0]
        returns_list.append(ret)
        indices_list.append(i)
    
    if not returns_list:
        print("没有足够的数据生成标签")
        df['label'] = labels
        return df
    
    # 计算分位数阈值
    upper_threshold = np.percentile(returns_list, percentile)
    lower_threshold = np.percentile(returns_list, 100-percentile)
    
    print(f"上涨阈值({percentile}分位数): {upper_threshold:.4f}")
    print(f"下跌阈值({100-percentile}分位数): {lower_threshold:.4f}")
    
    up_trends = 0
    down_trends = 0
    
    # 为高收益和低收益窗口分配标签
    for i, (ret, idx) in enumerate(zip(returns_list, indices_list)):
        center = idx + window_size // 2
        if ret >= upper_threshold:
            labels[center] = 1  # 上涨趋势
            up_trends += 1
        elif ret <= lower_threshold:
            labels[center] = 2  # 下跌趋势
            down_trends += 1

    print(f"上涨趋势标签数: {up_trends}")
    print(f"下跌趋势标签数: {down_trends}")
    print(f"总标签数: {up_trends + down_trends}")
    
    # 显示标签分布
    unique, counts = np.unique(labels, return_counts=True)
    print(f"标签分布: {dict(zip(unique, counts))}")
    
    df['label'] = labels
    return df

# 简化版本：基于简单移动平均的趋势判断
def generate_label_sma(df, window_size=300, sma_window=50):
    """
    基于简单移动平均的趋势判断生成标签
    """
    values = df['index_value'].values
    labels = [0]*len(values)
    
    # 计算简单移动平均
    if len(values) >= sma_window:
        sma = np.convolve(values, np.ones(sma_window)/sma_window, mode='valid')
        # 对齐SMA和原始数据
        sma = np.concatenate([np.full(sma_window//2, np.nan), sma, np.full(len(values)-len(sma)-sma_window//2, np.nan)])
        
        # 计算SMA的变化率
        sma_changes = np.diff(sma, prepend=sma[0])
        
        # 根据SMA变化率生成标签
        threshold = np.nanstd(sma_changes) * 0.5  # 使用标准差的倍数作为阈值
        
        for i in range(window_size//2, len(values) - window_size//2):
            if not np.isnan(sma_changes[i]):
                if sma_changes[i] > threshold:
                    labels[i] = 1  # 上涨趋势
                elif sma_changes[i] < -threshold:
                    labels[i] = 2  # 下跌趋势
    
    # 显示标签分布
    unique, counts = np.unique(labels, return_counts=True)
    print(f"SMA方法标签分布: {dict(zip(unique, counts))}")
    
    df['label'] = labels
    return df