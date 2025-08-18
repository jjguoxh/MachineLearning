import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def add_features(df):
    """
    为数据添加特征，仅使用index_value, a, b, c, d列
    """
    df = df.copy()
    
    # 基础特征
    df['index_value_pct_change'] = df['index_value'].pct_change().replace([np.inf, -np.inf], np.nan)
    df['a_pct_change'] = df['a'].pct_change().replace([np.inf, -np.inf], np.nan)
    df['b_pct_change'] = df['b'].pct_change().replace([np.inf, -np.inf], np.nan)
    df['c_pct_change'] = df['c'].pct_change().replace([np.inf, -np.inf], np.nan)
    df['d_pct_change'] = df['d'].pct_change().replace([np.inf, -np.inf], np.nan)
    
    # 移动平均线特征
    for window in [5, 10, 20]:
        df[f'index_value_ma_{window}'] = df['index_value'].rolling(window=window).mean()
        df[f'a_ma_{window}'] = df['a'].rolling(window=window).mean()
        df[f'b_ma_{window}'] = df['b'].rolling(window=window).mean()
        df[f'c_ma_{window}'] = df['c'].rolling(window=window).mean()
        df[f'd_ma_{window}'] = df['d'].rolling(window=window).mean()
        
        # 价格与均线的关系
        df[f'index_value_above_ma_{window}'] = (df['index_value'] > df[f'index_value_ma_{window}']).astype(int)
        df[f'a_above_ma_{window}'] = (df['a'] > df[f'a_ma_{window}']).astype(int)
        df[f'b_above_ma_{window}'] = (df['b'] > df[f'b_ma_{window}']).astype(int)
        df[f'c_above_ma_{window}'] = (df['c'] > df[f'c_ma_{window}']).astype(int)
        df[f'd_above_ma_{window}'] = (df['d'] > df[f'd_ma_{window}']).astype(int)
    
    # 新增特征1：a,b,c与index_value趋势一致性
    df['abc_trend_consistency'] = 0
    trend_window = 5  # 考虑5个点的趋势
    for i in range(trend_window, len(df)):
        # 计算index_value的趋势
        index_trend = df['index_value'].iloc[i] - df['index_value'].iloc[i-trend_window]
        # 计算a,b,c的趋势
        a_trend = df['a'].iloc[i] - df['a'].iloc[i-trend_window]
        b_trend = df['b'].iloc[i] - df['b'].iloc[i-trend_window]
        c_trend = df['c'].iloc[i] - df['c'].iloc[i-trend_window]
        
        # 判断趋势一致性（同向为正）
        consistent_count = 0
        if (index_trend > 0 and a_trend > 0) or (index_trend < 0 and a_trend < 0):
            consistent_count += 1
        if (index_trend > 0 and b_trend > 0) or (index_trend < 0 and b_trend < 0):
            consistent_count += 1
        if (index_trend > 0 and c_trend > 0) or (index_trend < 0 and c_trend < 0):
            consistent_count += 1
            
        df.loc[i, 'abc_trend_consistency'] = consistent_count / 3.0  # 一致性比例
    
    # 新增特征2：反转信号特征
    df['reversal_signal'] = 0
    lookback_window = 5
    for i in range(lookback_window, len(df)):
        current_idx = df['index_value'].iloc[i]
        prev_idx = df['index_value'].iloc[i-lookback_window]
        
        if current_idx > prev_idx:  # index_value上涨
            current_a, prev_a = df['a'].iloc[i], df['a'].iloc[i-lookback_window]
            current_b, prev_b = df['b'].iloc[i], df['b'].iloc[i-lookback_window]
            current_c, prev_c = df['c'].iloc[i], df['c'].iloc[i-lookback_window]
            current_d, prev_d = df['d'].iloc[i], df['d'].iloc[i-lookback_window]
            
            # a,b,c都下跌且d不反对（也下跌或持平）
            if (current_a < prev_a and current_b < prev_b and current_c < prev_c and 
                (current_d <= prev_d)):
                df.loc[i, 'reversal_signal'] = -1  # 可能的下跌反转
        else:  # index_value下跌
            current_a, prev_a = df['a'].iloc[i], df['a'].iloc[i-lookback_window]
            current_b, prev_b = df['b'].iloc[i], df['b'].iloc[i-lookback_window]
            current_c, prev_c = df['c'].iloc[i], df['c'].iloc[i-lookback_window]
            current_d, prev_d = df['d'].iloc[i], df['d'].iloc[i-lookback_window]
            
            # a,b,c都上涨且d不反对（也上涨或持平）
            if (current_a > prev_a and current_b > prev_b and current_c > prev_c and 
                (current_d >= prev_d)):
                df.loc[i, 'reversal_signal'] = 1  # 可能的上涨反转
    
    # 新增特征3：峰值和谷值特征
    # 找到index_value的峰值和谷值点
    peaks, _ = find_peaks(df['index_value'], distance=5)  # 峰值点
    valleys, _ = find_peaks(-df['index_value'], distance=5)  # 谷值点（对负值找峰值）
    
    df['is_peak'] = 0
    df['is_valley'] = 0
    df['peak_valley_distance'] = 0  # 距离最近的峰/谷的距离
    
    # 标记峰/谷点
    df.loc[peaks, 'is_peak'] = 1
    df.loc[valleys, 'is_valley'] = 1
    
    # 计算距离最近的峰/谷的距离
    for i in range(len(df)):
        # 找到最近的峰
        peak_distances = np.abs(peaks - i) if len(peaks) > 0 else np.array([len(df)])
        nearest_peak_dist = np.min(peak_distances) if len(peak_distances) > 0 else len(df)
        
        # 找到最近的谷
        valley_distances = np.abs(valleys - i) if len(valleys) > 0 else np.array([len(df)])
        nearest_valley_dist = np.min(valley_distances) if len(valley_distances) > 0 else len(df)
        
        # 取最近的距离
        df.loc[i, 'peak_valley_distance'] = min(nearest_peak_dist, nearest_valley_dist)
    
    # 价格波动特征
    df['volatility'] = df['index_value'].rolling(window=10).std()
    df['price_range'] = df['index_value'].rolling(window=10).max() - df['index_value'].rolling(window=10).min()
    
    # 滞后特征
    for lag in [1, 2, 3, 5]:
        df[f'index_value_lag_{lag}'] = df['index_value'].shift(lag)
        df[f'index_value_pct_change_lag_{lag}'] = df['index_value_pct_change'].shift(lag)
        df[f'a_lag_{lag}'] = df['a'].shift(lag)
        df[f'b_lag_{lag}'] = df['b'].shift(lag)
        df[f'c_lag_{lag}'] = df['c'].shift(lag)
        df[f'd_lag_{lag}'] = df['d'].shift(lag)
    
    # 差分特征
    for diff in [1, 2, 3]:
        df[f'index_value_diff_{diff}'] = df['index_value'].diff(diff)
        df[f'a_diff_{diff}'] = df['a'].diff(diff)
        df[f'b_diff_{diff}'] = df['b'].diff(diff)
        df[f'c_diff_{diff}'] = df['c'].diff(diff)
        df[f'd_diff_{diff}'] = df['d'].diff(diff)
    
    # 填充缺失值
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # 替换无穷大值和过大值
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # 替换无穷大值为NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # 将过大值限制在合理范围内
        if df[col].dtype == np.float64:
            # 对于浮点数，限制在±1e6范围内
            df[col] = np.clip(df[col], -1e6, 1e6)
    
    # 再次填充NaN值
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)  # 最后用0填充任何剩余的NaN值
    
    return df