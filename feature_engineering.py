# feature_engineering.py
import numpy as np
import pandas as pd

def add_features(df):
    # 原始特征工程
    df['a_diff'] = df['a'].diff().fillna(0)
    df['b_diff'] = df['b'].diff().fillna(0)
    df['c_diff'] = df['c'].diff().fillna(0)
    df['d_diff'] = df['d'].diff().fillna(0)

    df['a_roll_mean_5'] = df['a'].rolling(window=5).mean().bfill()
    df['b_roll_mean_5'] = df['b'].rolling(window=5).mean().bfill()
    df['c_roll_mean_5'] = df['c'].rolling(window=5).mean().bfill()
    df['d_roll_mean_5'] = df['d'].rolling(window=5).mean().bfill()
    
    # 增强特征工程
    
    # 更多滚动窗口统计特征
    for window in [10, 20]:
        df[f'a_roll_mean_{window}'] = df['a'].rolling(window=window).mean().bfill()
        df[f'b_roll_mean_{window}'] = df['b'].rolling(window=window).mean().bfill()
        df[f'c_roll_mean_{window}'] = df['c'].rolling(window=window).mean().bfill()
        df[f'd_roll_mean_{window}'] = df['d'].rolling(window=window).mean().bfill()
        
        df[f'a_roll_std_{window}'] = df['a'].rolling(window=window).std().bfill()
        df[f'b_roll_std_{window}'] = df['b'].rolling(window=window).std().bfill()
        df[f'c_roll_std_{window}'] = df['c'].rolling(window=window).std().bfill()
        df[f'd_roll_std_{window}'] = df['d'].rolling(window=window).std().bfill()
    
    # 滞后特征
    for lag in [1, 2, 3, 5]:
        df[f'a_lag_{lag}'] = df['a'].shift(lag).bfill()
        df[f'b_lag_{lag}'] = df['b'].shift(lag).bfill()
        df[f'c_lag_{lag}'] = df['c'].shift(lag).bfill()
        df[f'd_lag_{lag}'] = df['d'].shift(lag).bfill()
    
    # 比率特征
    df['a_b_ratio'] = np.where(df['b'] != 0, df['a'] / df['b'], 0)
    df['c_d_ratio'] = np.where(df['d'] != 0, df['c'] / df['d'], 0)
    
    # 交叉特征
    df['a_c_interaction'] = df['a'] * df['c']
    df['b_d_interaction'] = df['b'] * df['d']
    
    # 波动率特征
    df['a_volatility'] = df['a_diff'].rolling(window=10).std().bfill()
    df['b_volatility'] = df['b_diff'].rolling(window=10).std().bfill()
    df['c_volatility'] = df['c_diff'].rolling(window=10).std().bfill()
    df['d_volatility'] = df['d_diff'].rolling(window=10).std().bfill()
    
    # RSI风格指标（简化版）
    def calculate_simple_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = np.where(loss != 0, gain / loss, 0)
        rsi = 100 - (100 / (1 + rs))
        # 将numpy数组转换为pandas Series后使用bfill
        return pd.Series(rsi).bfill().values
    
    df['a_rsi'] = calculate_simple_rsi(df['a'])
    df['b_rsi'] = calculate_simple_rsi(df['b'])
    df['c_rsi'] = calculate_simple_rsi(df['c'])
    df['d_rsi'] = calculate_simple_rsi(df['d'])
    
    # 布林带风格特征
    def calculate_simple_bbands(series, window=20):
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        # 对numpy数组使用pandas方法需要转换
        return pd.Series(upper).bfill().values, pd.Series(middle).bfill().values, pd.Series(lower).bfill().values
    
    a_bb_upper, a_bb_middle, a_bb_lower = calculate_simple_bbands(df['a'])
    b_bb_upper, b_bb_middle, b_bb_lower = calculate_simple_bbands(df['b'])
    c_bb_upper, c_bb_middle, c_bb_lower = calculate_simple_bbands(df['c'])
    d_bb_upper, d_bb_middle, d_bb_lower = calculate_simple_bbands(df['d'])
    
    df['a_bb_position'] = np.where((a_bb_upper - a_bb_lower) != 0, 
                                   (df['a'] - a_bb_lower) / (a_bb_upper - a_bb_lower), 0.5)
    df['b_bb_position'] = np.where((b_bb_upper - b_bb_lower) != 0, 
                                   (df['b'] - b_bb_lower) / (b_bb_upper - b_bb_lower), 0.5)
    df['c_bb_position'] = np.where((c_bb_upper - c_bb_lower) != 0, 
                                   (df['c'] - c_bb_lower) / (c_bb_upper - c_bb_lower), 0.5)
    df['d_bb_position'] = np.where((d_bb_upper - d_bb_lower) != 0, 
                                   (df['d'] - d_bb_lower) / (d_bb_upper - d_bb_lower), 0.5)
    
    return df