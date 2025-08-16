# utils.py
# 公共工具：构造训练数据、映射 index -> 原始价格/位置、评估

import numpy as np
from sklearn.model_selection import train_test_split
from config import VAL_RATIO, RANDOM_SEED, SEQ_LEN


def make_xy(df_scaled, df_raw_index, seq_len=SEQ_LEN):
    # df_scaled 包含 标准化后的 a,b,c,d,index_value
    X = []
    # 使用窗口结束位置作为基点
    for t in range(0, len(df_scaled) - seq_len - 0 + 1):
        X.append(df_scaled[['a','b','c','d']].values[t:t+seq_len])
    X = np.array(X)
    # 对齐 raw index: 基点位置为 seq_end = seq_len-1 ... 对应原始index位置
    idx_positions = np.arange(seq_len-1, seq_len-1 + len(X))
    return X, idx_positions


def split_train_val(X, y, val_ratio=VAL_RATIO):
    # 时间序列按时间顺序划分（不 shuffle）
    n = len(X)
    n_val = int(n * val_ratio)
    if n_val < 1:
        return X, None, y, None
    split = n - n_val
    X_train = X[:split]
    X_val = X[split:]
    y_train = y[:split]
    y_val = y[split:]
    return X_train, X_val, y_train, y_val