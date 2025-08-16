# data_preprocessing.py
# 读取 CSV，标准化，并构造滑动窗口特征

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from config import DATA_CSV, SEQ_LEN, SCALER_SAVE_PATH

REQUIRED_COLS = ["a", "b", "c", "d", "index_value"]


def load_raw(csv_path=DATA_CSV):
    df = pd.read_csv(csv_path)
    # 要求 csv 按时间顺序，从上到下
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺失列: {missing}")
    df = df.reset_index(drop=True)
    return df


def fit_and_transform_scaler(df, feature_cols=["a","b","c","d","index_value"]):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols].values)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    return df_scaled, scaler


def load_scaler(path=SCALER_SAVE_PATH):
    return joblib.load(path)


def build_sequences(df_scaled, seq_len=SEQ_LEN):
    """
    输入：已标准化的 df（包含 a,b,c,d,index_value）
    输出：X (N, seq_len, 4), index_positions (对应原始第 t 的位置，表示用 t 作为基点预测未来 horizon)
    注意：本函数不做标签，标签由 label_generation.py 生成
    """
    feats = df_scaled[["a","b","c","d"]].values.astype(np.float32)
    N = len(df_scaled)
    X = []
    idx_pos = []
    for t in range(0, N - seq_len):
        X.append(feats[t:t+seq_len])
        idx_pos.append(t+seq_len-1)  # 用窗口结束位置作为基点
    X = np.array(X)
    return X, np.array(idx_pos)


if __name__ == "__main__":
    df = load_raw()
    df_scaled, scaler = fit_and_transform_scaler(df)
    X, idx = build_sequences(df_scaled)
    print("Loaded", len(df), "rows -> sequences:", X.shape)