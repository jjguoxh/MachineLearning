# train_longtrend_incremental.py
import os
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

from model import TransformerClassifier
from feature_engineering import add_features

# ---------------- 配置 ----------------
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
PATIENCE = 5
MODEL_PATH = "../model/best_longtrend_model.pth"
TRAINED_FILES_LOG = "../model/trained_files.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Focal Loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets].to(inputs.device)
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------------- 数据加载 ----------------
def load_trained_files():
    if os.path.exists(TRAINED_FILES_LOG):
        with open(TRAINED_FILES_LOG, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_trained_file(filename):
    with open(TRAINED_FILES_LOG, 'a') as f:
        f.write(filename + '\n')

def load_all_data(data_dir="../label"):
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{data_dir} 下没有 CSV 文件")
    csv_files.sort()
    trained_files = load_trained_files()
    all_data = []
    processed_files = []
    for f in csv_files:
        filename = os.path.basename(f)
        if filename in trained_files:
            continue
        try:
            df = pd.read_csv(f)
            all_data.append(df)
            processed_files.append(filename)
        except:
            continue
    if not all_data:
        raise ValueError("没有新的未训练文件")
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df, processed_files

# ---------------- 序列构造 ----------------
def create_sequences_np(features, labels, seq_len=SEQ_LEN):
    n_samples = len(features) - seq_len
    if n_samples <= 0:
        return np.array([]), np.array([])
    idx = np.arange(n_samples)[:,None] + np.arange(seq_len)
    X = features[idx]  # shape=(n_samples, seq_len, feature_dim)
    y = labels[seq_len:]
    return X, y

# ---------------- 训练函数 ----------------
def train():
    # 1. 加载数据
    df, processed_files = load_all_data()
    df = add_features(df)
    exclude_cols = ['label','index_value']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    features = df[feature_cols].values
    labels = df['label'].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X, y = create_sequences_np(features_scaled, labels, SEQ_LEN)
    if len(X) == 0:
        print("数据太少，无法训练")
        return

    print(f"序列总数: {len(X)}, 特征维度: {X.shape[2]}")
    print(f"标签分布: {Counter(y)}")

    # 2. 处理类别不平衡
    X_flat = X.reshape(len(X), -1)
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_flat, y)
    X_res = X_res.reshape(-1, SEQ_LEN, X.shape[2])
    print(f"重采样后序列总数: {len(X_res)}, 标签分布: {Counter(y_res)}")

    # 3. 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # 4. 转 torch 张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    # 5. 模型初始化
    input_dim = X_train.shape[2]
    num_classes = len(np.unique(y))
    model = TransformerClassifier(
        input_dim=input_dim, model_dim=64, num_heads=4,
        num_layers=2, num_classes=num_classes, dropout=0.1
    ).to(DEVICE)

    # 6. 损失函数 & 优化器
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 7. 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    n_batches = int(np.ceil(len(X_train_t)/BATCH_SIZE))
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train_t))
        train_loss = 0
        for i in range(0, len(X_train_t), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        avg_train_loss = train_loss / len(X_train_t)

        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_t)
            val_loss = criterion(outputs_val, y_val_t).item()
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} TrainLoss={avg_train_loss:.4f} ValLoss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("早停触发，训练结束")
                break

    # 8. 标记已训练文件
    for f in processed_files:
        save_trained_file(f)
    print(f"已标记 {len(processed_files)} 个文件为已训练")

if __name__ == "__main__":
    train()
