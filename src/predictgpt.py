# predict_longtrend.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from glob import glob
from model import TransformerClassifier
from feature_engineering import add_features
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../model/best_longtrend_model.pth"
SEQ_LEN = 30
DATA_DIR = "../predict"

# 1. 加载所有 CSV 文件
def load_data(data_dir=DATA_DIR):
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"目录 {data_dir} 下没有 CSV 文件")
    dfs = []
    for f in sorted(csv_files):
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)

# 2. 构造序列
def create_sequences(features, seq_len=SEQ_LEN):
    X = []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
    return np.array(X)

# 3. 加载数据并特征工程
df = load_data()
df = add_features(df)
exclude_cols = ['label','index_value']
feature_cols = [c for c in df.columns if c not in exclude_cols]
features = df[feature_cols].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X = create_sequences(features_scaled)
X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)

# 4. 加载模型
input_dim = X.shape[2]
num_classes = 5  # 0:保持,1:开多,2:平多,3:开空,4:平空
model = TransformerClassifier(input_dim=input_dim, model_dim=128,
                              num_heads=8, num_layers=4, num_classes=num_classes,
                              dropout=0.1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 5. 预测
with torch.no_grad():
    outputs = model(X_t)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1).cpu().numpy()

# 6. 绘图显示
plt.figure(figsize=(16,6))
plt.plot(df['index_value'].values[-len(preds):], label="指数走势")

# 7. 标记开平仓点
action_counts = {1:0,2:0,3:0,4:0}  # 开多,平多,开空,平空
for i, p in enumerate(preds):
    if p in [1,2]:
        plt.arrow(i, df['index_value'].values[-len(preds)+i], 0, 0.5,
                  color='red', head_width=0.5, head_length=0.5)
        action_counts[p] += 1
    elif p in [3,4]:
        plt.arrow(i, df['index_value'].values[-len(preds)+i], 0, -0.5,
                  color='blue', head_width=0.5, head_length=0.5)
        action_counts[p] += 1

plt.title("预测开平仓点 (红=多单, 蓝=空单)")
plt.xlabel("时间步")
plt.ylabel("指数")
plt.legend()

# 8. 显示统计信息
print("预测开平仓动作数量:")
print(f"开多单 (1): {action_counts[1]}")
print(f"平多单 (2): {action_counts[2]}")
print(f"开空单 (3): {action_counts[3]}")
print(f"平空单 (4): {action_counts[4]}")

plt.show()
