import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from model import TransformerClassifier

SEQ_LEN = 60
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4

# 读取数据
df = pd.read_csv("data_with_labels.csv")
features = df[['a','b','c','d']].values
labels = df['label'].values

# 标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 生成样本序列
X, y = [], []
for i in range(len(features) - SEQ_LEN):
    X.append(features[i:i+SEQ_LEN])
    y.append(labels[i+SEQ_LEN])
X = np.array(X)
y = np.array(y)

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# 类别权重
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 张量转换
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)
class_weights = class_weights.to(device)

# 模型
model = TransformerClassifier(input_dim=4, model_dim=64, num_heads=4, num_layers=2, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_train.size(0))
    sum_loss = 0
    for i in range(0, X_train.size(0), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    avg_loss = sum_loss / (len(X_train) / BATCH_SIZE)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# 验证
model.eval()
with torch.no_grad():
    outputs = model(X_val)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    print(classification_report(y_val.cpu().numpy(), preds))
    print(confusion_matrix(y_val.cpu().numpy(), preds))
