import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from model import TransformerClassifier
from feature_engineering import add_features, generate_labels

SEQ_LEN = 60
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-4
PATIENCE = 5
MODEL_PATH = "best_model.pth"

# 1. 读取数据
df = pd.read_csv("data.csv")

# 2. 特征工程
df = add_features(df)

# 3. 生成标签（二分类）
df = generate_labels(df, pred_horizon=60, label_threshold=0.003)

# 4. 准备特征和标签
exclude_cols = ['label', 'index_value']
feature_cols = [c for c in df.columns if c not in exclude_cols]

features = df[feature_cols].values
labels = df['label'].values

# 5. 标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 6. 构造序列样本
X, y = [], []
for i in range(len(features) - SEQ_LEN):
    X.append(features[i:i + SEQ_LEN])
    y.append(labels[i + SEQ_LEN])
X = np.array(X)
y = np.array(y)

# 7. 划分训练验证集（时间序列不shuffle）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# 8. 计算类别权重
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 9. 转张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)
class_weights = class_weights.to(device)

# 10. 模型
input_dim = X_train.shape[2]
model = TransformerClassifier(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=2, num_classes=2).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 11. 训练 + 早停
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_train.size(0))
    train_loss = 0
    for i in range(0, X_train.size(0), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / (len(X_train) / BATCH_SIZE)

    # 验证
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        val_loss = criterion(outputs, y_val).item()
        preds = torch.argmax(outputs, dim=1)
        val_acc = (preds == y_val).float().mean().item()

    print(f"Epoch {epoch + 1}/{EPOCHS} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    # 早停判断
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model.save(MODEL_PATH)
        print("Model saved.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

# 12. 加载最佳模型测试
model.load(MODEL_PATH, device)
model.eval()
with torch.no_grad():
    outputs = model(X_val)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    y_true = y_val.cpu().numpy()

print("\n=== Final Evaluation on Validation Set ===")
print(classification_report(y_true, preds))
print("Confusion Matrix:")
print(confusion_matrix(y_true, preds))

# 13. 保存预测结果到csv
result_df = pd.DataFrame({
    "true_label": y_true,
    "pred_label": preds
})
result_df.to_csv("val_predictions.csv", index=False)
print("Validation predictions saved to val_predictions.csv")
