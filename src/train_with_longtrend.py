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
from feature_engineering import add_features
import glob
import os

# Optimized parameters according to project specifications
SEQ_LEN = 15  # Reduced from 300 to 15 for faster response
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
PATIENCE = 5
MODEL_PATH = "./model/best_longtrend_model.pth"

# Ensure model directory exists
os.makedirs("./model", exist_ok=True)

# 1. Load data with relaxed labels (priority order)
data_dirs = [
    "./data_with_relaxed_labels/",   # Highest priority
    "./data_with_improved_labels/",  # Second priority
    "./data_with_complete_labels/",  # Third priority
    "./data/"                        # Lowest priority
]

selected_dir = None
for data_dir in data_dirs:
    if os.path.exists(data_dir):
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if csv_files:
            selected_dir = data_dir
            print(f"Using data directory: {data_dir} ({len(csv_files)} files)")
            break

if not selected_dir:
    raise FileNotFoundError("No valid data directory found. Please run generate_complete_trading_labels.py first.")

# Combine all CSV files
all_dfs = []
for csv_file in csv_files:
    df_temp = pd.read_csv(csv_file)
    if 'label' in df_temp.columns:
        all_dfs.append(df_temp)
        print(f"Loaded {csv_file}: {len(df_temp)} rows")

if not all_dfs:
    raise ValueError("No CSV files with 'label' column found.")

df = pd.concat(all_dfs, ignore_index=True)
print(f"Combined dataset: {len(df)} rows")

# 2. Apply feature engineering (must generate 88 features)
print("Applying feature engineering...")
df = add_features(df)
print(f"After feature engineering: {df.shape}")

# 3. Prepare feature columns and labels (5-class classification)
exclude_cols = ['label', 'index_value']
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"Feature count: {len(feature_cols)} (expected: 88)")
if len(feature_cols) != 88:
    print(f"Warning: Feature count mismatch! Expected 88, got {len(feature_cols)}")

features = df[feature_cols].values
labels = df['label'].values

# Check label distribution
label_counts = pd.Series(labels).value_counts().sort_index()
print(f"Label distribution: {dict(label_counts)}")

# 4. 标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 5. 构造序列样本
X, y = [], []
for i in range(len(features) - SEQ_LEN):
    X.append(features[i:i + SEQ_LEN])
    y.append(labels[i + SEQ_LEN])
X = np.array(X)
y = np.array(y)

# 6. 划分训练验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# 7. 计算类别权重
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 8. 转张量并转device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)
class_weights = class_weights.to(device)

# 8. Initialize model for 5-class classification
input_dim = X_train.shape[2]
model = TransformerClassifier(
    input_dim=input_dim, 
    model_dim=128,  # Increased for better performance
    num_heads=8, 
    num_layers=4, 
    num_classes=5,  # 5-class: 0=no_action, 1=long_entry, 2=long_exit, 3=short_entry, 4=short_exit
    dropout=0.1
).to(device)

print(f"Model initialized with input_dim={input_dim}, num_classes=5")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 10. 训练 + 早停
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

    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        val_loss = criterion(outputs, y_val).item()
        preds = torch.argmax(outputs, dim=1)
        val_acc = (preds == y_val).float().mean().item()

    print(f"Epoch {epoch + 1}/{EPOCHS} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

# 11. Load best model and evaluate
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
with torch.no_grad():
    outputs = model(X_val)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    y_true = y_val.cpu().numpy()

print("\n=== Validation Set Classification Report ===")
print(classification_report(y_true, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, preds))

# Calculate signal density
signal_count = np.sum(preds != 0)
signal_density = signal_count / len(preds)
print(f"\nSignal density: {signal_density:.4f} ({signal_density*100:.2f}%)")

print(f"\nModel saved to: {MODEL_PATH}")
print("Training completed successfully!")
