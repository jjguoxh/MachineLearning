import os
import numpy as np
import pandas as pd
from collections import Counter
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from model import TransformerClassifier
# 移除了对 generate_label_with_reversals 的导入
from feature_engineering import add_features

# 配置参数
SEQ_LEN = 300          # 序列长度，比如300秒
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
PATIENCE = 5
MODEL_PATH = "best_longtrend_model.pth"
TRAINED_FILES_LOG = "trained_files.log"  # 记录已训练文件的日志文件

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_sequences(features, labels, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X), np.array(y)

def load_trained_files():
    """加载已训练文件列表"""
    if os.path.exists(TRAINED_FILES_LOG):
        with open(TRAINED_FILES_LOG, 'r') as f:
            return set(line.strip() for line in f.readlines())
    return set()

def save_trained_file(filename):
    """记录已训练的文件"""
    with open(TRAINED_FILES_LOG, 'a') as f:
        f.write(filename + '\n')

def load_all_data(data_dir="E:/SnipingTactics/label"):
    """从指定目录加载所有带标签的CSV文件"""
    # 查找所有符合命名规则的CSV文件
    csv_pattern = os.path.join(data_dir, "[0-9][0-9][0-9][0-9][0-9][0-9].csv")
    csv_files = glob(csv_pattern)
    
    if not csv_files:
        # 如果没有找到带_reversal_labels的文件，尝试查找所有CSV文件
        csv_pattern = os.path.join(data_dir, "[0-9][0-9][0-9][0-9][0-9][0-9].csv")
        csv_files = glob(csv_pattern)
        
    if not csv_files:
        raise FileNotFoundError(f"在目录 {data_dir} 中未找到符合命名规则的CSV文件")
    
    # 按文件名排序，确保按日期顺序处理
    csv_files.sort()
    
    # 加载已训练文件列表
    trained_files = load_trained_files()
    
    all_data = []
    processed_files = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        # 检查是否已经训练过该文件
        if filename in trained_files:
            print(f"跳过已训练文件: {filename}")
            continue
            
        try:
            print(f"正在加载文件: {filename}")
            df = pd.read_csv(csv_file)
            all_data.append(df)
            processed_files.append(filename)
            print(f"加载完成: {filename} (数据条数: {len(df)})")
        except Exception as e:
            print(f"加载文件 {filename} 时出错: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("没有找到新的未训练文件")
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"总共加载 {len(processed_files)} 个新文件，合并后数据条数: {len(combined_df)}")
    
    return combined_df, processed_files

def train():
    # 1. 读取所有数据文件
    try:
        df, processed_files = load_all_data()
        print(f"数据总条数: {len(df)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载错误: {e}")
        return

    # 2. 特征工程
    df = add_features(df)

    # 3. 准备特征和标签
    exclude_cols = ['label', 'index_value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    features = df[feature_cols].values
    labels = df['label'].values

    # 新增：将标签从 {-1, 0, 1} 映射到 {0, 1, 2}
    # -1 (下跌) -> 0
    # 0 (未知) -> 1  
    # 1 (上涨) -> 2
    label_mapping = {-1: 0, 0: 1, 1: 2}
    labels = np.array([label_mapping[label] for label in labels])
    
    print(f"原始标签值分布: {Counter(df['label'].values)}")
    print(f"映射后标签值分布: {Counter(labels)}")

    # 4. 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 5. 构造序列数据
    X, y = create_sequences(features, labels, SEQ_LEN)
    
    # 如果序列数据中也需要映射标签，则上面的映射已经处理了这个问题

    # 6. 划分训练/验证集（时间序列不打乱）
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print("训练集标签分布:", Counter(y_train))
    print("验证集标签分布:", Counter(y_val))

    # 7. 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # 8. 转torch张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    # 9. 模型初始化
    input_dim = X_train.shape[2]
    num_classes = len(classes)  # 现在应该是3类 (0, 1, 2)
    model = TransformerClassifier(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=2, num_classes=num_classes).to(DEVICE)

    # ... 其余代码保持不变
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for i in range(0, len(X_train_t), BATCH_SIZE):
            xb = X_train_t[i:i + BATCH_SIZE]
            yb = y_train_t[i:i + BATCH_SIZE]

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
            preds_val = torch.argmax(outputs_val, dim=1)
            val_acc = (preds_val == y_val_t).float().mean().item()

        print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("保存最佳模型权重")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("早停，训练结束")
                break

    # 10. 验证集完整性能报告
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val_t)
        preds_val = torch.argmax(outputs_val, dim=1).cpu().numpy()
        y_val_np = y_val_t.cpu().numpy()

    print("\nTransformer模型 验证集分类报告:")
    print(classification_report(y_val_np, preds_val))
    print("混淆矩阵:")
    print(confusion_matrix(y_val_np, preds_val))

    # 11. 随机森林基线模型训练和评估
    X_train_rf = X_train.reshape(len(X_train), -1)
    X_val_rf = X_val.reshape(len(X_val), -1)
    y_train_rf = y_train
    y_val_rf = y_val

    print("\n训练随机森林基线模型...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    rf_preds = rf.predict(X_val_rf)

    print("\n随机森林基线模型 验证集分类报告:")
    print(classification_report(y_val_rf, rf_preds))
    print("混淆矩阵:")
    print(confusion_matrix(y_val_rf, rf_preds))
    
    # 12. 记录已训练的文件
    for filename in processed_files:
        save_trained_file(filename)
    print(f"已将 {len(processed_files)} 个文件标记为已训练")


if __name__ == "__main__":
    train()