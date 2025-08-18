# train_with_classweight_and_baseline.py
import os
import sys
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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import SelectKBest, f_classif

# 添加SMOTE导入
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler

from model import TransformerClassifier, MultiScaleTransformerClassifier
from feature_engineering import add_features

# 配置参数
SEQ_LEN = 5            # 序列长度改为5
BATCH_SIZE = 8         # 批量大小
EPOCHS = 50            # 训练轮数
LR = 1e-5              # 学习率
PATIENCE = 10          # 早停耐心
MODEL_PATH = "../model/best_longtrend_model.pth"
TRAINED_FILES_LOG = "../model/trained_files.log"  # 记录已训练文件的日志文件

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... existing code ...
# Focal Loss实现
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
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.tensor([self.alpha, 1-self.alpha]).to(inputs.device)
            else:
                alpha_t = self.alpha.to(inputs.device)
            alpha_t = alpha_t[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_sequences(features, labels, seq_len=SEQ_LEN, max_sequences=50000):
    X, y = [], []
    total_possible = len(features) - seq_len
    
    # 收集各类别样本
    class_indices = {}
    for i in range(total_possible):
        label = labels[i + seq_len]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    print(f"各类别样本数: { {k: len(v) for k, v in class_indices.items()} }")
    
    # 更合理的平衡采样策略
    target_samples_per_class = 800  # 每个类别目标样本数
    
    selected_indices = []
    for class_label, indices in class_indices.items():
        if len(indices) > target_samples_per_class:
            # 欠采样多数类
            selected = np.random.choice(indices, target_samples_per_class, replace=False)
        elif len(indices) > 0:
            # 过采样少数类，但设置合理上限
            repeat_times = max(1, target_samples_per_class // len(indices))
            # 限制重复次数，避免过度过采样
            repeat_times = min(repeat_times, 10)
            selected = np.tile(indices, repeat_times)
            # 如果仍不足目标数量，随机补充
            if len(selected) < target_samples_per_class:
                additional = np.random.choice(indices, target_samples_per_class - len(selected), replace=True)
                selected = np.concatenate([selected, additional])
            # 如果超过目标数量，随机选择
            if len(selected) > target_samples_per_class:
                selected = np.random.choice(selected, target_samples_per_class, replace=False)
        else:
            continue
        selected_indices.extend(selected)
    
    # 打乱顺序
    np.random.shuffle(selected_indices)
    
    # 限制总样本数
    if len(selected_indices) > max_sequences:
        selected_indices = np.random.choice(selected_indices, max_sequences, replace=False)
        print(f"总样本数限制为: {len(selected_indices)}")
    
    # 创建序列
    for i in selected_indices:
        X.append(features[i:i + seq_len])
        y.append(labels[i + seq_len])
        
        # 添加进度提示
        if len(X) % 1000 == 0 and len(X) > 0:
            print(f"已创建 {len(X)} 个序列")
            
    return np.array(X), np.array(y)

# ... existing code ...

def create_multiscale_sequences(features, labels, seq_lengths=[5, 10, 15], max_sequences=50000):
    """
    创建多尺度序列数据（调整为更适合的尺度）
    """
    X_multi = {length: [] for length in seq_lengths}
    y_multi = []
    
    total_possible = len(features) - max(seq_lengths)
    
    # 收集各类别样本
    class_indices = {}
    for i in range(total_possible):
        label = labels[i + max(seq_lengths)]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    print(f"各类别样本数: { {k: len(v) for k, v in class_indices.items()} }")
    
    # 平衡采样策略
    target_samples_per_class = 800  # 每个类别目标样本数
    
    selected_indices = []
    for class_label, indices in class_indices.items():
        if len(indices) > target_samples_per_class:
            # 欠采样多数类
            selected = np.random.choice(indices, target_samples_per_class, replace=False)
        elif len(indices) > 0:
            # 过采样少数类，但设置合理上限
            repeat_times = max(1, target_samples_per_class // len(indices))
            # 限制重复次数，避免过度过采样
            repeat_times = min(repeat_times, 10)
            selected = np.tile(indices, repeat_times)
            # 如果仍不足目标数量，随机补充
            if len(selected) < target_samples_per_class:
                additional = np.random.choice(indices, target_samples_per_class - len(selected), replace=True)
                selected = np.concatenate([selected, additional])
            # 如果超过目标数量，随机选择
            if len(selected) > target_samples_per_class:
                selected = np.random.choice(selected, target_samples_per_class, replace=False)
        else:
            continue
        selected_indices.extend(selected)
    
    # 打乱顺序
    np.random.shuffle(selected_indices)
    
    # 限制总样本数
    if len(selected_indices) > max_sequences:
        selected_indices = np.random.choice(selected_indices, max_sequences, replace=False)
        print(f"总样本数限制为: {len(selected_indices)}")
    
    # 创建多尺度序列
    for i in selected_indices:
        for seq_len in seq_lengths:
            X_multi[seq_len].append(features[i:i + seq_len])
        y_multi.append(labels[i + max(seq_lengths)])
        
        # 添加进度提示
        if len(y_multi) % 1000 == 0 and len(y_multi) > 0:
            print(f"已创建 {len(y_multi)} 个序列")
            
    # 转换为numpy数组
    for seq_len in seq_lengths:
        X_multi[seq_len] = np.array(X_multi[seq_len])
    return X_multi, np.array(y_multi)


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

def load_all_data(data_dir="../label"):
    """从指定目录加载所有带标签的CSV文件"""
    # 查找所有符合命名规则的CSV文件
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
    
    # 限制数据量以防止内存不足
    MAX_SAMPLES = 30000  # 进一步限制最大样本数
    if len(combined_df) > MAX_SAMPLES:
        print(f"数据量过大，随机采样 {MAX_SAMPLES} 条数据")
        combined_df = combined_df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"采样后数据条数: {len(combined_df)}")
    
    return combined_df, processed_files

def convert_to_binary_labels(labels):
    """
    将三分类标签转换为二分类标签
    原始标签: -1(下跌)->0, 0(未知)->忽略, 1(上涨)->1
    """
    # 创建掩码，过滤掉未知状态(标签为1的原始标签)
    mask = labels != 1  # 过滤掉原始标签为0的"未知"状态（映射后为1）
    binary_labels = labels[mask]
    # 将上涨(原始标签1，映射后为2)转换为1，下跌(原始标签-1，映射后为0)转换为0
    binary_labels = np.where(binary_labels == 2, 1, 0)
    return mask, binary_labels

def filter_sequences_for_binary(X, y):
    """
    过滤序列数据，只保留上涨和下跌的样本
    """
    mask, binary_y = convert_to_binary_labels(y)
    binary_X = X[mask]
    return binary_X, binary_y

def adjust_prediction_threshold(model, X, threshold=0.5):
    """
    调整预测阈值以优化上涨趋势识别率
    """
    if hasattr(model, "predict_proba"):
        # 对于有predict_proba方法的模型（如随机森林）
        probs = model.predict_proba(X)
        # 根据阈值调整预测结果，提高上涨类别召回率
        preds = (probs[:, 1] >= threshold).astype(int)
        return preds, probs
    else:
        # 对于其他模型使用默认预测
        return model.predict(X), None

def improved_random_forest_baseline(X_train_rf, y_train_rf, X_val_rf, y_val_rf):
    """
    改进的随机森林基线模型（二分类版本，优化开仓点识别）
    """
    print("\n训练改进的随机森林基线模型（二分类，优化开仓点识别）...")
    
    # 限制随机森林训练数据量以防止内存不足
    MAX_RF_SAMPLES = 5000
    if len(X_train_rf) > MAX_RF_SAMPLES:
        print(f"随机森林训练数据量过大，随机采样 {MAX_RF_SAMPLES} 条数据")
        indices = np.random.choice(len(X_train_rf), MAX_RF_SAMPLES, replace=False)
        X_train_rf_sample = X_train_rf[indices]
        y_train_rf_sample = y_train_rf[indices]
    else:
        X_train_rf_sample = X_train_rf
        y_train_rf_sample = y_train_rf
    
    # 添加特征选择
    print("进行特征选择...")
    try:
        n_features = min(300, X_train_rf_sample.shape[1])
        feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_rf_selected = feature_selector.fit_transform(X_train_rf_sample, y_train_rf_sample)
        X_val_rf_selected = feature_selector.transform(X_val_rf)
        print(f"特征选择后维度: {X_train_rf_selected.shape[1]}")
    except Exception as e:
        print(f"特征选择失败: {e}，使用原始特征")
        X_train_rf_selected = X_train_rf_sample
        X_val_rf_selected = X_val_rf
    
    # 处理类别不平衡
    print("应用重采样处理类别不平衡...")
    try:
        # 使用SMOTE或其他重采样方法
        sampling_strategy = 'auto'  # 或者自定义策略
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_train_balanced, y_train_balanced = ros.fit_resample(X_train_rf_selected, y_train_rf_sample)
        print(f"重采样处理后样本数: {len(X_train_balanced)}")
        print(f"重采样后标签分布: {Counter(y_train_balanced)}")
    except Exception as e:
        print(f"重采样处理失败: {e}，使用原始数据训练")
        X_train_balanced, y_train_balanced = X_train_rf_selected, y_train_rf_sample
    
    # 使用优化开仓点识别的随机森林参数
    rf = RandomForestClassifier(
        n_estimators=150,
        class_weight='balanced',  # 使用平衡权重
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.4,
        random_state=42,
        n_jobs=1
    )
    
    print("开始训练随机森林...")
    rf.fit(X_train_balanced, y_train_balanced)
    print("随机森林训练完成")
    
    # 使用优化阈值进行预测
    print("使用优化阈值进行预测...")
    OPTIMAL_THRESHOLD = 0.5  # 可以根据验证集调整
    rf_preds, rf_probs = adjust_prediction_threshold(rf, X_val_rf_selected, OPTIMAL_THRESHOLD)
    
    print(f"\n改进的随机森林基线模型 验证集分类报告 (阈值={OPTIMAL_THRESHOLD}):")
    print(classification_report(y_val_rf, rf_preds, target_names=['非开仓点', '开仓点']))
    print("混淆矩阵:")
    print(confusion_matrix(y_val_rf, rf_preds))
    
    # 分析不同阈值的效果
    print("\n不同阈值下的性能分析:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        preds, _ = adjust_prediction_threshold(rf, X_val_rf_selected, threshold)
        report = classification_report(y_val_rf, preds, output_dict=True, target_names=['非开仓点', '开仓点'])
        print(f"阈值 {threshold}: 开仓点召回率={report['开仓点']['recall']:.3f}, 开仓点精确率={report['开仓点']['precision']:.3f}")
    
    # 输出特征重要性
    feature_importance = rf.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    print(f"\n前10个最重要的特征索引: {top_features}")
    print(f"对应的重要性分数: {feature_importance[top_features]}")
    
    return rf

# ... existing code ...

def train(use_multiscale=False):
    # 1. 读取所有数据文件
    try:
        df, processed_files = load_all_data()
        print(f"数据总条数: {len(df)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载错误: {e}")
        return

    # 2. 特征工程
    df = add_features(df)
    
    # 检查是否有无穷大值或NaN值
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    nan_count = 0
    for col in numeric_columns:
        inf_count += np.isinf(df[col]).sum()
        nan_count += np.isnan(df[col]).sum()
    
    print(f"特征工程后数据中无穷大值数量: {inf_count}")
    print(f"特征工程后数据中NaN值数量: {nan_count}")

    # 3. 准备特征和标签
    exclude_cols = ['label', 'index_value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    features = df[feature_cols].values
    labels = df['label'].values

    print(f"标签值分布: {Counter(labels)}")
    print(f"特征数量: {features.shape[1]}")
    print(f"特征矩阵形状: {features.shape}")
    
    # 检查特征中的无穷大值和NaN值
    inf_features = np.isinf(features).sum()
    nan_features = np.isnan(features).sum()
    print(f"特征矩阵中无穷大值数量: {inf_features}")
    print(f"特征矩阵中NaN值数量: {nan_features}")
    
    # 检查标签质量
    unique_labels = np.unique(labels)
    print(f"唯一标签值: {unique_labels}")
    
    if len(unique_labels) < 2:
        print("错误: 标签类别数不足，无法进行分类训练")
        return

    # 4. 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 检查标准化后的特征
    inf_scaled = np.isinf(features).sum()
    nan_scaled = np.isnan(features).sum()
    print(f"标准化后特征中无穷大值数量: {inf_scaled}")
    print(f"标准化后特征中NaN值数量: {nan_scaled}")
    
    if inf_scaled > 0 or nan_scaled > 0:
        print("警告: 标准化后的特征中仍存在无穷大值或NaN值，将进行替换处理")
        # 将NaN替换为0，将无穷大替换为有限值
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    if use_multiscale:
        # 5. 构造多尺度序列数据
        print("开始构造多尺度序列数据...")
        SEQ_LENGTHS = [5, 10, 15]  # 调整为更适合的尺度
        X_dict, y = create_multiscale_sequences(features, labels, seq_lengths=SEQ_LENGTHS, max_sequences=50000)
        
        total_sequences = len(y)
        print(f"创建的序列数量: {total_sequences}")
        if total_sequences > 0:
            print(f"特征维度: {features.shape[1]}")
        
        if total_sequences < 10:
            print("数据量不足，无法训练模型")
            return
            
        # 检查序列标签分布
        print("序列标签分布:", Counter(y))
        if len(np.unique(y)) < 2:
            print("错误: 序列中标签类别数不足，无法进行分类训练")
            return

        # 6. 划分训练/验证集
        split_idx = int(len(y) * 0.8)
        if split_idx == 0:
            print("数据量过少，无法划分训练集和验证集")
            return
            
        # 划分多尺度数据
        X_train_dict = {}
        X_val_dict = {}
        for length in SEQ_LENGTHS:
            X_train_dict[str(length)] = X_dict[length][:split_idx]
            X_val_dict[str(length)] = X_dict[length][split_idx:]
        
        y_train, y_val = y[:split_idx], y[split_idx:]

        print("训练集标签分布:", Counter(y_train.flatten()))
        print("验证集标签分布:", Counter(y_val.flatten()))

        # 7. 计算类别权重（处理类别不平衡）
        y_train_flat = y_train.flatten()
        classes = np.unique(y_train_flat)
        print(f"训练集中的类别: {classes}")

        if len(classes) > 1:
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_flat)
            weights_dict = dict(zip(classes, class_weights))
            
            # 适度增强少数类权重
            for cls in weights_dict:
                weights_dict[cls] = min(weights_dict[cls] * 2.0, 50.0)
            
            print(f"计算得到的类别权重: {weights_dict}")
            
            weights_list = [weights_dict[cls] for cls in sorted(weights_dict.keys())]
            weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)
        else:
            print(f"警告: 训练数据中只包含一个类别 ({classes[0]})，将不使用类别权重")
            weights_tensor = None

        # 8. 转换为torch张量
        X_train_tensors = {}
        X_val_tensors = {}
        for length in SEQ_LENGTHS:
            X_train_tensors[str(length)] = torch.tensor(X_train_dict[str(length)], dtype=torch.float32).to(DEVICE)
            X_val_tensors[str(length)] = torch.tensor(X_val_dict[str(length)], dtype=torch.float32).to(DEVICE)
            
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

        # 9. 模型初始化（使用多尺度模型）
        input_dim = features.shape[1]
        num_classes = len(np.unique(np.concatenate([y_train.flatten(), y_val.flatten()])))
        print(f"总类别数: {num_classes}")
        
        model = MultiScaleTransformerClassifier(
            input_dim=input_dim, 
            model_dim=128,
            num_heads=8,
            num_layers=4,
            num_classes=num_classes,
            seq_lengths=SEQ_LENGTHS,
            dropout=0.1
        ).to(DEVICE)

        # 10. 训练模型
        if weights_tensor is not None:
            alpha_weights = weights_tensor / weights_tensor.sum()
            criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
        else:
            criterion = FocalLoss(gamma=2.0)
        
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            n_batches = 0
            
            # 创建批次索引
            indices = np.arange(len(y_train))
            np.random.shuffle(indices)
            
            for i in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[i:i + BATCH_SIZE]
                
                # 构造批次数据
                xb_dict = {}
                for length in SEQ_LENGTHS:
                    xb_dict[str(length)] = X_train_tensors[str(length)][batch_indices]
                yb = y_train_t[batch_indices]

                optimizer.zero_grad()
                outputs = model(xb_dict)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_indices)
                n_batches += len(batch_indices)
                
                if i % (BATCH_SIZE * 100) == 0:
                    print(f"  已处理 {i}/{len(indices)} 批次")

            avg_train_loss = train_loss / n_batches if n_batches > 0 else 0

            model.eval()
            with torch.no_grad():
                # 验证集评估
                val_outputs = model(X_val_tensors)
                val_loss = criterion(val_outputs, y_val_t).item()
                preds_val = torch.argmax(val_outputs, dim=1)
                val_acc = (preds_val == y_val_t).float().mean().item()

            print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

            scheduler.step(val_loss)

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

        # 11. 验证集完整性能报告
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            
            target_names = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
            
            with torch.no_grad():
                outputs_val = model(X_val_tensors)
                probs_val = torch.nn.functional.softmax(outputs_val, dim=1)
                preds_val = torch.argmax(outputs_val, dim=1).cpu().numpy()
                y_val_np = y_val_t.cpu().numpy()
                
            print("\n多尺度Transformer模型 验证集分类报告:")
            print(classification_report(y_val_np.flatten(), preds_val.flatten(), target_names=target_names))
            print("混淆矩阵:")
            cm = confusion_matrix(y_val_np.flatten(), preds_val.flatten())
            print(cm)
            
            print("\n各类别详细指标:")
            for i, name in enumerate(target_names):
                if len(cm) > i and np.sum(cm[i, :]) > 0:
                    precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
                    recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    support = np.sum(cm[i, :])
                    print(f"{name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
                elif len(cm) > i:
                    support = np.sum(cm[i, :])
                    print(f"{name}: Precision=0.0000, Recall=0.0000, F1=0.0000, Support={support}")
                else:
                    print(f"{name}: 无数据")
            
            print("\n各类别预测分布:")
            for i, name in enumerate(target_names):
                pred_count = np.sum(preds_val.flatten() == i)
                actual_count = np.sum(y_val_np.flatten() == i)
                print(f"{name}: 预测{pred_count}个, 实际{actual_count}个")
            
        except Exception as e:
            print(f"模型评估时出错: {e}")
            import traceback
            traceback.print_exc()

    else:
        # 使用原始的单尺度序列方法
        # 5. 构造序列数据
        print("开始构造序列数据...")
        X, y = create_sequences(features, labels, SEQ_LEN, max_sequences=30000)
        
        print(f"创建的序列数量: {len(X)}")
        if len(X) > 0:
            print(f"每序列长度: {len(X[0])}")
            print(f"特征维度: {len(X[0][0])}")
        
        if len(X) < 10:
            print("数据量不足，无法训练模型")
            return
            
        # 检查序列标签分布
        print("序列标签分布:", Counter(y))
        if len(np.unique(y)) < 2:
            print("错误: 序列中标签类别数不足，无法进行分类训练")
            return

        # 6. 划分训练/验证集
        split_idx = int(len(X) * 0.8)
        if split_idx == 0:
            print("数据量过少，无法划分训练集和验证集")
            return
            
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print("训练集标签分布:", Counter(y_train.flatten()))
        print("验证集标签分布:", Counter(y_val.flatten()))

        # 7. 计算类别权重（处理类别不平衡）- 更合理的策略
        y_train_flat = y_train.flatten()
        classes = np.unique(y_train_flat)
        print(f"训练集中的类别: {classes}")

        if len(classes) > 1:
            # 使用sklearn的balanced策略，但手动调整
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_flat)
            weights_dict = dict(zip(classes, class_weights))
            
            # 适度增强少数类权重，但不过度
            for cls in weights_dict:
                weights_dict[cls] = min(weights_dict[cls] * 2.0, 50.0)  # 适度增强，设置上限
            
            print(f"计算得到的类别权重: {weights_dict}")
            
            # 转换为tensor
            weights_list = [weights_dict[cls] for cls in sorted(weights_dict.keys())]
            weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)
        else:
            print(f"警告: 训练数据中只包含一个类别 ({classes[0]})，将不使用类别权重")
            weights_tensor = None

        # 8. 转torch张量
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

        # 9. 模型初始化
        input_dim = X_train.shape[2]
        num_classes = len(np.unique(np.concatenate([y_train.flatten(), y_val.flatten()])))
        print(f"总类别数: {num_classes}")
        
        model = TransformerClassifier(
            input_dim=input_dim, 
            model_dim=128,       # 增加模型维度
            num_heads=8,         # 增加注意力头数
            num_layers=4,        # 增加层数
            num_classes=num_classes,
            dropout=0.1
        ).to(DEVICE)

        # 10. 训练模型 - 使用适中的Focal Loss
        if weights_tensor is not None:
            # 使用alpha参数传递权重
            alpha_weights = weights_tensor / weights_tensor.sum()  # 归一化
            criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)  # 适中的gamma值
        else:
            criterion = FocalLoss(gamma=2.0)  # 适中的gamma值
        
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)  # 添加weight decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train_t), BATCH_SIZE):
                xb = X_train_t[i:i + BATCH_SIZE]
                yb = y_train_t[i:i + BATCH_SIZE]

                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xb.size(0)
                n_batches += xb.size(0)
                
                # 添加训练进度提示
                if i % (BATCH_SIZE * 100) == 0:
                    print(f"  已处理 {i}/{len(X_train_t)} 批次")

            avg_train_loss = train_loss / n_batches if n_batches > 0 else 0

            model.eval()
            with torch.no_grad():
                outputs_val = model(X_val_t)
                val_loss = criterion(outputs_val, y_val_t).item()
                preds_val = torch.argmax(outputs_val, dim=1)
                val_acc = (preds_val == y_val_t).float().mean().item()

            print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

            # 使用学习率调度器
            scheduler.step(val_loss)

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

        # 11. 验证集完整性能报告
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            
            # 类别名称
            target_names = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
            
            with torch.no_grad():
                outputs_val = model(X_val_t)
                probs_val = torch.nn.functional.softmax(outputs_val, dim=1)
                preds_val = torch.argmax(outputs_val, dim=1).cpu().numpy()
                y_val_np = y_val_t.cpu().numpy()
                
            print("\nTransformer模型 验证集分类报告:")
            print(classification_report(y_val_np.flatten(), preds_val.flatten(), target_names=target_names))
            print("混淆矩阵:")
            cm = confusion_matrix(y_val_np.flatten(), preds_val.flatten())
            print(cm)
            
            # 计算各类别的精确率、召回率、F1分数
            print("\n各类别详细指标:")
            for i, name in enumerate(target_names):
                if len(cm) > i and np.sum(cm[i, :]) > 0:  # 如果该类别在真实标签中存在
                    precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
                    recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    support = np.sum(cm[i, :])
                    print(f"{name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
                elif len(cm) > i:
                    support = np.sum(cm[i, :])
                    print(f"{name}: Precision=0.0000, Recall=0.0000, F1=0.0000, Support={support}")
                else:
                    print(f"{name}: 无数据")
            
            # 计算每个类别的预测分布
            print("\n各类别预测分布:")
            for i, name in enumerate(target_names):
                pred_count = np.sum(preds_val.flatten() == i)
                actual_count = np.sum(y_val_np.flatten() == i)
                print(f"{name}: 预测{pred_count}个, 实际{actual_count}个")
            
        except Exception as e:
            print(f"模型评估时出错: {e}")
            import traceback
            traceback.print_exc()

    # 12. 记录已训练的文件
    for filename in processed_files:
        save_trained_file(filename)
    print(f"已将 {len(processed_files)} 个文件标记为已训练")

# ... existing code ...
# ... existing code ...
if __name__ == "__main__":
    # train()
    # ... existing code ...
    import os
    # 检查命令行参数以决定是否使用多尺度方法
    use_multiscale = len(sys.argv) > 1 and sys.argv[1] == "--multiscale"
    print(f"使用{'多尺度' if use_multiscale else '单尺度'}训练方法")
    train(use_multiscale=use_multiscale)