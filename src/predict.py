# -*- coding: utf-8 -*-
"""
基于训练模型的预测和交易信号可视化
- 加载训练好的模型
- 对新数据进行预测
- 使用matplotlib绘制价格曲线和交易信号
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from model import TransformerClassifier, MultiScaleTransformerClassifier
from feature_engineering import add_features

# 配置参数
SEQ_LEN = 30
MODEL_PATH = "../model/best_longtrend_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, input_dim, num_classes, use_multiscale=False, seq_lengths=[10, 30, 60]):
    """
    加载训练好的模型
    """
    if use_multiscale:
        model = MultiScaleTransformerClassifier(
            input_dim=input_dim,
            model_dim=128,
            num_heads=8,
            num_layers=4,
            num_classes=num_classes,
            seq_lengths=seq_lengths,
            dropout=0.1
        ).to(DEVICE)
    else:
        model = TransformerClassifier(
            input_dim=input_dim,
            model_dim=128,
            num_heads=8,
            num_layers=4,
            num_classes=num_classes,
            seq_len=SEQ_LEN,
            dropout=0.1
        ).to(DEVICE)
    
    # 安全加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    except RuntimeError as e:
        if "Missing key" in str(e) or "Unexpected key" in str(e):
            print("警告: 模型结构与权重不匹配，尝试自动修复...")
            # 尝试加载不严格匹配的权重
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise e
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        raise e
        
    model.eval()
    return model

def prepare_data(df):
    """
    准备预测数据
    """
    # 特征工程
    df = add_features(df)
    
    # 准备特征
    exclude_cols = ['label', 'index_value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features = df[feature_cols].values
    
    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, df['index_value'].values

def create_sequences_for_prediction(features, seq_len=SEQ_LEN):
    """
    为预测创建序列数据
    """
    X = []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
    return np.array(X)

def create_multiscale_sequences_for_prediction(features, seq_lengths=[10, 30, 60]):
    """
    为多尺度模型创建序列数据
    """
    X_multi = {str(length): [] for length in seq_lengths}
    max_len = max(seq_lengths)
    
    for i in range(len(features) - max_len):
        for seq_len in seq_lengths:
            X_multi[str(seq_len)].append(features[i:i + seq_len])
    
    for seq_len in seq_lengths:
        X_multi[str(seq_len)] = np.array(X_multi[str(seq_len)])
    return X_multi

def predict(model, X, use_multiscale=False):
    """
    使用模型进行预测
    """
    with torch.no_grad():
        if use_multiscale:
            X_tensors = {}
            for key, value in X.items():
                X_tensors[key] = torch.tensor(value, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensors)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensor)
        
        # 获取预测结果
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds, probs.cpu().numpy()

def plot_trading_signals(index_values, predictions, use_multiscale=False):
    """
    绘制价格曲线和交易信号
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制价格曲线
    ax.plot(index_values, label='Price', color='blue', linewidth=1)
    
    # 根据是否使用多尺度模型调整索引
    start_idx = 60 if use_multiscale else SEQ_LEN  # 多尺度使用最大序列长度
    
    # 绘制交易信号
    long_entry_points = []   # 做多开仓点 (label=1)
    long_exit_points = []    # 做多平仓点 (label=2)
    short_entry_points = []  # 做空开仓点 (label=3)
    short_exit_points = []   # 做空平仓点 (label=4)
    
    for i, pred in enumerate(predictions):
        idx = i + start_idx
        if idx >= len(index_values):
            break
            
        if pred == 1:  # 做多开仓
            long_entry_points.append((idx, index_values[idx]))
        elif pred == 2:  # 做多平仓
            long_exit_points.append((idx, index_values[idx]))
        elif pred == 3:  # 做空开仓
            short_entry_points.append((idx, index_values[idx]))
        elif pred == 4:  # 做空平仓
            short_exit_points.append((idx, index_values[idx]))
    
    # 绘制交易信号箭头
    if long_entry_points:
        x, y = zip(*long_entry_points)
        ax.scatter(x, y, color='red', marker='^', s=100, label='Long Entry', zorder=5)
    
    if long_exit_points:
        x, y = zip(*long_exit_points)
        ax.scatter(x, y, color='red', marker='v', s=100, label='Long Exit', zorder=5)
    
    if short_entry_points:
        x, y = zip(*short_entry_points)
        ax.scatter(x, y, color='green', marker='^', s=100, label='Short Entry', zorder=5)
    
    if short_exit_points:
        x, y = zip(*short_exit_points)
        ax.scatter(x, y, color='green', marker='v', s=100, label='Short Exit', zorder=5)
    
    # 添加图例
    legend_elements = [
        Patch(facecolor='blue', label='Price'),
        Patch(facecolor='red', label='Long Entry (^) / Exit (v)'),
        Patch(facecolor='green', label='Short Entry (^) / Exit (v)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # 设置标题和标签
    ax.set_title('Price Chart with Trading Signals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表
    plt.show()
    
    # 打印交易信号统计
    print(f"交易信号统计:")
    print(f"  做多开仓信号: {len(long_entry_points)} 个")
    print(f"  做多平仓信号: {len(long_exit_points)} 个")
    print(f"  做空开仓信号: {len(short_entry_points)} 个")
    print(f"  做空平仓信号: {len(short_exit_points)} 个")

def detect_model_type(model_path):
    """
    尝试检测模型类型（单尺度或多尺度）
    """
    try:
        # 尝试加载模型状态字典
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # 检查是否存在多尺度特有的键
        multiscale_keys = [key for key in state_dict.keys() if 'input_fcs' in key or 'transformer_encoders' in key]
        if multiscale_keys:
            print("检测到多尺度模型权重")
            return True
        else:
            print("检测到单尺度模型权重")
            return False
    except Exception as e:
        print(f"检测模型类型时出错: {e}")
        # 默认返回单尺度
        return False

def main(data_file, use_multiscale=None):
    """
    主函数
    """
    print("开始预测和可视化...")
    
    # 读取数据
    print(f"读取数据文件: {data_file}")
    df = pd.read_csv(data_file)
    print(f"数据条数: {len(df)}")
    
    # 准备数据
    print("准备预测数据...")
    features, index_values = prepare_data(df)
    print(f"特征维度: {features.shape}")
    
    # 如果未指定模型类型，则尝试自动检测
    if use_multiscale is None:
        print("自动检测模型类型...")
        use_multiscale = detect_model_type(MODEL_PATH)
    
    # 创建序列
    print("创建序列数据...")
    if use_multiscale:
        X = create_multiscale_sequences_for_prediction(features, seq_lengths=[10, 30, 60])
        num_classes = 5  # 0:无操作, 1:做多开仓, 2:做多平仓, 3:做空开仓, 4:做空平仓
        print("使用多尺度模型")
    else:
        X = create_sequences_for_prediction(features, seq_len=SEQ_LEN)
        num_classes = 5  # 0:无操作, 1:做多开仓, 2:做多平仓, 3:做空开仓, 4:做空平仓
        print("使用单尺度模型")
    
    if len(X) == 0:
        print("序列数据为空，无法进行预测")
        return
    
    print(f"序列数量: {len(X)}")
    
    # 加载模型
    print("加载模型...")
    try:
        model = load_model(MODEL_PATH, features.shape[1], num_classes, use_multiscale, [10, 30, 60] if use_multiscale else None)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 进行预测
    print("进行预测...")
    predictions, probabilities = predict(model, X, use_multiscale)
    print(f"预测完成，共 {len(predictions)} 个预测结果")
    
    # 可视化结果
    print("绘制图表...")
    plot_trading_signals(index_values, predictions, use_multiscale)
    
    # 打印预测分布
    unique, counts = np.unique(predictions, return_counts=True)
    label_names = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
    print("\n预测结果分布:")
    for label, count in zip(unique, counts):
        print(f"  {label_names[label]} ({label}): {count} 个")
    
    print("预测和可视化完成!")

if __name__ == "__main__":
    # 检查命令行参数
    data_file = "250617.csv"
    use_multiscale = True  # 显式设置为多尺度
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        sys.exit(1)
    
    main(data_file, use_multiscale)