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

def plot_trading_signals(index_values, predictions, probabilities, use_multiscale=False):
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
    
    # 存储交易信号及其置信度
    long_trades = []   # 存储做多交易 (开仓点, 平仓点, 置信度)
    short_trades = []  # 存储做空交易 (开仓点, 平仓点, 置信度)
    
    # 收集所有信号点及其置信度
    all_long_entries = []
    all_long_exits = []
    all_short_entries = []
    all_short_exits = []
    
    for i, pred in enumerate(predictions):
        idx = i + start_idx
        if idx >= len(index_values):
            break
            
        if pred == 1:  # 做多开仓
            confidence = probabilities[i][1]  # 开仓置信度
            all_long_entries.append((idx, index_values[idx], confidence, i))
        elif pred == 2:  # 做多平仓
            confidence = probabilities[i][2]  # 平仓置信度
            all_long_exits.append((idx, index_values[idx], confidence, i))
        elif pred == 3:  # 做空开仓
            confidence = probabilities[i][3]  # 开仓置信度
            all_short_entries.append((idx, index_values[idx], confidence, i))
        elif pred == 4:  # 做空平仓
            confidence = probabilities[i][4]  # 平仓置信度
            all_short_exits.append((idx, index_values[idx], confidence, i))
    
    # 配对交易信号并过滤掉会立即亏损的交易
    # 做多交易配对
    for entry in all_long_entries:
        entry_idx, entry_price, entry_conf, entry_i = entry
        # 寻找最近的平仓信号
        for exit in all_long_exits:
            exit_idx, exit_price, exit_conf, exit_i = exit
            if exit_idx > entry_idx:  # 平仓点必须在开仓点之后
                # 检查开仓后短期内是否立即向相反方向运行（避免立即大幅亏损）
                max_adverse_excursion = 0  # 最大不利波动
                adverse_threshold = 0.005  # 0.5%的不利波动阈值
                
                # 检查开仓后到平仓前的价格波动
                min_price_after_entry = entry_price
                for j in range(entry_idx+1, min(exit_idx, len(index_values))):
                    current_price = index_values[j]
                    if current_price < min_price_after_entry:
                        min_price_after_entry = current_price
                        max_adverse_excursion = (entry_price - min_price_after_entry) / entry_price
                        if max_adverse_excursion > adverse_threshold:
                            # 立即向相反方向运行超过阈值，跳过这笔交易
                            break
                
                # 如果没有立即大幅向相反方向运行
                if max_adverse_excursion <= adverse_threshold:
                    avg_confidence = (entry_conf + exit_conf) / 2
                    expected_profit = (exit_price - entry_price) / entry_price
                    long_trades.append({
                        'entry_idx': entry_idx,
                        'entry_price': entry_price,
                        'exit_idx': exit_idx,
                        'exit_price': exit_price,
                        'confidence': avg_confidence,
                        'expected_profit': expected_profit,
                        'max_adverse_excursion': max_adverse_excursion,
                        'entry_i': entry_i,
                        'exit_i': exit_i
                    })
                break  # 只取最近的一个平仓信号
    
    # 做空交易配对
    for entry in all_short_entries:
        entry_idx, entry_price, entry_conf, entry_i = entry
        # 寻找最近的平仓信号
        for exit in all_short_exits:
            exit_idx, exit_price, exit_conf, exit_i = exit
            if exit_idx > entry_idx:  # 平仓点必须在开仓点之后
                # 检查开仓后短期内是否立即向相反方向运行（避免立即大幅亏损）
                max_adverse_excursion = 0  # 最大不利波动
                adverse_threshold = 0.005  # 0.5%的不利波动阈值
                
                # 检查开仓后到平仓前的价格波动
                max_price_after_entry = entry_price
                for j in range(entry_idx+1, min(exit_idx, len(index_values))):
                    current_price = index_values[j]
                    if current_price > max_price_after_entry:
                        max_price_after_entry = current_price
                        max_adverse_excursion = (max_price_after_entry - entry_price) / entry_price
                        if max_adverse_excursion > adverse_threshold:
                            # 立即向相反方向运行超过阈值，跳过这笔交易
                            break
                
                # 如果没有立即大幅向相反方向运行
                if max_adverse_excursion <= adverse_threshold:
                    avg_confidence = (entry_conf + exit_conf) / 2
                    expected_profit = (entry_price - exit_price) / entry_price
                    short_trades.append({
                        'entry_idx': entry_idx,
                        'entry_price': entry_price,
                        'exit_idx': exit_idx,
                        'exit_price': exit_price,
                        'confidence': avg_confidence,
                        'expected_profit': expected_profit,
                        'max_adverse_excursion': max_adverse_excursion,
                        'entry_i': entry_i,
                        'exit_i': exit_i
                    })
                break  # 只取最近的一个平仓信号
    
    # 如果配对后的完整交易太少，则退而求其次，只显示高置信度的信号点（但也要满足不会立即大幅亏损）
    min_trades = 5
    top_trades = []
    
    if len(long_trades) + len(short_trades) < min_trades:
        print("完整交易对不足，显示高置信度且不会立即亏损的信号点...")
        # 收集所有信号点并过滤掉会立即亏损的信号
        all_signals = []
        
        # 添加做多开仓信号（过滤掉会立即亏损的）
        for entry in all_long_entries:
            entry_idx, entry_price, confidence, i = entry
            # 检查开仓后短期内是否立即向相反方向运行
            will_immediately_lose = False
            adverse_threshold = 0.005  # 0.5%的不利波动阈值
            
            # 检查开仓后几个点的价格变化
            look_ahead = min(10, len(index_values) - entry_idx - 1)  # 最多看10个点
            min_price_after_entry = entry_price
            for j in range(entry_idx+1, entry_idx+1+look_ahead):
                current_price = index_values[j]
                if current_price < min_price_after_entry:
                    min_price_after_entry = current_price
                    max_adverse_excursion = (entry_price - min_price_after_entry) / entry_price
                    if max_adverse_excursion > adverse_threshold:
                        will_immediately_lose = True
                        break
            
            if not will_immediately_lose:
                all_signals.append({
                    'type': 'long_entry',
                    'idx': entry_idx,
                    'price': entry_price,
                    'confidence': confidence,
                    'i': i
                })
            
        # 添加做空开仓信号（过滤掉会立即亏损的）
        for entry in all_short_entries:
            entry_idx, entry_price, confidence, i = entry
            # 检查开仓后短期内是否立即向相反方向运行
            will_immediately_lose = False
            adverse_threshold = 0.005  # 0.5%的不利波动阈值
            
            # 检查开仓后几个点的价格变化
            look_ahead = min(10, len(index_values) - entry_idx - 1)  # 最多看10个点
            max_price_after_entry = entry_price
            for j in range(entry_idx+1, entry_idx+1+look_ahead):
                current_price = index_values[j]
                if current_price > max_price_after_entry:
                    max_price_after_entry = current_price
                    max_adverse_excursion = (max_price_after_entry - entry_price) / entry_price
                    if max_adverse_excursion > adverse_threshold:
                        will_immediately_lose = True
                        break
            
            if not will_immediately_lose:
                all_signals.append({
                    'type': 'short_entry',
                    'idx': entry_idx,
                    'price': entry_price,
                    'confidence': confidence,
                    'i': i
                })
        
        # 添加平仓信号（不做过滤）
        for exit in all_long_exits:
            idx, price, confidence, i = exit
            all_signals.append({
                'type': 'long_exit',
                'idx': idx,
                'price': price,
                'confidence': confidence,
                'i': i
            })
            
        for exit in all_short_exits:
            idx, price, confidence, i = exit
            all_signals.append({
                'type': 'short_exit',
                'idx': idx,
                'price': price,
                'confidence': confidence,
                'i': i
            })
        
        # 按置信度排序并取前20个
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)
        top_signals = all_signals[:20]
        
        # 分类信号点
        for signal in top_signals:
            if signal['type'] == 'long_entry':
                long_entry_points.append((signal['idx'], signal['price']))
            elif signal['type'] == 'long_exit':
                long_exit_points.append((signal['idx'], signal['price']))
            elif signal['type'] == 'short_entry':
                short_entry_points.append((signal['idx'], signal['price']))
            elif signal['type'] == 'short_exit':
                short_exit_points.append((signal['idx'], signal['price']))
        
        print(f"显示 {len(top_signals)} 个高置信度且不会立即亏损的信号点")
    else:
        # 根据置信度排序并选择前10个最佳交易
        all_trades = long_trades + short_trades
        all_trades.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 只保留前10个最佳交易
        top_trades = all_trades[:10]
        
        # 提取用于绘制的点
        long_entry_points = []
        long_exit_points = []
        short_entry_points = []
        short_exit_points = []
        
        for trade in top_trades:
            if trade in long_trades:
                long_entry_points.append((trade['entry_idx'], trade['entry_price']))
                long_exit_points.append((trade['exit_idx'], trade['exit_price']))
            elif trade in short_trades:
                short_entry_points.append((trade['entry_idx'], trade['entry_price']))
                short_exit_points.append((trade['exit_idx'], trade['exit_price']))
        
        print(f"显示 {len(top_trades)} 个完整交易对")
    
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
    print(f"  做多开仓信号: {len(all_long_entries)} 个")
    print(f"  做多平仓信号: {len(all_long_exits)} 个")
    print(f"  做空开仓信号: {len(all_short_entries)} 个")
    print(f"  做空平仓信号: {len(all_short_exits)} 个")
    print(f"  完整做多交易(过滤后): {len(long_trades)} 个")
    print(f"  完整做空交易(过滤后): {len(short_trades)} 个")
    
    # 打印前10个交易的详细信息（如果有完整交易）
    if top_trades:
        print("\n前10个最佳交易:")
        for i, trade in enumerate(top_trades):
            trade_type = "做多" if trade in long_trades else "做空"
            print(f"  {i+1}. {trade_type}交易 - 置信度: {trade['confidence']:.4f}, "
                  f"预期收益: {trade['expected_profit']*100:.2f}%, "
                  f"最大不利波动: {trade['max_adverse_excursion']*100:.2f}%, "
                  f"开仓点: {trade['entry_idx']}, 平仓点: {trade['exit_idx']}")

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
    plot_trading_signals(index_values, predictions, probabilities, use_multiscale)
    
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