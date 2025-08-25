"""
改进版预测脚本 - 针对预测效果差的问题进行优化
主要改进：
1. 更好的数据预处理和特征工程
2. 集成多种模型预测
3. 更合理的交易信号过滤
4. 详细的预测结果分析
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from model import TransformerClassifier, MultiScaleTransformerClassifier
from feature_engineering import add_features

# 配置参数
SEQ_LEN = 30
MODEL_PATH = "../model/best_longtrend_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 新增：改进的特征工程
def enhanced_feature_engineering(df):
    """
    增强的特征工程，添加更多有效特征
    """
    df = df.copy()
    
    # 1. 基础技术指标
    for window in [5, 10, 20, 50]:
        # RSI指标
        delta = df['index_value'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # 布林带
        sma = df['index_value'].rolling(window=window).mean()
        std = df['index_value'].rolling(window=window).std()
        df[f'bb_upper_{window}'] = sma + (std * 2)
        df[f'bb_lower_{window}'] = sma - (std * 2)
        df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / sma
        df[f'bb_position_{window}'] = (df['index_value'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # MACD
        if window <= 26:
            ema_fast = df['index_value'].ewm(span=12).mean()
            ema_slow = df['index_value'].ewm(span=26).mean()
            df[f'macd_{window}'] = ema_fast - ema_slow
            df[f'macd_signal_{window}'] = df[f'macd_{window}'].ewm(span=9).mean()
            df[f'macd_histogram_{window}'] = df[f'macd_{window}'] - df[f'macd_signal_{window}']
    
    # 2. 价格模式特征
    # 支撑阻力位
    df['resistance_distance'] = 0.0
    df['support_distance'] = 0.0
    
    lookback = 20
    for i in range(lookback, len(df)):
        recent_highs = df['index_value'].iloc[i-lookback:i].max()
        recent_lows = df['index_value'].iloc[i-lookback:i].min()
        current_price = df['index_value'].iloc[i]
        
        df.loc[i, 'resistance_distance'] = (recent_highs - current_price) / current_price
        df.loc[i, 'support_distance'] = (current_price - recent_lows) / current_price
    
    # 3. 多时间周期特征
    for window in [3, 7, 14, 21]:
        # 价格动量
        df[f'momentum_{window}'] = df['index_value'].pct_change(window)
        
        # 价格排名（当前价格在过去N期中的位置）
        df[f'price_rank_{window}'] = df['index_value'].rolling(window=window).rank(pct=True)
        
        # 波动率
        df[f'volatility_{window}'] = df['index_value'].pct_change().rolling(window=window).std()
        
        # 趋势强度
        df[f'trend_strength_{window}'] = (df['index_value'] - df['index_value'].shift(window)) / df['index_value'].rolling(window=window).std()
    
    # 4. 成交量相关特征（如果有volume列）
    if 'volume' in df.columns:
        for window in [5, 10, 20]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
    
    # 5. 市场结构特征
    # 高低点识别
    df['local_high'] = df['index_value'].shift(1) < df['index_value']
    df['local_low'] = df['index_value'].shift(1) > df['index_value']
    
    # 价格缺口
    df['gap_up'] = (df['index_value'] - df['index_value'].shift(1)) / df['index_value'].shift(1) > 0.002
    df['gap_down'] = (df['index_value'] - df['index_value'].shift(1)) / df['index_value'].shift(1) < -0.002
    
    # 应用原有特征工程
    df = add_features(df)
    
    # 处理缺失值和异常值
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # 处理无穷大值
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 异常值处理（使用IQR方法）
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 填充缺失值
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def improved_data_preprocessing(df):
    """
    改进的数据预处理
    """
    # 增强特征工程
    df = enhanced_feature_engineering(df)
    
    # 准备特征
    exclude_cols = ['label', 'index_value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features = df[feature_cols].values
    
    # 使用RobustScaler，对异常值更鲁棒
    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    
    return features, df['index_value'].values, feature_cols

def ensemble_predict(models, X, use_multiscale=False):
    """
    集成多个模型的预测结果
    """
    all_predictions = []
    all_probabilities = []
    
    for model in models:
        with torch.no_grad():
            if use_multiscale:
                X_tensors = {}
                for key, value in X.items():
                    X_tensors[key] = torch.tensor(value, dtype=torch.float32).to(DEVICE)
                outputs = model(X_tensors)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
                outputs = model(X_tensor)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_predictions.append(preds)
            all_probabilities.append(probs.cpu().numpy())
    
    # 投票法集成预测
    ensemble_predictions = []
    ensemble_probabilities = []
    
    for i in range(len(all_predictions[0])):
        # 预测投票
        votes = [pred[i] for pred in all_predictions]
        ensemble_pred = max(set(votes), key=votes.count)
        ensemble_predictions.append(ensemble_pred)
        
        # 概率平均
        avg_prob = np.mean([prob[i] for prob in all_probabilities], axis=0)
        ensemble_probabilities.append(avg_prob)
    
    return np.array(ensemble_predictions), np.array(ensemble_probabilities)

def improved_signal_filtering(predictions, probabilities, index_values, confidence_threshold=0.3):
    """
    改进的信号过滤，减少虚假信号
    降低置信度阈值以增加信号密度
    """
    filtered_predictions = predictions.copy()
    
    print(f"📊 应用信号过滤 - 置信度阈值: {confidence_threshold}")
    original_signal_count = np.sum(predictions != 0)
    
    for i in range(len(predictions)):
        # 1. 置信度过滤 - 降低阈值
        max_prob = np.max(probabilities[i])
        if max_prob < confidence_threshold:
            filtered_predictions[i] = 0  # 设为无操作
        
        # 2. 连续信号过滤（避免频繁交易）- 放宽条件
        if i > 2 and predictions[i] == predictions[i-1] == predictions[i-2] and predictions[i] != 0:
            filtered_predictions[i] = 0
        
        # 3. 价格确认过滤 - 放宽条件
        if i >= 3:  # 减少需要的历史数据
            recent_trend = (index_values[i] - index_values[i-3]) / index_values[i-3]
            
            # 放宽价格确认条件
            if (predictions[i] == 1 and recent_trend < -0.003) or \
               (predictions[i] == 3 and recent_trend > 0.003):
                # 只在趋势明显相反时才过滤
                filtered_predictions[i] = 0
    
    filtered_signal_count = np.sum(filtered_predictions != 0)
    filter_ratio = filtered_signal_count / original_signal_count if original_signal_count > 0 else 0
    
    print(f"📈 信号过滤结果: {original_signal_count} -> {filtered_signal_count} (保留率: {filter_ratio:.1%})")
    
    return filtered_predictions

def detailed_performance_analysis(predictions, probabilities, index_values):
    """
    详细的预测性能分析 - 增加开仓平仓匹配检查
    """
    print("\n" + "="*60)
    print("详细预测性能分析")
    print("="*60)
    
    # 1. 预测分布分析
    unique, counts = np.unique(predictions, return_counts=True)
    label_names = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
    
    print("\n1. 预测分布:")
    for label, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        print(f"   {label_names[label]} ({label}): {count} 个 ({percentage:.1f}%)")
    
    # 2. 置信度分析
    print("\n2. 置信度分析:")
    max_confidences = np.max(probabilities, axis=1)
    print(f"   平均置信度: {np.mean(max_confidences):.3f}")
    print(f"   置信度标准差: {np.std(max_confidences):.3f}")
    print(f"   高置信度预测(>0.7): {np.sum(max_confidences > 0.7)} 个")
    print(f"   低置信度预测(<0.5): {np.sum(max_confidences < 0.5)} 个")
    
    # 3. 信号密度分析
    non_zero_signals = np.sum(predictions != 0)
    signal_density = non_zero_signals / len(predictions)
    print(f"\n3. 信号密度: {signal_density:.3f} ({non_zero_signals}/{len(predictions)})")
    
    # 4. 简单回测分析（包含开仓平仓匹配检查）
    print("\n4. 简单回测分析:")
    analyze_simple_backtest(predictions, index_values)
    
    return {
        'prediction_distribution': dict(zip(unique, counts)),
        'avg_confidence': np.mean(max_confidences),
        'signal_density': signal_density
    }

def analyze_simple_backtest(predictions, index_values):
    """
    简单的回测分析 - 修复开仓无平仓的问题
    """
    position = 0  # 0: 无仓位, 1: 多头, -1: 空头
    entry_price = 0
    trades = []
    equity_curve = [1.0]  # 从1开始的权益曲线
    
    # 记录未平仓的开仓信号
    unmatched_long_entries = 0
    unmatched_short_entries = 0
    
    for i in range(len(predictions)):
        pred = predictions[i]
        current_price = index_values[i] if i < len(index_values) else index_values[-1]
        
        # 开仓信号
        if pred == 1 and position == 0:  # 做多开仓
            position = 1
            entry_price = current_price
        elif pred == 3 and position == 0:  # 做空开仓
            position = -1
            entry_price = current_price
        
        # 平仓信号
        elif pred == 2 and position == 1:  # 做多平仓
            profit = (current_price - entry_price) / entry_price
            trades.append(profit)
            position = 0
        elif pred == 4 and position == -1:  # 做空平仓
            profit = (entry_price - current_price) / entry_price
            trades.append(profit)
            position = 0
        
        # 计算未匹配的信号
        elif pred == 1 and position != 0:  # 开仓信号但已有仓位
            unmatched_long_entries += 1
        elif pred == 3 and position != 0:
            unmatched_short_entries += 1
        elif pred == 2 and position != 1:  # 平仓信号但无对应仓位
            pass  # 无法平仓
        elif pred == 4 and position != -1:
            pass  # 无法平仓
        
        # 更新权益曲线
        if len(trades) > 0:
            total_return = np.prod([1 + t for t in trades])
            equity_curve.append(total_return)
        else:
            equity_curve.append(equity_curve[-1])
    
    # 检查未平仓的交易
    open_position_warning = ""
    if position != 0:
        pos_type = "多头" if position == 1 else "空头"
        open_position_warning = f"\n   ⚠️  最后仍有未平仓的{pos_type}仓位！"
    
    if trades:
        win_rate = np.sum(np.array(trades) > 0) / len(trades)
        avg_profit = np.mean(trades)
        max_profit = np.max(trades)
        max_loss = np.min(trades)
        total_return = np.prod([1 + t for t in trades]) - 1
        
        print(f"   总交易次数: {len(trades)}")
        print(f"   胜率: {win_rate:.2%}")
        print(f"   平均收益: {avg_profit:.4f} ({avg_profit*100:.2f}%)")
        print(f"   最大盈利: {max_profit:.4f} ({max_profit*100:.2f}%)")
        print(f"   最大亏损: {max_loss:.4f} ({max_loss*100:.2f}%)")
        print(f"   总收益率: {total_return:.4f} ({total_return*100:.2f}%)")
        
        # 显示问题诊断
        if unmatched_long_entries > 0 or unmatched_short_entries > 0:
            print(f"   ⚠️  发现标签问题:")
            if unmatched_long_entries > 0:
                print(f"     无法匹配的做多开仓: {unmatched_long_entries} 个")
            if unmatched_short_entries > 0:
                print(f"     无法匹配的做空开仓: {unmatched_short_entries} 个")
            print(f"     💡 建议使用 generate_complete_trading_labels.py 生成完整交易标签")
        
        print(open_position_warning)
        
    else:
        print("   没有完成的交易")
        if unmatched_long_entries > 0 or unmatched_short_entries > 0:
            print(f"   ⚠️  但发现未匹配的开仓信号:")
            print(f"     做多开仓: {unmatched_long_entries} 个")
            print(f"     做空开仓: {unmatched_short_entries} 个")
            print(f"     💡 说明标签生成有问题，建议使用 generate_complete_trading_labels.py")

def plot_improved_signals(index_values, predictions, probabilities, output_filename=None, use_multiscale=False):
    """
    改进的信号绘制，包含更多分析信息
    """
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # 根据是否使用多尺度模型调整索引
    start_idx = 60 if use_multiscale else SEQ_LEN
    
    # 子图1: 价格和交易信号
    ax1.plot(index_values, label='Price', color='blue', linewidth=1)
    
    # 绘制交易信号
    signal_colors = {1: 'red', 2: 'red', 3: 'green', 4: 'green'}
    signal_markers = {1: '^', 2: 'v', 3: '^', 4: 'v'}
    signal_labels = {1: 'Long Entry', 2: 'Long Exit', 3: 'Short Entry', 4: 'Short Exit'}
    
    for pred_type in [1, 2, 3, 4]:
        indices = []
        prices = []
        for i, pred in enumerate(predictions):
            if pred == pred_type:
                idx = i + start_idx
                if idx < len(index_values):
                    indices.append(idx)
                    prices.append(index_values[idx])
        
        if indices:
            ax1.scatter(indices, prices, color=signal_colors[pred_type], 
                       marker=signal_markers[pred_type], s=60, 
                       label=signal_labels[pred_type], zorder=5, alpha=0.8)
    
    ax1.set_title('价格图表与交易信号')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 预测置信度
    max_confidences = np.max(probabilities, axis=1)
    x_range = range(start_idx, start_idx + len(max_confidences))
    ax2.plot(x_range, max_confidences, color='purple', alpha=0.7)
    ax2.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='置信度阈值')
    ax2.fill_between(x_range, max_confidences, alpha=0.3, color='purple')
    ax2.set_title('预测置信度')
    ax2.set_ylabel('置信度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 预测分布
    pred_counts = np.bincount(predictions, minlength=5)
    label_names = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
    colors = ['gray', 'red', 'red', 'green', 'green']
    
    bars = ax3.bar(range(5), pred_counts, color=colors, alpha=0.7)
    ax3.set_title('预测分布统计')
    ax3.set_xlabel('预测类型')
    ax3.set_ylabel('数量')
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(label_names, rotation=45)
    
    # 在柱状图上显示数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存或显示图表
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"改进版图表已保存为: {output_filename}")
        plt.close()
    else:
        plt.show()

def main_improved(data_file, use_multiscale=None, output_filename=None):
    """
    改进版主函数
    """
    print("开始改进版预测...")
    
    # 读取数据
    print(f"读取数据文件: {data_file}")
    df = pd.read_csv(data_file)
    print(f"数据条数: {len(df)}")
    
    # 改进的数据预处理
    print("进行改进的数据预处理...")
    features, index_values, feature_cols = improved_data_preprocessing(df)
    print(f"特征维度: {features.shape}")
    print(f"特征数量: {len(feature_cols)}")
    
    # 如果未指定模型类型，则尝试自动检测
    if use_multiscale is None:
        print("自动检测模型类型...")
        use_multiscale = detect_model_type(MODEL_PATH)
    
    # 创建序列
    print("创建序列数据...")
    if use_multiscale:
        X = create_multiscale_sequences_for_prediction(features, seq_lengths=[10, 30, 60])
        num_classes = 5
        print("使用多尺度模型")
    else:
        X = create_sequences_for_prediction(features, seq_len=SEQ_LEN)
        num_classes = 5
        print("使用单尺度模型")
    
    if len(X) == 0:
        print("序列数据为空，无法进行预测")
        return
    
    print(f"序列数量: {len(X)}")
    
    # 加载模型
    print("加载模型...")
    try:
        model = load_model(MODEL_PATH, features.shape[1], num_classes, use_multiscale, 
                          [10, 30, 60] if use_multiscale else None)
        models = [model]  # 可以扩展为多个模型
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 进行预测
    print("进行集成预测...")
    if len(models) == 1:
        predictions, probabilities = predict(models[0], X, use_multiscale)
    else:
        predictions, probabilities = ensemble_predict(models, X, use_multiscale)
    
    print(f"预测完成，共 {len(predictions)} 个预测结果")
    
    # 改进的信号过滤
    print("应用改进的信号过滤...")
    original_signals = np.sum(predictions != 0)
    
    # 如果原始信号就很少，使用更宽松的过滤
    if original_signals / len(predictions) < 0.01:  # 如果信号密度<1%
        print("⚠️  检测到信号密度过低，使用宽松过滤策略")
        predictions = improved_signal_filtering(predictions, probabilities, index_values, confidence_threshold=0.25)
    elif original_signals / len(predictions) < 0.03:  # 如果信号密度<3%
        print("📊 检测到信号密度较低，使用中等过滤策略")
        predictions = improved_signal_filtering(predictions, probabilities, index_values, confidence_threshold=0.35)
    else:
        predictions = improved_signal_filtering(predictions, probabilities, index_values, confidence_threshold=0.45)
    
    filtered_signals = np.sum(predictions != 0)
    print(f"信号过滤: {original_signals} -> {filtered_signals} ({filtered_signals/original_signals:.2%} 保留)" if original_signals > 0 else "信号过滤: 无原始信号")
    
    # 详细性能分析
    analysis_results = detailed_performance_analysis(predictions, probabilities, index_values)
    
    # 可视化结果
    print("生成改进版图表...")
    plot_improved_signals(index_values, predictions, probabilities, output_filename, use_multiscale)
    
    print("改进版预测和分析完成!")
    return analysis_results

# 导入原有函数
def load_model(model_path, input_dim, num_classes, use_multiscale=False, seq_lengths=[10, 30, 60]):
    """加载训练好的模型"""
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
            dropout=0.1
        ).to(DEVICE)
    
    # 安全加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    except RuntimeError as e:
        if "Missing key" in str(e) or "Unexpected key" in str(e):
            print("警告: 模型结构与权重不匹配，尝试自动修复...")
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise e
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        raise e
        
    model.eval()
    return model

def create_sequences_for_prediction(features, seq_len=SEQ_LEN):
    """为预测创建序列数据"""
    X = []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
    return np.array(X)

def create_multiscale_sequences_for_prediction(features, seq_lengths=[10, 30, 60]):
    """为多尺度模型创建序列数据"""
    X_multi = {str(length): [] for length in seq_lengths}
    max_len = max(seq_lengths)
    
    for i in range(len(features) - max_len):
        for seq_len in seq_lengths:
            X_multi[str(seq_len)].append(features[i:i + seq_len])
    
    for seq_len in seq_lengths:
        X_multi[str(seq_len)] = np.array(X_multi[str(seq_len)])
    return X_multi

def predict(model, X, use_multiscale=False):
    """使用模型进行预测"""
    with torch.no_grad():
        if use_multiscale:
            X_tensors = {}
            for key, value in X.items():
                X_tensors[key] = torch.tensor(value, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensors)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensor)
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds, probs.cpu().numpy()

def detect_model_type(model_path):
    """尝试检测模型类型（单尺度或多尺度）"""
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        multiscale_keys = [key for key in state_dict.keys() if 'input_fcs' in key or 'transformer_encoders' in key]
        if multiscale_keys:
            print("检测到多尺度模型权重")
            return True
        else:
            print("检测到单尺度模型权重")
            return False
    except Exception as e:
        print(f"检测模型类型时出错: {e}")
        return False

if __name__ == "__main__":
    # 批量处理../data/目录下的所有CSV文件
    # 优先级：宽松标签 > 改进标签 > 完整交易标签 > 原始数据
    relaxed_labels_dir = "../data_with_relaxed_labels/"
    improved_data_dir = "../data_with_improved_labels/"
    complete_labels_dir = "../data_with_complete_labels/"
    original_data_dir = "../data/"
    
    # 检查数据目录优先级
    if os.path.exists(relaxed_labels_dir):
        data_dir = relaxed_labels_dir
        data_type = "relaxed"
        print(f"🎆 使用宽松参数标签数据目录: {data_dir}")
        print(f"📊 期望信号密度: 2.5-3.1%, 胜率: 100%")
    elif os.path.exists(improved_data_dir):
        data_dir = improved_data_dir
        data_type = "improved"
        print(f"🔄 使用改进标签数据目录: {data_dir}")
        print(f"📊 期望信号密度: ~37.8%")
    elif os.path.exists(complete_labels_dir):
        data_dir = complete_labels_dir
        data_type = "complete"
        print(f"✅ 使用完整交易标签数据目录: {data_dir}")
    else:
        data_dir = original_data_dir
        data_type = "original"
        print(f"⚠️  使用原始数据目录: {data_dir}")
        print(f"💡 提示: 运行 diagnose_signal_density.py 生成改进标签")
    
    use_multiscale = True  # 显式设置为多尺度
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        sys.exit(1)
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"在目录 {data_dir} 中未找到CSV文件")
        sys.exit(1)
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 确保输出目录存在
    if data_type == "relaxed":
        output_dir = "../predictions_with_relaxed_labels/"
    elif data_type == "improved":
        output_dir = "../predictions_with_improved_labels/"
    elif data_type == "complete":
        output_dir = "../predictions_with_complete_labels/"
    else:
        output_dir = "../predictions_improved/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个CSV文件
    all_results = {}
    for csv_file in csv_files:
        print(f"\n{'='*80}")
        print(f"处理文件: {csv_file}")
        
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        if data_type == "relaxed":
            output_filename = os.path.join(output_dir, f"{base_name}_relaxed_labels.png")
        elif data_type == "improved":
            output_filename = os.path.join(output_dir, f"{base_name}_improved_labels.png")
        elif data_type == "complete":
            output_filename = os.path.join(output_dir, f"{base_name}_complete_labels.png")
        else:
            output_filename = os.path.join(output_dir, f"{base_name}_improved_prediction.png")
        
        try:
            results = main_improved(csv_file, use_multiscale, output_filename)
            all_results[base_name] = results
        except Exception as e:
            print(f"处理文件 {csv_file} 时发生错误: {e}")
            continue
    
    # 总结所有文件的结果
    print(f"\n{'='*80}")
    print("所有文件处理完成！总结报告:")
    print(f"改进版预测结果图片保存在: {output_dir}")
    
    # 给出进一步改进建议
    if data_type == "relaxed":
        if all_results and np.mean([r['signal_density'] for r in all_results.values()]) > 0.02:
            print(f"\n🎆 宽松标签效果很好！")
            print(f"   建议下一步:")
            print(f"   1. 使用当前数据重新训练模型")
            print(f"   2. 考虑进一步优化特征工程")
            print(f"   3. 验证实际交易效果")
        else:
            print(f"\n🔧 宽松标签仍需调整:")
            print(f"   1. 进一步降低 min_profit_target")
            print(f"   2. 减少 min_hold_time")
            print(f"   3. 缩小 min_signal_gap")
    elif data_type == "original":
        print(f"\n💡 建议的下一步改进:")
        print(f"   1. 运行 'python diagnose_signal_density.py' 生成改进标签")
        print(f"   2. 使用新标签重新训练模型")
        print(f"   3. 再次运行此预测脚本查看效果")
    elif data_type == "improved":
        print(f"\n📊 改进标签效果分析:")
        if all_results:
            avg_signal_density = np.mean([r['signal_density'] for r in all_results.values()])
            if avg_signal_density > 0.3:
                print(f"   ✅ 信号密度很高: {avg_signal_density:.1%}")
                print(f"   建议: 检查交易频率是否过高，考虑使用完整交易标签")
            else:
                print(f"   信号密度: {avg_signal_density:.1%}")
    else:
        if all_results and np.mean([r['signal_density'] for r in all_results.values()]) < 0.01:
            print(f"\n🚨 信号密度仍然过低的解决建议:")
            print(f"   1. 检查模型训练数据质量")
            print(f"   2. 运行 diagnose_signal_density.py 生成宽松标签")
            print(f"   3. 重新训练模型")
        else:
            print(f"\n🎯 如果预测效果仍不理想，可以尝试:")
            print(f"   1. 调整 generate_complete_trading_labels.py 中的参数")
            print(f"   2. 重新训练模型")
            print(f"   3. 使用集成学习方法")
    
    if all_results:
        avg_confidence = np.mean([r['avg_confidence'] for r in all_results.values()])
        if avg_confidence < 0.4:
            print(f"平均预测置信度较低: {avg_confidence:.3f}")
        avg_signal_density = np.mean([r['signal_density'] for r in all_results.values()])
        print(f"平均信号密度: {avg_signal_density:.3f}")
        
        if avg_signal_density < 0.001:
            print(f"\n🚨 严重问题: 平均信号密度 {avg_signal_density:.3f} 极低!")
            print(f"建议立即:")
            print(f"  1. 检查模型是否正确加载")
            print(f"  2. 验证训练数据标签分布")
            print(f"  3. 重新生成标签并训练模型")