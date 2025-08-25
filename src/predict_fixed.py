#!/usr/bin/env python3
"""
修复版预测脚本 - 解决特征数量不匹配和无信号问题
主要修复：
1. 确保使用完整的88个特征（与训练时一致）
2. 降低预测置信度阈值
3. 修复开仓无平仓问题的诊断
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
import glob
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from model import TransformerClassifier, MultiScaleTransformerClassifier
from feature_engineering import add_features

# 配置参数 - 优化序列长度以捕获早盘机会
SEQ_LEN = 15  # 15个点 × 4秒 = 60秒历史数据，适合早盘快速反应
MODEL_PATH = "../model/best_longtrend_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data_fixed(csv_file):
    """
    修复版数据加载和预处理，确保使用完整的88个特征
    """
    print(f"📁 加载数据文件: {os.path.basename(csv_file)}")
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"   原始数据维度: {df.shape}")
    print(f"   数据列: {list(df.columns)}")
    
    # 确保必要列存在
    required_cols = ['a', 'b', 'c', 'd', 'index_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必要列: {missing_cols}")
    
    # 应用完整的特征工程（确保生成88个特征）
    print("   应用完整特征工程...")
    df_with_features = add_features(df)
    print(f"   特征工程后维度: {df_with_features.shape}")
    
    # 准备特征列（排除标签和目标列）
    exclude_cols = ['label', 'index_value']
    feature_cols = [col for col in df_with_features.columns if col not in exclude_cols]
    
    print(f"   ✅ 特征数量: {len(feature_cols)} (期望: 88)")
    if len(feature_cols) != 88:
        print(f"   ⚠️  特征数量不匹配！期望88个，实际{len(feature_cols)}个")
        # 列出特征以便调试
        print(f"   特征列表: {feature_cols}")
    
    # 提取特征和标签
    features = df_with_features[feature_cols].values
    labels = df_with_features['label'].values if 'label' in df_with_features.columns else None
    index_values = df_with_features['index_value'].values
    
    # 处理缺失值和异常值
    print("   处理缺失值和异常值...")
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"   ✅ 最终特征维度: {features.shape}")
    
    return features, labels, index_values, feature_cols

def load_model_fixed(model_path, input_dim, num_classes=5):
    """
    修复版模型加载，自动检测模型类型
    """
    print(f"🤖 加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 检测模型类型
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model_keys = list(state_dict.keys())
        
        # 检查是否是多尺度模型
        multiscale_keys = [key for key in model_keys if 'input_fcs' in key or 'transformer_encoders' in key]
        use_multiscale = len(multiscale_keys) > 0
        
        print(f"   检测到{'多尺度' if use_multiscale else '单尺度'}模型")
        print(f"   期望输入维度: {input_dim}")
        
        # 创建模型
        if use_multiscale:
            model = MultiScaleTransformerClassifier(
                input_dim=input_dim,
                model_dim=128,
                num_heads=8,
                num_layers=4,
                num_classes=num_classes,
                seq_lengths=[10, 30, 60],
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
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"   ✅ 模型加载成功")
        return model, use_multiscale
        
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        raise e

def predict_fixed(model, X, use_multiscale=False, confidence_threshold=0.2):
    """
    修复版预测，使用更低的置信度阈值
    """
    print(f"🔮 开始预测 (置信度阈值: {confidence_threshold})")
    
    with torch.no_grad():
        if use_multiscale:
            X_tensors = {}
            for key, value in X.items():
                X_tensors[key] = torch.tensor(value, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensors)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensor)
        
        # 获取概率和预测
        probs = torch.nn.functional.softmax(outputs, dim=1)
        raw_preds = torch.argmax(outputs, dim=1).cpu().numpy()
        probabilities = probs.cpu().numpy()
        
        # 应用置信度过滤
        predictions = raw_preds.copy()
        max_probs = np.max(probabilities, axis=1)
        low_confidence_mask = max_probs < confidence_threshold
        predictions[low_confidence_mask] = 0  # 低置信度设为无操作
        
        # 统计信息
        original_signals = np.sum(raw_preds != 0)
        filtered_signals = np.sum(predictions != 0)
        
        print(f"   原始信号: {original_signals}")
        print(f"   过滤后信号: {filtered_signals}")
        print(f"   平均置信度: {np.mean(max_probs):.3f}")
        
        return predictions, probabilities

def create_sequences_fixed(features, seq_len=SEQ_LEN):
    """
    创建序列数据
    """
    print(f"📊 创建序列数据 (序列长度: {seq_len})")
    
    if len(features) < seq_len:
        raise ValueError(f"数据长度不足以创建序列: 需要{seq_len}，实际{len(features)}")
    
    X = []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
    
    X = np.array(X)
    print(f"   ✅ 序列数据维度: {X.shape}")
    
    return X

def analyze_predictions_fixed(predictions, probabilities, index_values):
    """
    修复版预测分析，详细检查开仓平仓匹配
    """
    print("\n" + "="*60)
    print("📊 预测结果分析")
    print("="*60)
    
    # 1. 预测分布
    pred_counts = Counter(predictions)
    label_names = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
    
    print("\n1. 📈 预测分布:")
    for label in range(5):
        count = pred_counts.get(label, 0)
        percentage = count / len(predictions) * 100
        print(f"   {label_names[label]} ({label}): {count} 个 ({percentage:.1f}%)")
    
    # 2. 信号密度
    signal_count = sum(count for label, count in pred_counts.items() if label != 0)
    signal_density = signal_count / len(predictions)
    print(f"\n2. 🎯 信号密度: {signal_density:.4f} ({signal_density*100:.2f}%)")
    
    # 3. 置信度分析
    max_confidences = np.max(probabilities, axis=1)
    print(f"\n3. 💪 置信度分析:")
    print(f"   平均置信度: {np.mean(max_confidences):.3f}")
    print(f"   高置信度(>0.7): {np.sum(max_confidences > 0.7)} 个")
    print(f"   中置信度(0.5-0.7): {np.sum((max_confidences >= 0.5) & (max_confidences <= 0.7))} 个")
    print(f"   低置信度(<0.5): {np.sum(max_confidences < 0.5)} 个")
    
    # 4. 开仓平仓匹配检查
    print(f"\n4. 🔍 开仓平仓匹配检查:")
    long_entries = pred_counts.get(1, 0)  # 做多开仓
    long_exits = pred_counts.get(2, 0)    # 做多平仓
    short_entries = pred_counts.get(3, 0) # 做空开仓
    short_exits = pred_counts.get(4, 0)   # 做空平仓
    
    print(f"   做多: 开仓{long_entries}个, 平仓{long_exits}个")
    print(f"   做空: 开仓{short_entries}个, 平仓{short_exits}个")
    
    # 匹配度检查
    long_match = abs(long_entries - long_exits)
    short_match = abs(short_entries - short_exits)
    
    if long_match == 0 and short_match == 0:
        print(f"   ✅ 开仓平仓完全匹配!")
    else:
        print(f"   ⚠️  开仓平仓不匹配:")
        if long_match > 0:
            print(f"     做多缺失: {long_match} 个{'平仓' if long_entries > long_exits else '开仓'}信号")
        if short_match > 0:
            print(f"     做空缺失: {short_match} 个{'平仓' if short_entries > short_exits else '开仓'}信号")
    
    # 5. 问题诊断
    print(f"\n5. 🔧 问题诊断:")
    issues = []
    
    if signal_density < 0.001:
        issues.append("信号密度极低(<0.1%)")
    elif signal_density < 0.01:
        issues.append("信号密度偏低(<1%)")
    
    if np.mean(max_confidences) < 0.4:
        issues.append("平均置信度过低")
    
    if (long_entries + short_entries) == 0:
        issues.append("没有任何开仓信号")
    
    if long_match > 0 or short_match > 0:
        issues.append("开仓平仓信号不匹配")
    
    if issues:
        print(f"   发现问题:")
        for issue in issues:
            print(f"     ❌ {issue}")
    else:
        print(f"   ✅ 预测质量良好!")
    
    return {
        'signal_density': signal_density,
        'avg_confidence': np.mean(max_confidences),
        'prediction_distribution': dict(pred_counts),
        'issues': issues
    }

def plot_results_fixed(index_values, predictions, probabilities, output_filename):
    """
    可视化预测结果
    """
    print(f"📊 生成预测图表: {output_filename}")
    
    plt.figure(figsize=(15, 10))
    
    # 主图：价格曲线和信号
    plt.subplot(3, 1, 1)
    plt.plot(index_values, label='价格', alpha=0.7)
    
    # 标记信号
    for i, pred in enumerate(predictions):
        if pred == 1:  # 做多开仓
            plt.scatter(i, index_values[i], color='green', marker='^', s=50, alpha=0.8)
        elif pred == 2:  # 做多平仓
            plt.scatter(i, index_values[i], color='red', marker='v', s=50, alpha=0.8)
        elif pred == 3:  # 做空开仓
            plt.scatter(i, index_values[i], color='blue', marker='v', s=50, alpha=0.8)
        elif pred == 4:  # 做空平仓
            plt.scatter(i, index_values[i], color='orange', marker='^', s=50, alpha=0.8)
    
    plt.title('修复版预测结果 - 价格与交易信号')
    plt.legend(['价格', '做多开仓', '做多平仓', '做空开仓', '做空平仓'])
    plt.grid(True, alpha=0.3)
    
    # 子图：预测分布
    plt.subplot(3, 1, 2)
    pred_counts = Counter(predictions)
    labels = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
    values = [pred_counts.get(i, 0) for i in range(5)]
    colors = ['gray', 'green', 'red', 'blue', 'orange']
    
    bars = plt.bar(labels, values, color=colors, alpha=0.7)
    plt.title('预测信号分布')
    plt.ylabel('数量')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, values):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom')
    
    # 子图：置信度分布
    plt.subplot(3, 1, 3)
    max_confidences = np.max(probabilities, axis=1)
    plt.hist(max_confidences, bins=20, alpha=0.7, color='purple')
    plt.axvline(np.mean(max_confidences), color='red', linestyle='--', 
               label=f'平均置信度: {np.mean(max_confidences):.3f}')
    plt.title('预测置信度分布')
    plt.xlabel('置信度')
    plt.ylabel('频次')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 图表已保存: {output_filename}")

def main_fixed(csv_file, output_filename=None):
    """
    修复版主函数
    """
    print(f"🚀 修复版预测开始!")
    print(f"📁 处理文件: {os.path.basename(csv_file)}")
    
    try:
        # 1. 加载和预处理数据
        features, labels, index_values, feature_cols = load_and_preprocess_data_fixed(csv_file)
        
        # 2. 加载模型
        model, use_multiscale = load_model_fixed(MODEL_PATH, input_dim=features.shape[1])
        
        # 3. 创建序列数据
        X = create_sequences_fixed(features)
        
        # 4. 进行预测（使用更低的置信度阈值）
        predictions, probabilities = predict_fixed(model, X, use_multiscale, confidence_threshold=0.15)
        
        # 5. 分析结果
        results = analyze_predictions_fixed(predictions, probabilities, index_values)
        
        # 6. 可视化（如果需要）
        if output_filename:
            plot_results_fixed(index_values, predictions, probabilities, output_filename)
        
        print(f"\n✅ 预测完成!")
        return results
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔧 修复版预测脚本")
    print("="*50)
    
    # 检查数据目录
    data_dirs = [
        "../data_with_relaxed_labels/",   # 最高优先级：宽松标签
        "../data_with_improved_labels/",  # 第二优先级：改进标签
        "../data_with_complete_labels/",  # 第三优先级：完整标签
        "../data/"                        # 最低优先级：原始数据
    ]
    
    selected_dir = None
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            if csv_files:
                selected_dir = data_dir
                print(f"🎯 使用数据目录: {data_dir} ({len(csv_files)}个文件)")
                break
    
    if not selected_dir:
        print("❌ 没有找到有效的数据目录")
        exit(1)
    
    # 测试单个文件
    test_file = csv_files[0]
    output_file = f"./fixed_prediction_{os.path.splitext(os.path.basename(test_file))[0]}.png"
    
    print(f"\n🧪 测试文件: {os.path.basename(test_file)}")
    
    results = main_fixed(test_file, output_file)
    
    if results:
        print(f"\n🎉 测试成功完成!")
        print(f"📊 结果摘要:")
        print(f"   信号密度: {results['signal_density']:.4f}")
        print(f"   平均置信度: {results['avg_confidence']:.3f}")
        
        if results['issues']:
            print(f"   需要关注的问题: {', '.join(results['issues'])}")
        else:
            print(f"   ✅ 预测质量良好!")
    else:
        print(f"❌ 测试失败")