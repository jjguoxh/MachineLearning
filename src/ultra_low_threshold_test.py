#!/usr/bin/env python3
"""
超低置信度预测测试 - 强制产生交易信号
用于验证模型是否能产生非零预测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_fixed import *

def ultra_low_threshold_test(csv_file):
    """
    使用超低置信度阈值进行测试
    """
    print("🧪 超低置信度测试开始...")
    print(f"目标: 强制产生交易信号，验证模型预测能力")
    
    try:
        # 1. 加载数据和模型
        features, labels, index_values, feature_cols = load_and_preprocess_data_fixed(csv_file)
        model, use_multiscale = load_model_fixed(MODEL_PATH, input_dim=features.shape[1])
        X = create_sequences_fixed(features)
        
        # 2. 获取原始预测（无过滤）
        print("\n🔍 获取原始模型输出...")
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            raw_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            probabilities = probs.cpu().numpy()
        
        # 3. 详细分析原始预测
        print(f"\n📊 原始预测详细分析:")
        raw_counts = Counter(raw_preds)
        for label in range(5):
            count = raw_counts.get(label, 0)
            percentage = count / len(raw_preds) * 100
            print(f"   标签{label}: {count} 个 ({percentage:.1f}%)")
        
        # 4. 分析概率分布
        print(f"\n🎲 概率分布分析:")
        for label in range(5):
            label_probs = probabilities[:, label]
            print(f"   标签{label} - 最大概率: {np.max(label_probs):.4f}, 平均概率: {np.mean(label_probs):.4f}")
        
        # 5. 寻找最有可能的非零预测
        print(f"\n🔎 寻找潜在交易信号...")
        
        # 找到每个类别概率最高的样本
        for label in range(1, 5):  # 跳过无操作(0)
            max_prob_idx = np.argmax(probabilities[:, label])
            max_prob = probabilities[max_prob_idx, label]
            print(f"   标签{label}最高概率: {max_prob:.4f} (位置{max_prob_idx})")
        
        # 6. 使用极低阈值强制产生信号
        print(f"\n⚡ 使用极低阈值强制产生信号...")
        thresholds = [0.05, 0.02, 0.01, 0.001]
        
        for threshold in thresholds:
            # 重新预测并统计
            predictions = raw_preds.copy()
            max_probs = np.max(probabilities, axis=1)
            
            # 只过滤极低置信度
            low_confidence_mask = max_probs < threshold
            predictions[low_confidence_mask] = 0
            
            signal_count = np.sum(predictions != 0)
            signal_density = signal_count / len(predictions)
            
            print(f"   阈值{threshold}: {signal_count}个信号 ({signal_density:.4f})")
            
            if signal_count > 0:
                print(f"   ✅ 发现信号! 使用阈值{threshold}")
                
                # 详细分析这些信号
                filtered_counts = Counter(predictions)
                for label in range(1, 5):
                    count = filtered_counts.get(label, 0)
                    if count > 0:
                        print(f"     标签{label}: {count}个")
                
                return predictions, probabilities, threshold
        
        print(f"   ❌ 即使使用最低阈值也无法产生信号")
        return None, None, None
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None, None, None

def analyze_model_bias(csv_file):
    """
    分析模型偏向性问题
    """
    print(f"\n🔬 模型偏向性分析...")
    
    try:
        features, labels, index_values, feature_cols = load_and_preprocess_data_fixed(csv_file)
        model, use_multiscale = load_model_fixed(MODEL_PATH, input_dim=features.shape[1])
        X = create_sequences_fixed(features)
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        
        # 分析每个类别的概率分布
        print(f"\n📈 各类别概率统计:")
        label_names = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
        
        for label in range(5):
            probs = probabilities[:, label]
            print(f"   {label_names[label]}:")
            print(f"     平均概率: {np.mean(probs):.4f}")
            print(f"     最大概率: {np.max(probs):.4f}")
            print(f"     标准差: {np.std(probs):.4f}")
            print(f"     >0.5的样本: {np.sum(probs > 0.5)}个")
        
        # 检查是否存在极端偏向
        avg_probs = np.mean(probabilities, axis=0)
        max_avg_prob = np.max(avg_probs)
        max_label = np.argmax(avg_probs)
        
        print(f"\n⚖️  偏向性检查:")
        print(f"   最偏向的类别: {label_names[max_label]} (平均概率: {max_avg_prob:.4f})")
        
        if max_avg_prob > 0.8:
            print(f"   ❌ 发现严重偏向! 模型过度偏向'{label_names[max_label]}'")
        elif max_avg_prob > 0.6:
            print(f"   ⚠️  发现中度偏向")
        else:
            print(f"   ✅ 模型偏向性在合理范围内")
        
        return avg_probs
        
    except Exception as e:
        print(f"❌ 偏向性分析失败: {e}")
        return None

if __name__ == "__main__":
    print("🧪 超低置信度预测测试")
    print("="*50)
    
    # 使用宽松标签数据进行测试
    test_file = "./data_with_relaxed_labels/240110.csv"
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        exit(1)
    
    print(f"🎯 测试文件: {os.path.basename(test_file)}")
    
    # 1. 超低置信度测试
    predictions, probabilities, best_threshold = ultra_low_threshold_test(test_file)
    
    # 2. 模型偏向性分析
    bias_analysis = analyze_model_bias(test_file)
    
    # 3. 总结和建议
    print(f"\n" + "="*60)
    print(f"🎯 测试总结和建议")
    print(f"="*60)
    
    if predictions is not None:
        print(f"✅ 成功产生交易信号!")
        print(f"   最佳阈值: {best_threshold}")
        print(f"   建议: 在实际预测中使用此阈值")
    else:
        print(f"❌ 无法产生任何交易信号!")
        print(f"   问题: 模型可能训练过度保守")
        print(f"   建议:")
        print(f"     1. 🔄 使用宽松标签数据重新训练模型")
        print(f"     2. 📊 调整训练时的类别权重")
        print(f"     3. 🎯 使用成本敏感学习方法")
        print(f"     4. ⚖️  平衡训练数据中各类别的比例")
    
    if bias_analysis is not None:
        max_bias = np.max(bias_analysis)
        if max_bias > 0.8:
            print(f"\n⚠️  发现严重的模型偏向问题!")
            print(f"   模型过度偏向'无操作'类别")
            print(f"   强烈建议重新训练模型")
        else:
            print(f"\n💡 模型偏向性在可接受范围内")
            print(f"   可以尝试调整预测阈值解决问题")