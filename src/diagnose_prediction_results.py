#!/usr/bin/env python3
"""
预测结果诊断工具
分析为什么预测文件中没有开仓和平仓信号
"""

import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt

def diagnose_prediction_issue():
    """
    诊断预测结果问题
    """
    print("🔍 预测结果问题诊断工具")
    print("="*60)
    
    # 1. 检查数据目录和文件
    print("\n📁 检查数据目录和文件...")
    
    relaxed_dir = "./data_with_relaxed_labels/"
    model_path = "./model/best_longtrend_model.pth"
    
    # 检查宽松标签数据
    if os.path.exists(relaxed_dir):
        csv_files = glob.glob(os.path.join(relaxed_dir, "*.csv"))
        print(f"✅ 宽松标签数据目录存在: {len(csv_files)} 个文件")
        
        # 检查标签分布
        sample_file = csv_files[0] if csv_files else None
        if sample_file:
            df = pd.read_csv(sample_file)
            if 'label' in df.columns:
                labels = df['label'].values
                label_counts = Counter(labels)
                signal_density = sum(count for label, count in label_counts.items() if label != 0) / len(labels)
                
                print(f"   样本文件: {os.path.basename(sample_file)}")
                print(f"   标签分布: {dict(label_counts)}")
                print(f"   信号密度: {signal_density:.4f} ({signal_density*100:.2f}%)")
                
                if signal_density > 0.02:
                    print(f"   ✅ 训练数据标签质量良好")
                    data_quality_ok = True
                else:
                    print(f"   ⚠️  训练数据信号密度偏低")
                    data_quality_ok = False
            else:
                print(f"   ❌ 样本文件缺少label列")
                data_quality_ok = False
        else:
            print(f"   ❌ 没有找到CSV文件")
            data_quality_ok = False
    else:
        print(f"❌ 宽松标签数据目录不存在")
        data_quality_ok = False
    
    # 2. 检查模型文件
    print(f"\n🤖 检查模型文件...")
    if os.path.exists(model_path):
        print(f"✅ 模型文件存在: {model_path}")
        
        # 尝试加载模型权重查看信息
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            model_keys = list(state_dict.keys())
            print(f"   模型参数数量: {len(model_keys)}")
            print(f"   前几个参数名: {model_keys[:3]}")
            
            # 检查是否是多尺度模型
            multiscale_keys = [key for key in model_keys if 'input_fcs' in key or 'transformer_encoders' in key]
            if multiscale_keys:
                print(f"   ✅ 检测到多尺度模型")
                model_type = "multiscale"
            else:
                print(f"   📊 检测到单尺度模型")
                model_type = "single"
                
            model_ok = True
        except Exception as e:
            print(f"   ❌ 模型文件损坏: {e}")
            model_ok = False
    else:
        print(f"❌ 模型文件不存在")
        model_ok = False
    
    # 3. 简单预测测试
    print(f"\n🧪 进行简单预测测试...")
    if data_quality_ok and model_ok and csv_files:
        try:
            # 导入必要的模型类
            from model import TransformerClassifier, MultiScaleTransformerClassifier
            
            # 读取测试数据
            test_file = csv_files[0]
            df = pd.read_csv(test_file)
            
            print(f"   测试文件: {os.path.basename(test_file)}")
            print(f"   数据行数: {len(df)}")
            
            # 检查数据列
            required_cols = ['a', 'b', 'c', 'd', 'index_value']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"   ❌ 缺少必要列: {missing_cols}")
                return
            
            # 准备特征数据
            features = df[['a', 'b', 'c', 'd', 'index_value']].values
            print(f"   特征维度: {features.shape}")
            
            # 创建序列数据（简化版本）
            seq_len = 60
            if len(features) > seq_len:
                X = []
                for i in range(len(features) - seq_len):
                    X.append(features[i:i + seq_len])
                X = np.array(X)
                print(f"   序列数据维度: {X.shape}")
                
                # 加载模型
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                if model_type == "multiscale":
                    model = MultiScaleTransformerClassifier(
                        input_dim=features.shape[1],
                        model_dim=128,
                        num_heads=8,
                        num_layers=4,
                        num_classes=5,  # 0,1,2,3,4
                        seq_lengths=[10, 30, 60],
                        dropout=0.1
                    ).to(device)
                else:
                    model = TransformerClassifier(
                        input_dim=features.shape[1],
                        model_dim=128,
                        num_heads=8,
                        num_layers=4,
                        num_classes=5,  # 0,1,2,3,4
                        dropout=0.1
                    ).to(device)
                
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                model.eval()
                
                # 进行预测（取前100个样本测试）
                test_samples = min(100, len(X))
                X_test = X[:test_samples]
                
                with torch.no_grad():
                    if model_type == "multiscale":
                        # 为多尺度模型准备数据
                        X_tensors = {}
                        seq_lengths = [10, 30, 60]
                        for seq_len in seq_lengths:
                            X_seq = []
                            for i in range(len(X_test)):
                                if seq_len <= X_test.shape[1]:
                                    X_seq.append(X_test[i][:seq_len])
                                else:
                                    # 如果序列长度不够，重复最后一个值
                                    seq_data = X_test[i]
                                    while len(seq_data) < seq_len:
                                        seq_data = np.vstack([seq_data, seq_data[-1:]])
                                    X_seq.append(seq_data[:seq_len])
                            X_tensors[str(seq_len)] = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)
                        
                        outputs = model(X_tensors)
                    else:
                        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                        outputs = model(X_tensor)
                    
                    # 获取预测结果
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                    probabilities = probs.cpu().numpy()
                
                # 分析预测结果
                pred_counts = Counter(predictions)
                total_preds = len(predictions)
                signal_count = sum(count for label, count in pred_counts.items() if label != 0)
                prediction_signal_density = signal_count / total_preds
                
                print(f"   ✅ 预测测试完成")
                print(f"   预测标签分布: {dict(pred_counts)}")
                print(f"   预测信号密度: {prediction_signal_density:.4f} ({prediction_signal_density*100:.2f}%)")
                
                # 分析置信度
                max_probs = np.max(probabilities, axis=1)
                avg_confidence = np.mean(max_probs)
                print(f"   平均预测置信度: {avg_confidence:.3f}")
                
                # 诊断结果
                print(f"\n📋 诊断结果:")
                issues = []
                
                if prediction_signal_density < 0.001:
                    issues.append("❌ 预测信号密度极低 (<0.1%)")
                    print(f"   问题1: 预测几乎没有产生任何交易信号")
                    
                if avg_confidence < 0.4:
                    issues.append("⚠️  预测置信度过低")
                    print(f"   问题2: 模型预测置信度不足，可能需要重新训练")
                
                if 1 not in pred_counts and 3 not in pred_counts:
                    issues.append("❌ 没有开仓信号")
                    print(f"   问题3: 模型没有预测出任何开仓信号")
                
                if 2 not in pred_counts and 4 not in pred_counts:
                    issues.append("❌ 没有平仓信号")
                    print(f"   问题4: 模型没有预测出任何平仓信号")
                
                # 给出解决建议
                print(f"\n💡 解决建议:")
                if prediction_signal_density < 0.001:
                    print(f"   1. 🎯 降低预测置信度阈值（在predict_improved.py中调整）")
                    print(f"   2. 🔄 使用宽松标签数据重新训练模型")
                    print(f"   3. 📊 检查模型是否正确加载了宽松标签训练的权重")
                
                if avg_confidence < 0.4:
                    print(f"   4. 🚀 重新训练模型，增加训练轮数")
                    print(f"   5. 🎛️  调整模型超参数（学习率、模型大小等）")
                
                if len(issues) == 0:
                    print(f"   ✅ 预测功能正常，可能是可视化或信号过滤的问题")
                    print(f"   建议检查 plot_improved_signals 函数")
                
                return True
            else:
                print(f"   ❌ 数据长度不足以创建序列")
                return False
                
        except Exception as e:
            print(f"   ❌ 预测测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"   ❌ 无法进行预测测试（数据或模型问题）")
        return False

def quick_prediction_with_low_threshold():
    """
    使用低阈值快速预测测试
    """
    print(f"\n🧪 低阈值预测测试...")
    
    try:
        from predict_improved import main_improved
        
        # 找一个测试文件
        relaxed_dir = "./data_with_relaxed_labels/"
        csv_files = glob.glob(os.path.join(relaxed_dir, "*.csv"))
        
        if csv_files:
            test_file = csv_files[0]
            output_file = "./test_low_threshold_prediction.png"
            
            print(f"   使用文件: {os.path.basename(test_file)}")
            
            # 临时修改置信度阈值进行测试
            results = main_improved(test_file, use_multiscale=True, output_filename=output_file)
            
            if results:
                print(f"   ✅ 低阈值预测完成")
                print(f"   信号密度: {results.get('signal_density', 0):.4f}")
                print(f"   平均置信度: {results.get('avg_confidence', 0):.3f}")
                
                if results.get('signal_density', 0) > 0.001:
                    print(f"   🎯 建议：预测功能正常，调整可视化参数")
                else:
                    print(f"   🔧 建议：需要重新训练模型或调整模型参数")
            else:
                print(f"   ❌ 预测失败")
        else:
            print(f"   ❌ 找不到测试文件")
            
    except Exception as e:
        print(f"   ❌ 低阈值预测测试失败: {e}")

if __name__ == "__main__":
    success = diagnose_prediction_issue()
    
    if success:
        print(f"\n" + "="*60)
        print(f"🚀 下一步建议:")
        print(f"   1. 如果预测信号密度过低，考虑重新训练模型")
        print(f"   2. 调整 predict_improved.py 中的置信度阈值")
        print(f"   3. 检查模型是否使用了正确的宽松标签数据训练")
        print(f"   4. 运行 quick_prediction_with_low_threshold() 进行低阈值测试")
    else:
        print(f"\n❌ 诊断未能完成，请检查数据和模型文件")