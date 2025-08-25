"""
信号密度快速诊断和修复工具
专门解决预测置信度低、信号密度为0的问题
"""

import os
import numpy as np
import pandas as pd
import torch
import sys
import glob

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_signal_density_problem():
    """
    诊断信号密度问题的根本原因
    """
    print("🔍 开始信号密度问题诊断...")
    
    problems = []
    solutions = []
    
    # 1. 检查数据目录
    data_dirs = [
        ("../data_with_complete_labels/", "完整交易标签"),
        ("../data_with_improved_labels/", "改进标签"),
        ("../data/", "原始数据")
    ]
    
    available_data = []
    for data_dir, desc in data_dirs:
        if os.path.exists(data_dir):
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            if csv_files:
                available_data.append((data_dir, desc, len(csv_files)))
    
    print(f"\n📁 可用数据目录:")
    for data_dir, desc, count in available_data:
        print(f"   {desc}: {data_dir} ({count}个文件)")
    
    if not available_data:
        problems.append("❌ 没有找到任何数据文件")
        solutions.append("检查数据目录并确保有CSV文件")
        return problems, solutions
    
    # 2. 检查模型文件
    model_path = "../model/best_longtrend_model.pth"
    if not os.path.exists(model_path):
        problems.append(f"❌ 模型文件不存在: {model_path}")
        solutions.append("检查模型路径或重新训练模型")
    else:
        print(f"✅ 模型文件存在: {model_path}")
    
    # 3. 检查训练数据标签分布
    print(f"\n📊 检查训练数据标签分布...")
    for data_dir, desc, count in available_data:
        sample_file = glob.glob(os.path.join(data_dir, "*.csv"))[0]
        try:
            df = pd.read_csv(sample_file)
            if 'label' in df.columns:
                labels = df['label'].values
                label_counts = np.bincount(labels, minlength=5)
                signal_density = np.sum(labels != 0) / len(labels)
                
                print(f"   {desc} 样本文件标签分布:")
                print(f"     无操作(0): {label_counts[0]} ({label_counts[0]/len(labels)*100:.1f}%)")
                if len(label_counts) > 1:
                    print(f"     做多开仓(1): {label_counts[1]} ({label_counts[1]/len(labels)*100:.1f}%)")
                if len(label_counts) > 2:
                    print(f"     做多平仓(2): {label_counts[2]} ({label_counts[2]/len(labels)*100:.1f}%)")
                if len(label_counts) > 3:
                    print(f"     做空开仓(3): {label_counts[3]} ({label_counts[3]/len(labels)*100:.1f}%)")
                if len(label_counts) > 4:
                    print(f"     做空平仓(4): {label_counts[4]} ({label_counts[4]/len(labels)*100:.1f}%)")
                print(f"     信号密度: {signal_density:.4f} ({signal_density*100:.2f}%)")
                
                if signal_density < 0.01:
                    problems.append(f"❌ {desc} 训练数据信号密度过低: {signal_density:.4f}")
                    solutions.append(f"重新生成 {desc} 的标签，降低生成阈值")
                elif signal_density < 0.05:
                    problems.append(f"⚠️ {desc} 训练数据信号密度较低: {signal_density:.4f}")
                else:
                    print(f"   ✅ {desc} 信号密度正常")
            else:
                problems.append(f"❌ {desc} 数据文件没有label列")
                solutions.append(f"为 {desc} 生成标签")
        except Exception as e:
            problems.append(f"❌ 读取 {desc} 样本文件失败: {e}")
    
    return problems, solutions

def quick_fix_signal_density():
    """
    快速修复信号密度问题
    """
    print("🛠️ 开始快速修复信号密度问题...")
    
    # 1. 生成完整交易标签（使用更宽松的参数）
    print("\n1. 生成宽松参数的完整交易标签...")
    
    # 检查是否有generate_complete_trading_labels.py
    script_path = "generate_complete_trading_labels.py"
    if not os.path.exists(script_path):
        print(f"❌ 找不到 {script_path}")
        return False
    
    # 修改参数以增加信号密度
    print("   使用宽松参数生成标签:")
    print("   - min_profit_target: 0.005 (降低最小盈利要求)")
    print("   - optimal_profit: 0.010 (降低理想盈利)")
    print("   - min_hold_time: 10 (减少最小持仓时间)")
    print("   - min_signal_gap: 15 (减少信号间隔)")
    
    # 动态生成修改后的标签生成脚本
    return generate_relaxed_labels()

def generate_relaxed_labels():
    """
    使用宽松参数生成标签
    """
    try:
        # 导入必要模块
        from generate_complete_trading_labels import generate_complete_trading_labels
        
        data_dir = "../data/"
        output_dir = "../data_with_relaxed_labels/"
        
        if not os.path.exists(data_dir):
            print(f"❌ 数据目录不存在: {data_dir}")
            return False
        
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            print(f"❌ 没有找到CSV文件")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 处理 {len(csv_files)} 个文件...")
        success_count = 0
        
        for csv_file in csv_files[:3]:  # 先处理前3个文件测试
            try:
                print(f"   处理: {os.path.basename(csv_file)}")
                df = pd.read_csv(csv_file)
                
                # 使用宽松参数
                df_with_labels = generate_complete_trading_labels(
                    df, method="profit_target",
                    min_profit_target=0.005,  # 0.5% 最小盈利
                    optimal_profit=0.010,     # 1.0% 理想盈利
                    stop_loss=0.003,          # 0.3% 止损
                    min_hold_time=8,          # 最小持仓8个时间点
                    max_hold_time=60,         # 最大持仓60个时间点
                    min_signal_gap=12         # 信号间隔12个时间点
                )
                
                # 保存文件
                output_file = os.path.join(output_dir, os.path.basename(csv_file))
                df_with_labels.to_csv(output_file, index=False)
                
                # 检查标签分布
                labels = df_with_labels['label'].values
                signal_density = np.sum(labels != 0) / len(labels)
                print(f"     信号密度: {signal_density:.4f}")
                
                if signal_density > 0.01:
                    success_count += 1
                
            except Exception as e:
                print(f"   ❌ 处理文件失败: {e}")
        
        print(f"\n✅ 成功处理 {success_count}/3 个文件")
        print(f"宽松标签文件保存在: {output_dir}")
        
        if success_count > 0:
            print("\n🚀 下一步:")
            print("1. 使用新生成的标签重新训练模型")
            print("2. 运行预测脚本测试效果")
            return True
        
        return False
        
    except ImportError:
        print("❌ 无法导入标签生成模块")
        return False
    except Exception as e:
        print(f"❌ 生成标签失败: {e}")
        return False

def create_emergency_predict_script():
    """
    创建应急预测脚本，使用极低的置信度阈值
    """
    script_content = '''
"""
应急预测脚本 - 使用极低置信度阈值
专门解决信号密度为0的问题
"""

import numpy as np
import pandas as pd
import torch
import sys
import os

# 设置极低的置信度阈值
EMERGENCY_CONFIDENCE_THRESHOLD = 0.15

def emergency_predict():
    """应急预测"""
    print("🚨 启动应急预测模式...")
    print(f"使用极低置信度阈值: {EMERGENCY_CONFIDENCE_THRESHOLD}")
    
    # 这里可以调用原有的预测逻辑，但修改置信度阈值
    # 由于模块导入问题，这里提供建议而不是直接执行
    
    print("请手动修改 predict_improved.py 中的置信度阈值:")
    print("将 confidence_threshold=0.6 改为 confidence_threshold=0.15")
    print("然后重新运行预测脚本")

if __name__ == "__main__":
    emergency_predict()
'''
    
    emergency_script_path = "emergency_predict.py"
    with open(emergency_script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 应急预测脚本已创建: {emergency_script_path}")

def main():
    """
    主诊断和修复流程
    """
    print("🎯 信号密度问题诊断和修复工具")
    print("=" * 50)
    
    # 诊断问题
    problems, solutions = diagnose_signal_density_problem()
    
    print(f"\n📋 诊断结果:")
    if problems:
        print("发现的问题:")
        for i, problem in enumerate(problems, 1):
            print(f"   {i}. {problem}")
        
        print("\n建议的解决方案:")
        for i, solution in enumerate(solutions, 1):
            print(f"   {i}. {solution}")
    else:
        print("✅ 没有发现明显问题")
    
    # 尝试快速修复
    if any("信号密度" in p for p in problems):
        print(f"\n🔧 尝试快速修复信号密度问题...")
        if quick_fix_signal_density():
            print("✅ 快速修复成功!")
        else:
            print("❌ 快速修复失败")
            create_emergency_predict_script()
    
    # 给出最终建议
    print(f"\n💡 最终建议:")
    print("1. 如果是训练数据问题，重新生成标签并训练模型")
    print("2. 如果是模型问题，检查模型架构和权重")
    print("3. 如果是预测阈值问题，降低置信度阈值")
    print("4. 考虑使用更简单的模型进行baseline测试")

if __name__ == "__main__":
    main()