#!/usr/bin/env python3
"""
快速测试宽松标签数据的预测效果
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter

def quick_test_relaxed_labels():
    """
    快速测试宽松标签数据的效果
    """
    print("🧪 快速测试宽松标签数据效果...")
    
    # 检查数据目录
    relaxed_dir = "../data_with_relaxed_labels/"
    
    if not os.path.exists(relaxed_dir):
        print(f"❌ 宽松标签目录不存在: {relaxed_dir}")
        return False
    
    # 获取文件列表
    csv_files = glob.glob(os.path.join(relaxed_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ 在 {relaxed_dir} 中未找到CSV文件")
        return False
    
    print(f"📁 找到 {len(csv_files)} 个宽松标签文件")
    
    # 随机选择几个文件进行测试
    test_files = csv_files[:3]  # 测试前3个文件
    
    total_stats = {
        'files': 0,
        'signal_densities': [],
        'label_distributions': [],
        'win_rates': []
    }
    
    for file_path in test_files:
        try:
            print(f"\n📊 测试文件: {os.path.basename(file_path)}")
            
            # 读取数据
            df = pd.read_csv(file_path)
            
            if 'label' not in df.columns:
                print(f"   ❌ 文件缺少label列")
                continue
            
            # 分析标签分布
            labels = df['label'].values
            label_counts = Counter(labels)
            
            # 计算信号密度
            total_signals = sum(count for label, count in label_counts.items() if label != 0)
            signal_density = total_signals / len(labels)
            
            # 计算完整交易数
            long_entries = label_counts.get(1, 0)
            long_exits = label_counts.get(2, 0)
            short_entries = label_counts.get(3, 0)
            short_exits = label_counts.get(4, 0)
            
            complete_trades = min(long_entries, long_exits) + min(short_entries, short_exits)
            
            print(f"   标签分布: {dict(label_counts)}")
            print(f"   信号密度: {signal_density:.4f} ({signal_density*100:.2f}%)")
            print(f"   完整交易数: {complete_trades}")
            print(f"   做多交易: {min(long_entries, long_exits)} 对")
            print(f"   做空交易: {min(short_entries, short_exits)} 对")
            
            # 验证完整性
            if long_entries == long_exits and short_entries == short_exits:
                print(f"   ✅ 交易完整性验证通过")
                integrity_ok = True
            else:
                print(f"   ❌ 交易完整性验证失败!")
                integrity_ok = False
            
            # 模拟计算胜率（基于标签逻辑）
            if 'index_value' in df.columns and complete_trades > 0:
                win_rate = simulate_win_rate(df, labels)
                print(f"   模拟胜率: {win_rate:.2%}")
                total_stats['win_rates'].append(win_rate)
            
            # 统计
            total_stats['files'] += 1
            total_stats['signal_densities'].append(signal_density)
            total_stats['label_distributions'].append(label_counts)
            
        except Exception as e:
            print(f"   ❌ 处理文件时出错: {e}")
            continue
    
    # 输出总结
    if total_stats['files'] > 0:
        print(f"\n📈 宽松标签数据质量总结:")
        print(f"   测试文件数: {total_stats['files']}")
        
        avg_density = np.mean(total_stats['signal_densities'])
        print(f"   平均信号密度: {avg_density:.4f} ({avg_density*100:.2f}%)")
        
        if total_stats['win_rates']:
            avg_win_rate = np.mean(total_stats['win_rates'])
            print(f"   平均模拟胜率: {avg_win_rate:.2%}")
        
        # 质量评估
        if avg_density >= 0.02:  # 2%以上
            print(f"   ✅ 信号密度优秀!")
        elif avg_density >= 0.01:  # 1-2%
            print(f"   📊 信号密度良好")
        else:
            print(f"   ⚠️  信号密度仍需改进")
        
        print(f"\n🚀 下一步建议:")
        if avg_density >= 0.015:
            print(f"   1. ✅ 宽松标签质量很好，可以进行预测测试")
            print(f"   2. 🎯 运行 predict_improved.py 查看预测效果")
            print(f"   3. 🔄 考虑重新训练模型以适应新标签")
        else:
            print(f"   1. 🔧 进一步调整标签生成参数")
            print(f"   2. 📊 重新运行 diagnose_signal_density.py")
            print(f"   3. 🎯 尝试更宽松的参数设置")
        
        return True
    else:
        print(f"\n❌ 没有成功测试任何文件")
        return False

def simulate_win_rate(df, labels):
    """
    模拟计算胜率（基于标签生成的交易逻辑）
    """
    try:
        prices = df['index_value'].values
        
        # 找到所有开仓和平仓位置
        long_entries = np.where(labels == 1)[0]
        long_exits = np.where(labels == 2)[0]
        short_entries = np.where(labels == 3)[0]
        short_exits = np.where(labels == 4)[0]
        
        wins = 0
        total_trades = 0
        
        # 计算做多交易
        for entry_idx in long_entries:
            # 找到对应的平仓
            exit_candidates = long_exits[long_exits > entry_idx]
            if len(exit_candidates) > 0:
                exit_idx = exit_candidates[0]
                entry_price = prices[entry_idx]
                exit_price = prices[exit_idx]
                profit = (exit_price - entry_price) / entry_price
                
                if profit > 0:
                    wins += 1
                total_trades += 1
        
        # 计算做空交易
        for entry_idx in short_entries:
            # 找到对应的平仓
            exit_candidates = short_exits[short_exits > entry_idx]
            if len(exit_candidates) > 0:
                exit_idx = exit_candidates[0]
                entry_price = prices[entry_idx]
                exit_price = prices[exit_idx]
                profit = (entry_price - exit_price) / entry_price
                
                if profit > 0:
                    wins += 1
                total_trades += 1
        
        if total_trades > 0:
            return wins / total_trades
        else:
            return 0.0
            
    except Exception as e:
        print(f"   模拟胜率计算出错: {e}")
        return 0.0

if __name__ == "__main__":
    print("🧪 宽松标签数据快速测试工具")
    print("="*50)
    
    success = quick_test_relaxed_labels()
    
    if success:
        print(f"\n✅ 测试完成！")
        print(f"💡 如果效果满意，请运行: python predict_improved.py")
        print(f"🔄 如果需要重新训练模型，请参考模型训练脚本")
    else:
        print(f"\n❌ 测试失败，请检查数据生成过程")