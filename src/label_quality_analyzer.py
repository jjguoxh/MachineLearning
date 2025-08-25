"""
标签质量分析工具
用于诊断当前标签生成策略的问题，并提供改进建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter

def analyze_label_quality(csv_file):
    """
    分析单个文件的标签质量
    """
    print(f"\n分析文件: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if 'label' not in df.columns:
        print("❌ 文件中没有label列")
        return None
    
    labels = df['label'].values
    index_values = df['index_value'].values
    
    # 1. 标签分布分析
    label_counts = Counter(labels)
    total_labels = len(labels)
    
    print("📊 标签分布:")
    label_names = {0: '无操作', 1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
    for label, count in sorted(label_counts.items()):
        percentage = count / total_labels * 100
        label_name = label_names.get(label, f'未知标签{label}')
        print(f"   {label_name}: {count} ({percentage:.1f}%)")
    
    # 2. 标签连续性分析
    label_changes = np.diff(labels != 0).sum()
    signal_density = np.sum(labels != 0) / total_labels
    print(f"\n🔄 标签连续性:")
    print(f"   信号变化次数: {label_changes}")
    print(f"   信号密度: {signal_density:.3f}")
    
    # 3. 标签有效性分析（检查标签是否与价格变化一致）
    analyze_label_effectiveness(labels, index_values)
    
    # 4. 标签时间分布
    analyze_label_timing(labels)
    
    return {
        'file': csv_file,
        'label_distribution': label_counts,
        'signal_density': signal_density,
        'label_changes': label_changes
    }

def analyze_label_effectiveness(labels, prices):
    """
    分析标签的有效性 - 检查标签是否与实际价格变化一致
    """
    print(f"\n✅ 标签有效性分析:")
    
    # 找到所有交易信号
    long_entries = np.where(labels == 1)[0]
    long_exits = np.where(labels == 2)[0]
    short_entries = np.where(labels == 3)[0]
    short_exits = np.where(labels == 4)[0]
    
    def analyze_signal_effectiveness(entry_points, exit_points, signal_type):
        if len(entry_points) == 0:
            print(f"   {signal_type}: 无信号")
            return
        
        profits = []
        for entry_idx in entry_points:
            # 找到最近的平仓点
            future_exits = exit_points[exit_points > entry_idx]
            if len(future_exits) > 0:
                exit_idx = future_exits[0]
                if signal_type == "做多":
                    profit = (prices[exit_idx] - prices[entry_idx]) / prices[entry_idx]
                else:  # 做空
                    profit = (prices[entry_idx] - prices[exit_idx]) / prices[entry_idx]
                profits.append(profit)
        
        if profits:
            win_rate = np.sum(np.array(profits) > 0) / len(profits)
            avg_profit = np.mean(profits)
            print(f"   {signal_type}: {len(profits)}个交易, 胜率: {win_rate:.2%}, 平均收益: {avg_profit:.4f}")
        else:
            print(f"   {signal_type}: 无完整交易")
    
    analyze_signal_effectiveness(long_entries, long_exits, "做多")
    analyze_signal_effectiveness(short_entries, short_exits, "做空")

def analyze_label_timing(labels):
    """
    分析标签的时间分布
    """
    print(f"\n⏰ 标签时间分布:")
    
    # 计算信号间隔
    signal_indices = np.where(labels != 0)[0]
    if len(signal_indices) > 1:
        intervals = np.diff(signal_indices)
        print(f"   平均信号间隔: {np.mean(intervals):.1f} 个时间点")
        print(f"   最小信号间隔: {np.min(intervals)} 个时间点")
        print(f"   最大信号间隔: {np.max(intervals)} 个时间点")
    
    # 检查信号聚集情况
    consecutive_signals = 0
    for i in range(1, len(labels)):
        if labels[i] != 0 and labels[i-1] != 0:
            consecutive_signals += 1
    
    print(f"   连续信号数: {consecutive_signals}")

def plot_label_analysis(csv_file, output_dir="../analysis/"):
    """
    绘制标签分析图表
    """
    df = pd.read_csv(csv_file)
    if 'label' not in df.columns:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: 价格和标签
    ax1.plot(df['index_value'], label='Price', alpha=0.7)
    
    # 标记不同的标签点
    colors = {1: 'red', 2: 'red', 3: 'green', 4: 'green'}
    markers = {1: '^', 2: 'v', 3: '^', 4: 'v'}
    labels_map = {1: 'Long Entry', 2: 'Long Exit', 3: 'Short Entry', 4: 'Short Exit'}
    
    for label_type in [1, 2, 3, 4]:
        indices = np.where(df['label'] == label_type)[0]
        if len(indices) > 0:
            ax1.scatter(indices, df['index_value'].iloc[indices], 
                       color=colors[label_type], marker=markers[label_type], 
                       s=50, label=labels_map[label_type], alpha=0.8)
    
    ax1.set_title('价格图表与标签分布')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 标签分布饼图
    label_counts = df['label'].value_counts()
    label_names = {0: '无操作', 1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
    labels_for_pie = [label_names.get(i, f'标签{i}') for i in label_counts.index]
    
    ax2.pie(label_counts.values, labels=labels_for_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('标签分布')
    
    # 子图3: 信号时间分布
    signal_mask = df['label'] != 0
    signal_positions = np.where(signal_mask)[0]
    if len(signal_positions) > 0:
        ax3.hist(signal_positions, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('信号时间分布')
        ax3.set_xlabel('时间点')
        ax3.set_ylabel('信号数量')
        ax3.grid(True, alpha=0.3)
    
    # 子图4: 价格变化vs标签
    price_changes = df['index_value'].pct_change().fillna(0)
    for label_type in [1, 2, 3, 4]:
        label_indices = df['label'] == label_type
        if label_indices.sum() > 0:
            ax4.scatter(np.where(label_indices)[0], price_changes[label_indices], 
                       color=colors[label_type], label=labels_map[label_type], alpha=0.6)
    
    ax4.set_title('标签点的价格变化')
    ax4.set_xlabel('时间点')
    ax4.set_ylabel('价格变化率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_label_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📈 分析图表已保存: {output_file}")
    plt.close()

def comprehensive_label_analysis(data_dir="../data_with_complete_labels/"):
    """
    对所有数据文件进行综合标签分析
    """
    print("🔍 开始综合标签质量分析...")
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ 在目录 {data_dir} 中未找到CSV文件")
        return
    
    print(f"📁 找到 {len(csv_files)} 个CSV文件")
    
    all_results = []
    
    for csv_file in csv_files:
        try:
            result = analyze_label_quality(csv_file)
            if result:
                all_results.append(result)
                # 生成可视化分析
                plot_label_analysis(csv_file)
        except Exception as e:
            print(f"❌ 分析文件 {csv_file} 时出错: {e}")
    
    # 综合统计
    if all_results:
        print(f"\n📋 综合分析结果 (共{len(all_results)}个文件):")
        
        # 平均信号密度
        avg_signal_density = np.mean([r['signal_density'] for r in all_results])
        print(f"   平均信号密度: {avg_signal_density:.3f}")
        
        # 标签分布统计
        all_label_counts = Counter()
        for result in all_results:
            all_label_counts.update(result['label_distribution'])
        
        total_all_labels = sum(all_label_counts.values())
        print(f"   整体标签分布:")
        label_names = {0: '无操作', 1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
        for label, count in sorted(all_label_counts.items()):
            percentage = count / total_all_labels * 100
            label_name = label_names.get(label, f'未知标签{label}')
            print(f"     {label_name}: {count} ({percentage:.1f}%)")
    
    # 给出改进建议
    provide_improvement_suggestions(all_results)

def provide_improvement_suggestions(results):
    """
    基于分析结果提供改进建议
    """
    print(f"\n💡 改进建议:")
    
    if not results:
        print("   无法提供建议，请检查数据文件")
        return
    
    # 分析信号密度
    avg_signal_density = np.mean([r['signal_density'] for r in results])
    
    if avg_signal_density < 0.05:
        print("   🔸 信号密度过低(<5%)，建议：")
        print("     - 降低标签生成的阈值")
        print("     - 使用更敏感的信号检测方法")
        print("     - 考虑使用更短的时间窗口")
    elif avg_signal_density > 0.3:
        print("   🔸 信号密度过高(>30%)，建议：")
        print("     - 提高标签生成的阈值")
        print("     - 增加信号过滤条件")
        print("     - 使用更长的时间窗口")
    else:
        print("   ✅ 信号密度适中 (5%-30%)")
    
    # 分析标签平衡性
    all_label_counts = Counter()
    for result in results:
        all_label_counts.update(result['label_distribution'])
    
    # 排除无操作标签
    signal_labels = {k: v for k, v in all_label_counts.items() if k != 0}
    if signal_labels:
        max_count = max(signal_labels.values())
        min_count = min(signal_labels.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 5:
            print("   🔸 标签严重不平衡，建议：")
            print("     - 使用SMOTE等过采样技术")
            print("     - 调整标签生成策略使各类别更平衡")
            print("     - 使用加权损失函数")
        elif imbalance_ratio > 2:
            print("   🔸 标签轻微不平衡，建议：")
            print("     - 使用类别权重")
            print("     - 考虑Focal Loss")
        else:
            print("   ✅ 标签分布相对平衡")
    
    print("\n🚀 下一步行动建议：")
    print("   1. 运行 predict_improved.py 查看当前模型效果")
    print("   2. 根据分析结果调整标签生成参数")
    print("   3. 增加技术指标特征")
    print("   4. 使用改进的模型架构")
    print("   5. 实施集成学习方法")

if __name__ == "__main__":
    # 运行综合分析
    comprehensive_label_analysis()
    
    print(f"\n{'='*60}")
    print("标签质量分析完成！")
    print("请查看生成的分析图表和建议，然后针对性地改进模型。")