"""
改进的标签重新生成工具
针对信号密度过低的问题，提供多种降低阈值的方法
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt

def regenerate_labels_low_threshold(df, method="balanced_percentile", **kwargs):
    """
    使用更低阈值重新生成标签
    
    Parameters:
    - method: 标签生成方法
        - "balanced_percentile": 使用分位数方法（推荐）
        - "fixed_threshold": 使用固定阈值方法
        - "adaptive_threshold": 使用自适应阈值方法
        - "multi_level": 多层次标签生成
    """
    
    if method == "balanced_percentile":
        return generate_label_balanced_improved(df, **kwargs)
    elif method == "fixed_threshold":
        return generate_label_fixed_threshold(df, **kwargs)
    elif method == "adaptive_threshold":
        return generate_label_adaptive_threshold(df, **kwargs)
    elif method == "multi_level":
        return generate_label_multi_level(df, **kwargs)
    else:
        raise ValueError(f"未知的标签生成方法: {method}")

def generate_label_balanced_improved(df, window_size=60, percentile=85, min_signal_density=0.05):
    """
    改进的分位数标签生成方法
    - 使用更短的窗口和更高的分位数来增加信号密度
    - 动态调整分位数直到达到最小信号密度
    """
    values = df['index_value'].values
    
    print(f"📊 使用改进分位数方法生成标签...")
    print(f"   初始参数: window_size={window_size}, percentile={percentile}")
    
    # 尝试不同的分位数设置
    for current_percentile in range(percentile, 50, -5):  # 从高到低尝试
        labels = [0] * len(values)
        returns_list = []
        
        # 计算所有窗口的收益率
        for i in range(len(values) - window_size):
            window_vals = values[i:i+window_size]
            ret = (window_vals[-1] - window_vals[0]) / window_vals[0]
            returns_list.append(ret)
        
        if not returns_list:
            continue
            
        # 计算分位数阈值
        upper_threshold = np.percentile(returns_list, current_percentile)
        lower_threshold = np.percentile(returns_list, 100-current_percentile)
        
        up_trends = 0
        down_trends = 0
        
        # 生成标签
        for i, ret in enumerate(returns_list):
            center = i + window_size // 2
            if ret >= upper_threshold:
                labels[center] = 1  # 上涨趋势
                up_trends += 1
            elif ret <= lower_threshold:
                labels[center] = 2  # 下跌趋势
                down_trends += 1
        
        signal_density = (up_trends + down_trends) / len(labels)
        
        print(f"   尝试 percentile={current_percentile}: 信号密度={signal_density:.4f}")
        
        if signal_density >= min_signal_density:
            print(f"✅ 达到目标信号密度!")
            break
    
    print(f"📈 最终结果:")
    print(f"   上涨标签数: {up_trends}")
    print(f"   下跌标签数: {down_trends}")
    print(f"   信号密度: {signal_density:.4f}")
    print(f"   上涨阈值: {upper_threshold:.6f}")
    print(f"   下跌阈值: {lower_threshold:.6f}")
    
    df['label'] = labels
    return df

def generate_label_fixed_threshold(df, window_size=30, change_threshold=0.003):
    """
    使用更低的固定阈值生成标签
    """
    values = df['index_value'].values
    labels = [0] * len(values)
    
    print(f"📊 使用固定阈值方法生成标签...")
    print(f"   参数: window_size={window_size}, change_threshold={change_threshold}")
    
    up_trends = 0
    down_trends = 0
    
    for i in range(len(values) - window_size):
        window_vals = values[i:i+window_size]
        ret = (window_vals[-1] - window_vals[0]) / window_vals[0]
        
        center = i + window_size // 2
        if ret > change_threshold:
            labels[center] = 1  # 上涨趋势
            up_trends += 1
        elif ret < -change_threshold:
            labels[center] = 2  # 下跌趋势
            down_trends += 1
    
    signal_density = (up_trends + down_trends) / len(labels)
    
    print(f"📈 结果:")
    print(f"   上涨标签数: {up_trends}")
    print(f"   下跌标签数: {down_trends}")
    print(f"   信号密度: {signal_density:.4f}")
    
    df['label'] = labels
    return df

def generate_label_adaptive_threshold(df, window_size=60, target_signal_density=0.10):
    """
    自适应阈值标签生成
    根据目标信号密度自动调整阈值
    """
    values = df['index_value'].values
    
    print(f"📊 使用自适应阈值方法生成标签...")
    print(f"   目标信号密度: {target_signal_density:.1%}")
    
    # 计算所有可能的收益率
    all_returns = []
    for window in [30, 60, 90]:  # 多个窗口大小
        for i in range(len(values) - window):
            ret = (values[i+window] - values[i]) / values[i]
            all_returns.append((ret, i + window//2))
    
    # 按绝对收益率排序
    all_returns.sort(key=lambda x: abs(x[0]), reverse=True)
    
    # 选择前N%作为信号
    num_signals = int(len(all_returns) * target_signal_density)
    selected_returns = all_returns[:num_signals]
    
    labels = [0] * len(values)
    up_trends = 0
    down_trends = 0
    
    for ret, idx in selected_returns:
        if idx < len(labels):
            if ret > 0:
                labels[idx] = 1  # 上涨
                up_trends += 1
            else:
                labels[idx] = 2  # 下跌
                down_trends += 1
    
    actual_density = (up_trends + down_trends) / len(labels)
    
    print(f"📈 结果:")
    print(f"   上涨标签数: {up_trends}")
    print(f"   下跌标签数: {down_trends}")
    print(f"   实际信号密度: {actual_density:.4f}")
    
    df['label'] = labels
    return df

def generate_label_multi_level(df, window_size=60):
    """
    多层次标签生成：同时生成开仓和平仓信号
    """
    values = df['index_value'].values
    labels = [0] * len(values)
    
    print(f"📊 使用多层次方法生成标签...")
    
    # 计算短期和长期移动平均
    short_ma = pd.Series(values).rolling(window=5).mean()
    long_ma = pd.Series(values).rolling(window=20).mean()
    
    # 计算价格相对于移动平均的位置
    price_above_short = values > short_ma
    price_above_long = values > long_ma
    
    # RSI指标
    delta = pd.Series(values).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    entry_signals = 0
    exit_signals = 0
    
    for i in range(20, len(values)-5):  # 确保有足够的历史数据
        current_price = values[i]
        
        # 做多开仓条件
        if (price_above_short.iloc[i] and price_above_long.iloc[i] and 
            rsi.iloc[i] > 30 and rsi.iloc[i] < 70):
            # 检查未来是否有盈利机会
            future_max = np.max(values[i+1:i+6]) if i+6 < len(values) else current_price
            if (future_max - current_price) / current_price > 0.002:  # 0.2%的潜在收益
                labels[i] = 1  # 做多开仓
                entry_signals += 1
        
        # 做多平仓条件
        elif (not price_above_short.iloc[i] or rsi.iloc[i] > 70):
            labels[i] = 2  # 做多平仓
            exit_signals += 1
        
        # 做空开仓条件
        elif (not price_above_short.iloc[i] and not price_above_long.iloc[i] and 
              rsi.iloc[i] < 70 and rsi.iloc[i] > 30):
            # 检查未来是否有盈利机会
            future_min = np.min(values[i+1:i+6]) if i+6 < len(values) else current_price
            if (current_price - future_min) / current_price > 0.002:  # 0.2%的潜在收益
                labels[i] = 3  # 做空开仓
                entry_signals += 1
        
        # 做空平仓条件
        elif (price_above_short.iloc[i] or rsi.iloc[i] < 30):
            labels[i] = 4  # 做空平仓
            exit_signals += 1
    
    total_signals = entry_signals + exit_signals
    signal_density = total_signals / len(labels)
    
    print(f"📈 结果:")
    print(f"   开仓信号数: {entry_signals}")
    print(f"   平仓信号数: {exit_signals}")
    print(f"   总信号数: {total_signals}")
    print(f"   信号密度: {signal_density:.4f}")
    
    df['label'] = labels
    return df

def regenerate_all_labels(data_dir="../data/", output_dir="../data_with_new_labels/", 
                         method="balanced_percentile", **kwargs):
    """
    重新生成所有文件的标签
    """
    print(f"🔄 开始重新生成标签...")
    print(f"   方法: {method}")
    print(f"   参数: {kwargs}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ 在目录 {data_dir} 中未找到CSV文件")
        return
    
    print(f"📁 找到 {len(csv_files)} 个CSV文件")
    
    total_stats = {
        'total_files': 0,
        'total_signals': 0,
        'total_datapoints': 0,
        'avg_density': 0
    }
    
    for csv_file in csv_files:
        try:
            print(f"\n{'='*60}")
            print(f"处理文件: {os.path.basename(csv_file)}")
            
            # 读取数据
            df = pd.read_csv(csv_file)
            
            # 重新生成标签
            df_with_labels = regenerate_labels_low_threshold(df, method=method, **kwargs)
            
            # 保存新文件
            output_file = os.path.join(output_dir, os.path.basename(csv_file))
            df_with_labels.to_csv(output_file, index=False)
            
            # 统计信息
            labels = df_with_labels['label'].values
            num_signals = np.sum(labels != 0)
            signal_density = num_signals / len(labels)
            
            total_stats['total_files'] += 1
            total_stats['total_signals'] += num_signals
            total_stats['total_datapoints'] += len(labels)
            
            print(f"✅ 完成: {num_signals} 个信号, 密度: {signal_density:.4f}")
            
        except Exception as e:
            print(f"❌ 处理文件 {csv_file} 时出错: {e}")
            continue
    
    # 计算总体统计
    if total_stats['total_files'] > 0:
        total_stats['avg_density'] = total_stats['total_signals'] / total_stats['total_datapoints']
        
        print(f"\n{'='*60}")
        print(f"📊 重新生成标签完成！总体统计:")
        print(f"   处理文件数: {total_stats['total_files']}")
        print(f"   总信号数: {total_stats['total_signals']}")
        print(f"   总数据点数: {total_stats['total_datapoints']}")
        print(f"   平均信号密度: {total_stats['avg_density']:.4f} ({total_stats['avg_density']*100:.1f}%)")
        print(f"   新标签文件保存在: {output_dir}")

def compare_label_methods(csv_file, output_dir="../label_comparison/"):
    """
    比较不同标签生成方法的效果
    """
    print(f"🔬 比较不同标签生成方法...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    
    methods = [
        ("balanced_percentile", {"percentile": 85, "min_signal_density": 0.05}),
        ("balanced_percentile", {"percentile": 80, "min_signal_density": 0.08}),
        ("fixed_threshold", {"change_threshold": 0.003}),
        ("fixed_threshold", {"change_threshold": 0.001}),
        ("adaptive_threshold", {"target_signal_density": 0.05}),
        ("adaptive_threshold", {"target_signal_density": 0.10}),
        ("multi_level", {})
    ]
    
    results = []
    
    for method, params in methods:
        try:
            df_test = df.copy()
            df_with_labels = regenerate_labels_low_threshold(df_test, method=method, **params)
            
            labels = df_with_labels['label'].values
            num_signals = np.sum(labels != 0)
            signal_density = num_signals / len(labels)
            
            # 标签分布
            label_counts = Counter(labels)
            
            result = {
                'method': method,
                'params': params,
                'signal_density': signal_density,
                'num_signals': num_signals,
                'label_distribution': dict(label_counts)
            }
            results.append(result)
            
            print(f"   {method} {params}: 密度={signal_density:.4f}")
            
        except Exception as e:
            print(f"   ❌ {method} 失败: {e}")
    
    # 保存比较结果
    comparison_file = os.path.join(output_dir, "method_comparison.txt")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("标签生成方法比较结果\n")
        f.write("="*50 + "\n\n")
        
        for result in results:
            f.write(f"方法: {result['method']}\n")
            f.write(f"参数: {result['params']}\n")
            f.write(f"信号密度: {result['signal_density']:.4f}\n")
            f.write(f"信号数量: {result['num_signals']}\n")
            f.write(f"标签分布: {result['label_distribution']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"📄 比较结果已保存: {comparison_file}")
    
    return results

if __name__ == "__main__":
    print("🚀 标签重新生成工具")
    print("="*50)
    
    # 可以选择不同的方法和参数
    methods_to_try = [
        {
            "name": "改进分位数方法(推荐)",
            "method": "balanced_percentile",
            "params": {"percentile": 80, "min_signal_density": 0.05}
        },
        {
            "name": "低阈值固定方法",
            "method": "fixed_threshold", 
            "params": {"change_threshold": 0.002, "window_size": 30}
        },
        {
            "name": "自适应阈值方法",
            "method": "adaptive_threshold",
            "params": {"target_signal_density": 0.08}
        },
        {
            "name": "多层次信号方法",
            "method": "multi_level",
            "params": {}
        }
    ]
    
    print("可选的标签重新生成方法：")
    for i, method_info in enumerate(methods_to_try):
        print(f"  {i+1}. {method_info['name']}")
    
    # 默认使用推荐方法
    selected_method = methods_to_try[0]
    
    print(f"\n使用方法: {selected_method['name']}")
    
    # 重新生成所有标签
    regenerate_all_labels(
        data_dir="../data/",
        output_dir="../data_with_improved_labels/",
        method=selected_method["method"],
        **selected_method["params"]
    )
    
    print(f"\n✅ 标签重新生成完成！")
    print(f"新的标签文件已保存在 ../data_with_improved_labels/ 目录")
    print(f"请使用新的数据文件重新训练模型以获得更好的预测效果。")