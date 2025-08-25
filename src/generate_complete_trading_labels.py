"""
完整交易标签生成工具
确保生成的标签包含完整的开仓+平仓交易对
解决只有开仓没有平仓的问题
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt

def generate_complete_trading_labels(df, method="profit_target", **kwargs):
    """
    生成完整的交易标签，确保每个开仓都有对应的平仓
    
    标签定义：
    0: 无操作
    1: 做多开仓  
    2: 做多平仓
    3: 做空开仓
    4: 做空平仓
    """
    
    if method == "profit_target":
        return generate_profit_target_labels(df, **kwargs)
    elif method == "time_based":
        return generate_time_based_labels(df, **kwargs)
    elif method == "technical_indicator":
        return generate_technical_indicator_labels(df, **kwargs)
    elif method == "adaptive_exit":
        return generate_adaptive_exit_labels(df, **kwargs)
    else:
        raise ValueError(f"未知的标签生成方法: {method}")

def generate_profit_target_labels(df, min_profit_target=0.008, optimal_profit=0.015, 
                                 stop_loss=0.005, min_hold_time=15, max_hold_time=120, 
                                 min_signal_gap=30):
    """
    基于趋势最大段的完整交易标签生成
    
    Parameters:
    - min_profit_target: 最小盈利目标 (默认0.8%，覆盖手续费)
    - optimal_profit: 理想盈利目标 (默认1.5%，趋势段目标)
    - stop_loss: 止损比例 (默认0.5%)
    - min_hold_time: 最小持仓时间 (默认15个时间点，避免快进快出)
    - max_hold_time: 最大持仓时间 (默认120个时间点)
    - min_signal_gap: 信号之间最小间隔 (默认30个时间点，避免频繁交易)
    """
    values = df['index_value'].values
    labels = [0] * len(values)
    
    print(f"📊 使用趋势最大段方法生成完整交易标签...")
    print(f"   参数: min_profit_target={min_profit_target}, optimal_profit={optimal_profit}")
    print(f"   stop_loss={stop_loss}, min_hold_time={min_hold_time}")
    print(f"   max_hold_time={max_hold_time}, min_signal_gap={min_signal_gap}")
    
    i = 0
    complete_trades = []
    
    while i < len(values) - max_hold_time:
        current_price = values[i]
        
        # 检查是否可以开仓（距离上次信号足够远）
        if i > 0:
            recent_signals = any(labels[max(0, i-min_signal_gap):i])
            if recent_signals:
                i += 1
                continue
        
        # 寻找做多趋势段机会
        long_entry_idx, long_exit_idx, long_exit_type, long_max_profit = find_trend_based_long_trade(
            values, i, min_profit_target, optimal_profit, stop_loss, min_hold_time, max_hold_time)
        
        # 寻找做空趋势段机会  
        short_entry_idx, short_exit_idx, short_exit_type, short_max_profit = find_trend_based_short_trade(
            values, i, min_profit_target, optimal_profit, stop_loss, min_hold_time, max_hold_time)
        
        # 选择更好的交易机会（考虑最大盈利潜力）
        long_profit = long_max_profit if long_exit_idx is not None else 0
        short_profit = short_max_profit if short_exit_idx is not None else 0
        
        # 执行交易
        if long_profit > short_profit and long_profit > 0:
            # 做多交易
            labels[long_entry_idx] = 1  # 做多开仓
            labels[long_exit_idx] = 2   # 做多平仓
            complete_trades.append({
                'type': 'long',
                'entry_idx': long_entry_idx,
                'exit_idx': long_exit_idx,
                'entry_price': values[long_entry_idx],
                'exit_price': values[long_exit_idx],
                'profit': long_profit,
                'exit_reason': long_exit_type
            })
            i = long_exit_idx + min_signal_gap
            
        elif short_profit > 0:
            # 做空交易
            labels[short_entry_idx] = 3  # 做空开仓  
            labels[short_exit_idx] = 4   # 做空平仓
            complete_trades.append({
                'type': 'short',
                'entry_idx': short_entry_idx,
                'exit_idx': short_exit_idx,
                'entry_price': values[short_entry_idx],
                'exit_price': values[short_exit_idx],
                'profit': short_profit,
                'exit_reason': short_exit_type
            })
            i = short_exit_idx + min_signal_gap
        else:
            i += 1
    
    # 统计结果
    long_entries = sum(1 for label in labels if label == 1)
    long_exits = sum(1 for label in labels if label == 2)
    short_entries = sum(1 for label in labels if label == 3)
    short_exits = sum(1 for label in labels if label == 4)
    
    print(f"📈 生成结果:")
    print(f"   做多开仓: {long_entries} 个")
    print(f"   做多平仓: {long_exits} 个") 
    print(f"   做空开仓: {short_entries} 个")
    print(f"   做空平仓: {short_exits} 个")
    print(f"   完整交易数: {len(complete_trades)} 个")
    
    # 验证完整性
    if long_entries == long_exits and short_entries == short_exits:
        print("✅ 交易标签完整性验证通过！")
    else:
        print(f"❌ 交易标签不完整！开仓={long_entries + short_entries}, 平仓={long_exits + short_exits}")
    
    # 分析交易质量
    if complete_trades:
        profits = [trade['profit'] for trade in complete_trades]
        win_rate = sum(1 for p in profits if p > 0) / len(profits)
        avg_profit = np.mean(profits)
        
        print(f"   胜率: {win_rate:.2%}")
        print(f"   平均收益: {avg_profit:.4f}")
        
        # 分析平仓原因
        exit_reasons = Counter([trade['exit_reason'] for trade in complete_trades])
        print(f"   平仓原因分布: {dict(exit_reasons)}")
    
    df['label'] = labels
    return df

def find_trend_based_long_trade(values, start_idx, min_profit_target, optimal_profit, 
                               stop_loss, min_hold_time, max_hold_time):
    """
    基于趋势寻找做多交易的最佳平仓点
    策略：寻找趋势的峰值，而不是快进快出
    """
    if start_idx >= len(values) - min_hold_time:
        return start_idx, None, None, 0
        
    entry_price = values[start_idx]
    max_profit_seen = 0
    best_exit_idx = None
    best_exit_reason = None
    
    # 第一阶段：必须持有最小时间
    for i in range(start_idx + 1, min(start_idx + min_hold_time, len(values))):
        current_price = values[i]
        profit_rate = (current_price - entry_price) / entry_price
        
        # 更新最大盈利
        if profit_rate > max_profit_seen:
            max_profit_seen = profit_rate
            
        # 严格止损（亏损过大必须退出）
        if profit_rate <= -stop_loss * 1.5:  # 1.5倍止损线
            return start_idx, i, 'strict_stop_loss', profit_rate
    
    # 第二阶段：寻找趋势峰值
    peak_price = entry_price
    peak_idx = start_idx
    consecutive_down_count = 0
    
    for i in range(start_idx + min_hold_time, min(start_idx + max_hold_time, len(values))):
        current_price = values[i]
        profit_rate = (current_price - entry_price) / entry_price
        
        # 更新峰值
        if current_price > peak_price:
            peak_price = current_price
            peak_idx = i
            consecutive_down_count = 0
            max_profit_seen = max(max_profit_seen, profit_rate)
        else:
            consecutive_down_count += 1
        
        # 达到理想盈利且开始回调，考虑平仓
        if max_profit_seen >= optimal_profit and consecutive_down_count >= 3:
            # 从峰值回调超过20%，平仓
            drawdown = (peak_price - current_price) / peak_price
            if drawdown > 0.2:
                return start_idx, i, 'trend_reversal', profit_rate
        
        # 达到最小盈利目标且趋势明显反转
        elif max_profit_seen >= min_profit_target and consecutive_down_count >= 5:
            # 从峰值回调超过30%，平仓
            drawdown = (peak_price - current_price) / peak_price
            if drawdown > 0.3:
                return start_idx, i, 'min_profit_exit', profit_rate
        
        # 止损
        if profit_rate <= -stop_loss:
            return start_idx, i, 'stop_loss', profit_rate
    
    # 时间到期，检查是否盈利
    final_idx = start_idx + max_hold_time - 1
    if final_idx < len(values):
        final_profit = (values[final_idx] - entry_price) / entry_price
        if final_profit >= min_profit_target:
            return start_idx, final_idx, 'time_exit_profit', final_profit
        elif max_profit_seen >= min_profit_target:
            # 曾经盈利过，在峰值附近退出
            return start_idx, peak_idx, 'peak_exit', (peak_price - entry_price) / entry_price
    
    return start_idx, None, None, 0

def find_trend_based_short_trade(values, start_idx, min_profit_target, optimal_profit,
                                stop_loss, min_hold_time, max_hold_time):
    """
    基于趋势寻找做空交易的最佳平仓点
    策略：寻找趋势的谷底，而不是快进快出
    """
    if start_idx >= len(values) - min_hold_time:
        return start_idx, None, None, 0
        
    entry_price = values[start_idx]
    max_profit_seen = 0
    best_exit_idx = None
    best_exit_reason = None
    
    # 第一阶段：必须持有最小时间
    for i in range(start_idx + 1, min(start_idx + min_hold_time, len(values))):
        current_price = values[i]
        profit_rate = (entry_price - current_price) / entry_price
        
        # 更新最大盈利
        if profit_rate > max_profit_seen:
            max_profit_seen = profit_rate
            
        # 严格止损（亏损过大必须退出）
        if profit_rate <= -stop_loss * 1.5:  # 1.5倍止损线
            return start_idx, i, 'strict_stop_loss', profit_rate
    
    # 第二阶段：寻找趋势谷底
    trough_price = entry_price
    trough_idx = start_idx
    consecutive_up_count = 0
    
    for i in range(start_idx + min_hold_time, min(start_idx + max_hold_time, len(values))):
        current_price = values[i]
        profit_rate = (entry_price - current_price) / entry_price
        
        # 更新谷底
        if current_price < trough_price:
            trough_price = current_price
            trough_idx = i
            consecutive_up_count = 0
            max_profit_seen = max(max_profit_seen, profit_rate)
        else:
            consecutive_up_count += 1
        
        # 达到理想盈利且开始反弹，考虑平仓
        if max_profit_seen >= optimal_profit and consecutive_up_count >= 3:
            # 从谷底反弹超过20%，平仓
            rebound = (current_price - trough_price) / trough_price
            if rebound > 0.2:
                return start_idx, i, 'trend_reversal', profit_rate
        
        # 达到最小盈利目标且趋势明显反转
        elif max_profit_seen >= min_profit_target and consecutive_up_count >= 5:
            # 从谷底反弹超过30%，平仓
            rebound = (current_price - trough_price) / trough_price
            if rebound > 0.3:
                return start_idx, i, 'min_profit_exit', profit_rate
        
        # 止损
        if profit_rate <= -stop_loss:
            return start_idx, i, 'stop_loss', profit_rate
    
    # 时间到期，检查是否盈利
    final_idx = start_idx + max_hold_time - 1
    if final_idx < len(values):
        final_profit = (entry_price - values[final_idx]) / entry_price
        if final_profit >= min_profit_target:
            return start_idx, final_idx, 'time_exit_profit', final_profit
        elif max_profit_seen >= min_profit_target:
            # 曾经盈利过，在谷底附近退出
            return start_idx, trough_idx, 'trough_exit', (entry_price - trough_price) / entry_price
    
    return start_idx, None, None, 0

def generate_technical_indicator_labels(df, rsi_period=14, ma_short=5, ma_long=20, 
                                      profit_target=0.004, max_hold_time=25):
    """
    基于技术指标的完整交易标签生成
    """
    values = df['index_value'].values
    labels = [0] * len(values)
    
    print(f"📊 使用技术指标方法生成完整交易标签...")
    
    # 计算技术指标
    prices_series = pd.Series(values)
    
    # RSI
    delta = prices_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 移动平均
    ma_short_series = prices_series.rolling(window=ma_short).mean()
    ma_long_series = prices_series.rolling(window=ma_long).mean()
    
    complete_trades = []
    i = max(rsi_period, ma_long)  # 确保指标有效
    
    while i < len(values) - max_hold_time:
        current_rsi = rsi.iloc[i]
        current_ma_short = ma_short_series.iloc[i]
        current_ma_long = ma_long_series.iloc[i]
        current_price = values[i]
        
        # 做多信号：RSI超卖 + 短均线上穿长均线
        if (current_rsi < 30 and current_ma_short > current_ma_long and 
            not pd.isna(current_rsi)):
            
            exit_idx = find_technical_exit(values, rsi, ma_short_series, ma_long_series, 
                                         i, 'long', profit_target, max_hold_time)
            if exit_idx is not None:
                labels[i] = 1      # 做多开仓
                labels[exit_idx] = 2  # 做多平仓
                
                profit = (values[exit_idx] - values[i]) / values[i]
                complete_trades.append({
                    'type': 'long',
                    'entry_idx': i,
                    'exit_idx': exit_idx,
                    'profit': profit
                })
                
                i = exit_idx + 5  # 间隔
                continue
        
        # 做空信号：RSI超买 + 短均线下穿长均线
        elif (current_rsi > 70 and current_ma_short < current_ma_long and 
              not pd.isna(current_rsi)):
            
            exit_idx = find_technical_exit(values, rsi, ma_short_series, ma_long_series,
                                         i, 'short', profit_target, max_hold_time)
            if exit_idx is not None:
                labels[i] = 3      # 做空开仓
                labels[exit_idx] = 4  # 做空平仓
                
                profit = (values[i] - values[exit_idx]) / values[i]
                complete_trades.append({
                    'type': 'short',
                    'entry_idx': i,
                    'exit_idx': exit_idx,
                    'profit': profit
                })
                
                i = exit_idx + 5  # 间隔
                continue
        
        i += 1
    
    # 统计结果
    print_trading_stats(labels, complete_trades)
    
    df['label'] = labels
    return df

def find_technical_exit(values, rsi, ma_short, ma_long, entry_idx, trade_type, 
                       profit_target, max_hold_time):
    """
    基于技术指标寻找平仓点
    """
    entry_price = values[entry_idx]
    
    for i in range(entry_idx + 1, min(entry_idx + max_hold_time, len(values))):
        current_price = values[i]
        current_rsi = rsi.iloc[i] if i < len(rsi) else None
        current_ma_short = ma_short.iloc[i] if i < len(ma_short) else None
        current_ma_long = ma_long.iloc[i] if i < len(ma_long) else None
        
        if pd.isna(current_rsi) or pd.isna(current_ma_short) or pd.isna(current_ma_long):
            continue
            
        if trade_type == 'long':
            profit_rate = (current_price - entry_price) / entry_price
            
            # 平仓条件：止盈 或 RSI超买 或 均线死叉
            if (profit_rate >= profit_target or 
                current_rsi > 70 or 
                current_ma_short < current_ma_long):
                return i
                
        elif trade_type == 'short':
            profit_rate = (entry_price - current_price) / entry_price
            
            # 平仓条件：止盈 或 RSI超卖 或 均线金叉
            if (profit_rate >= profit_target or
                current_rsi < 30 or 
                current_ma_short > current_ma_long):
                return i
    
    # 时间止损
    return entry_idx + max_hold_time - 1 if entry_idx + max_hold_time < len(values) else None

def generate_adaptive_exit_labels(df, entry_threshold=0.002, trailing_stop=0.001, 
                                 max_hold_time=40, min_profit=0.001):
    """
    自适应出场的完整交易标签生成
    使用追踪止损等高级技术
    """
    values = df['index_value'].values
    labels = [0] * len(values)
    
    print(f"📊 使用自适应出场方法生成完整交易标签...")
    
    complete_trades = []
    i = 10  # 留出一些历史数据
    
    while i < len(values) - max_hold_time:
        # 计算近期波动率作为入场信号
        recent_volatility = np.std(values[max(0, i-10):i]) / np.mean(values[max(0, i-10):i])
        
        if recent_volatility > entry_threshold:
            # 预测未来趋势方向
            short_trend = np.mean(values[i-5:i]) - np.mean(values[i-10:i-5])
            
            if short_trend > 0:  # 上升趋势，做多
                exit_idx, exit_reason = find_adaptive_exit(values, i, 'long', trailing_stop, max_hold_time, min_profit)
                if exit_idx is not None:
                    labels[i] = 1         # 做多开仓
                    labels[exit_idx] = 2   # 做多平仓
                    
                    profit = (values[exit_idx] - values[i]) / values[i]
                    complete_trades.append({
                        'type': 'long',
                        'entry_idx': i,
                        'exit_idx': exit_idx,
                        'profit': profit,
                        'exit_reason': exit_reason
                    })
                    
                    i = exit_idx + 3
                    continue
            
            elif short_trend < 0:  # 下降趋势，做空
                exit_idx, exit_reason = find_adaptive_exit(values, i, 'short', trailing_stop, max_hold_time, min_profit)
                if exit_idx is not None:
                    labels[i] = 3         # 做空开仓  
                    labels[exit_idx] = 4   # 做空平仓
                    
                    profit = (values[i] - values[exit_idx]) / values[i]
                    complete_trades.append({
                        'type': 'short',
                        'entry_idx': i,
                        'exit_idx': exit_idx,
                        'profit': profit,
                        'exit_reason': exit_reason
                    })
                    
                    i = exit_idx + 3
                    continue
        
        i += 1
    
    print_trading_stats(labels, complete_trades)
    
    df['label'] = labels
    return df

def find_adaptive_exit(values, entry_idx, trade_type, trailing_stop, max_hold_time, min_profit):
    """
    自适应寻找平仓点，使用追踪止损
    """
    entry_price = values[entry_idx]
    best_price = entry_price
    trail_stop_price = entry_price
    
    for i in range(entry_idx + 1, min(entry_idx + max_hold_time, len(values))):
        current_price = values[i]
        
        if trade_type == 'long':
            # 更新最高价和追踪止损价
            if current_price > best_price:
                best_price = current_price
                trail_stop_price = best_price * (1 - trailing_stop)
            
            # 检查平仓条件
            profit_rate = (current_price - entry_price) / entry_price
            
            if current_price <= trail_stop_price and profit_rate >= min_profit:
                return i, 'trailing_stop'
            
        elif trade_type == 'short':
            # 更新最低价和追踪止损价
            if current_price < best_price:
                best_price = current_price
                trail_stop_price = best_price * (1 + trailing_stop)
            
            # 检查平仓条件
            profit_rate = (entry_price - current_price) / entry_price
            
            if current_price >= trail_stop_price and profit_rate >= min_profit:
                return i, 'trailing_stop'
    
    # 时间止损
    return entry_idx + max_hold_time - 1, 'time_stop'

def print_trading_stats(labels, complete_trades):
    """
    打印交易统计信息
    """
    long_entries = sum(1 for label in labels if label == 1)
    long_exits = sum(1 for label in labels if label == 2)
    short_entries = sum(1 for label in labels if label == 3)
    short_exits = sum(1 for label in labels if label == 4)
    
    print(f"📈 生成结果:")
    print(f"   做多开仓: {long_entries} 个")
    print(f"   做多平仓: {long_exits} 个")
    print(f"   做空开仓: {short_entries} 个") 
    print(f"   做空平仓: {short_exits} 个")
    print(f"   完整交易数: {len(complete_trades)} 个")
    
    # 验证完整性
    if long_entries == long_exits and short_entries == short_exits:
        print("✅ 交易标签完整性验证通过！")
    else:
        print(f"❌ 交易标签不完整！")
    
    # 分析交易质量
    if complete_trades:
        profits = [trade['profit'] for trade in complete_trades]
        win_rate = sum(1 for p in profits if p > 0) / len(profits)
        avg_profit = np.mean(profits)
        
        print(f"   胜率: {win_rate:.2%}")
        print(f"   平均收益: {avg_profit:.4f}")

def regenerate_complete_trading_labels(data_dir="../data/", 
                                     output_dir="../data_with_complete_labels/",
                                     method="profit_target"):
    """
    重新生成所有文件的完整交易标签
    """
    print(f"🔄 开始生成完整交易标签...")
    print(f"   方法: {method}")
    
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
        'total_complete_trades': 0,
        'total_signals': 0,
        'avg_win_rate': 0
    }
    
    for csv_file in csv_files:
        try:
            print(f"\n{'='*60}")
            print(f"处理文件: {os.path.basename(csv_file)}")
            
            # 读取数据
            df = pd.read_csv(csv_file)
            
            # 根据方法生成完整交易标签
            if method == "profit_target":
                df_with_labels = generate_complete_trading_labels(
                    df, method="profit_target",
                    min_profit_target=0.008,  # 0.8% 最小盈利（覆盖手续费）
                    optimal_profit=0.015,     # 1.5% 理想盈利
                    stop_loss=0.005,          # 0.5% 止损
                    min_hold_time=15,         # 最小持仓15个时间点
                    max_hold_time=80,         # 最大持仓80个时间点
                    min_signal_gap=25         # 信号间隔25个时间点
                )
            elif method == "technical_indicator":
                df_with_labels = generate_complete_trading_labels(
                    df, method="technical_indicator",
                    profit_target=0.003,
                    max_hold_time=20
                )
            elif method == "adaptive_exit":
                df_with_labels = generate_complete_trading_labels(
                    df, method="adaptive_exit",
                    entry_threshold=0.001,
                    trailing_stop=0.001,
                    max_hold_time=30
                )
            else:
                print(f"未知方法: {method}")
                continue
            
            # 保存新文件
            output_file = os.path.join(output_dir, os.path.basename(csv_file))
            df_with_labels.to_csv(output_file, index=False)
            
            # 统计信息
            labels = df_with_labels['label'].values
            num_signals = np.sum(labels != 0)
            num_complete_trades = (np.sum(labels == 1) + np.sum(labels == 3))  # 开仓次数
            
            total_stats['total_files'] += 1
            total_stats['total_signals'] += num_signals
            total_stats['total_complete_trades'] += num_complete_trades
            
            print(f"✅ 完成: {num_complete_trades} 个完整交易, {num_signals} 个信号")
            
        except Exception as e:
            print(f"❌ 处理文件 {csv_file} 时出错: {e}")
            continue
    
    # 计算总体统计
    if total_stats['total_files'] > 0:
        print(f"\n{'='*60}")
        print(f"📊 完整交易标签生成完成！总体统计:")
        print(f"   处理文件数: {total_stats['total_files']}")
        print(f"   总完整交易数: {total_stats['total_complete_trades']}")
        print(f"   总信号数: {total_stats['total_signals']}")
        print(f"   平均每文件交易数: {total_stats['total_complete_trades']/total_stats['total_files']:.1f}")
        print(f"   完整交易标签文件保存在: {output_dir}")

if __name__ == "__main__":
    print("🚀 完整交易标签生成工具")
    print("="*50)
    
    methods = [
        "profit_target",      # 止盈止损方法（推荐）
        "technical_indicator", # 技术指标方法
        "adaptive_exit"       # 自适应出场方法
    ]
    
    print("可选的完整交易标签生成方法：")
    for i, method in enumerate(methods):
        print(f"  {i+1}. {method}")
    
    # 默认使用推荐方法（趋势最大段）
    selected_method = "profit_target"
    
    print(f"\n使用方法: {selected_method} (趋势最大段策略)")
    print(f"策略说明: 捕获趋势的峰值/谷底，避免快进快出，确保交易有足够盈利空间")
    
    # 生成完整交易标签
    regenerate_complete_trading_labels(
        data_dir="../data/",
        output_dir="../data_with_complete_labels/",
        method=selected_method
    )
    
    print(f"\n✅ 完整交易标签生成完成！")
    print(f"新的完整交易标签文件已保存在 ../data_with_complete_labels/ 目录")
    print(f"策略特点:")
    print(f"  ✅ 每个开仓信号都有对应的平仓信号")
    print(f"  ✅ 最小持仓时间15个点，避免快进快出")
    print(f"  ✅ 目标盈利0.8%-1.5%，覆盖手续费成本")
    print(f"  ✅ 捕获趋势峰值/谷底，最大化盈利")
    print(f"  ✅ 信号间隔25个点，避免过度交易")