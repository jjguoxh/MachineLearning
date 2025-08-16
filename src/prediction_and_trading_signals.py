# prediction_and_trading_signals.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from model import TransformerClassifier
from feature_engineering import add_features

# 配置参数
SEQ_LEN = 30
MODEL_PATH = "../model/best_longtrend_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(input_dim, num_classes=5):
    """
    加载训练好的模型
    """
    model = TransformerClassifier(
        input_dim=input_dim,
        model_dim=128,       # 与训练时保持一致
        num_heads=8,         # 与训练时保持一致
        num_layers=4,        # 与训练时保持一致
        num_classes=num_classes,
        dropout=0.1
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载成功")
    return model

def prepare_data_for_prediction(df):
    """
    为预测准备数据
    """
    # 特征工程
    df = add_features(df)
    
    # 准备特征
    exclude_cols = ['label', 'index_value']  # 根据实际情况调整
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features = df[feature_cols].values
    
    # 标准化（需要使用训练时的scaler）
    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # 注意：实际应用中应该加载训练时保存的scaler
    
    return features, df['index_value'].values if 'index_value' in df.columns else None

def create_prediction_sequences(features, seq_len=SEQ_LEN):
    """
    创建预测序列
    """
    X = []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i:i + seq_len])
    return np.array(X)

def generate_trading_signals(model, X, index_values=None):
    """
    根据模型预测生成交易信号（五分类版本）
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # 获取各类别的概率
        prob_no_action = probs[:, 0].cpu().numpy()      # 无操作
        prob_long_entry = probs[:, 1].cpu().numpy()     # 做多开仓
        prob_long_exit = probs[:, 2].cpu().numpy()      # 做多平仓
        prob_short_entry = probs[:, 3].cpu().numpy()    # 做空开仓
        prob_short_exit = probs[:, 4].cpu().numpy()     # 做空平仓
    
    # 生成交易信号
    signals = []
    
    # 只对序列最后一个时间点生成信号
    for i in range(len(preds)):
        # 信号点是序列的最后一个点
        signal_point = i + SEQ_LEN - 1
        signal_info = {
            'point': signal_point,
            'prediction': preds[i],
            'probability': probs[i].cpu().numpy(),  # 所有类别的概率
            'probabilities': {
                'no_action': prob_no_action[i],
                'long_entry': prob_long_entry[i],
                'long_exit': prob_long_exit[i],
                'short_entry': prob_short_entry[i],
                'short_exit': prob_short_exit[i]
            },
            'index_value': index_values[signal_point] if index_values is not None and signal_point < len(index_values) else None
        }
        signals.append(signal_info)
    
    return signals

def identify_trading_points(signals, min_hold_period=5):
    """
    根据信号识别交易点 - 五分类版本
    """
    actions = []  # 记录所有动作
    
    if len(signals) == 0:
        return actions
    
    i = 0
    last_action = None
    
    while i < len(signals):
        signal = signals[i]
        action_type = signal['prediction']
        probabilities = signal['probabilities']
        
        # 获取当前动作类型的概率
        action_prob = signal['probability'][action_type]
        
        # 对于高置信度的交易动作信号进行处理
        if action_type != 0 and action_prob > 0.80:  # 高置信度阈值
            # 避免频繁的相反操作
            if last_action is not None:
                # 如果上一个动作是做多开仓，下一个不应该是做空开仓
                if (last_action == 1 and action_type == 3) or (last_action == 3 and action_type == 1):
                    if action_prob < 0.95:  # 置信度不够高则跳过
                        i += 1
                        continue
            
            action = {
                'point': signal['point'],
                'type': action_type,
                'probability': action_prob,
                'probabilities': probabilities,
                'index_value': signal['index_value'],
                'description': ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓'][action_type]
            }
            actions.append(action)
            last_action = action_type
            i += min_hold_period  # 至少间隔一定时间再考虑下一个动作
        else:
            i += 1
    
    return actions

def apply_risk_management(actions, max_positions=2):
    """
    应用风险管理策略
    """
    # 限制同时持仓数量
    long_positions = 0
    short_positions = 0
    filtered_actions = []
    
    for action in actions:
        action_type = action['type']
        
        if action_type == 1:  # 做多开仓
            if long_positions < max_positions:
                filtered_actions.append(action)
                long_positions += 1
        elif action_type == 2:  # 做多平仓
            if long_positions > 0:
                filtered_actions.append(action)
                long_positions -= 1
        elif action_type == 3:  # 做空开仓
            if short_positions < max_positions:
                filtered_actions.append(action)
                short_positions += 1
        elif action_type == 4:  # 做空平仓
            if short_positions > 0:
                filtered_actions.append(action)
                short_positions -= 1
        else:  # 无操作
            filtered_actions.append(action)
    
    return filtered_actions

def convert_actions_to_trading_points(actions):
    """
    将动作转换为交易点对（开仓-平仓）
    """
    long_entries = []
    long_exits = []
    short_entries = []
    short_exits = []
    
    long_stack = []  # 用于匹配做多开仓和平仓
    short_stack = []  # 用于匹配做空开仓和平仓
    
    for action in actions:
        action_type = action['type']
        
        if action_type == 1:  # 做多开仓
            long_stack.append(action)
        elif action_type == 2:  # 做多平仓
            if long_stack:
                long_entries.append(long_stack.pop())
                long_exits.append(action)
        elif action_type == 3:  # 做空开仓
            short_stack.append(action)
        elif action_type == 4:  # 做空平仓
            if short_stack:
                short_entries.append(short_stack.pop())
                short_exits.append(action)
    
    return long_entries, long_exits, short_entries, short_exits

def save_trading_signals(actions, filename="trading_actions.csv"):
    """
    保存交易动作到CSV文件
    """
    if len(actions) > 0:
        signals_data = []
        for action in actions:
            signals_data.append({
                'point': action['point'],
                'type': action['type'],
                'action': action['description'],
                'probability': action['probability'],
                'index_value': action['index_value']
            })
        
        signals_df = pd.DataFrame(signals_data)
        signals_df = signals_df.sort_values('point')
        signals_df.to_csv(filename, index=False)
        print(f"交易动作已保存到: {filename}")
    else:
        signals_df = pd.DataFrame(columns=['point', 'type', 'action', 'probability', 'index_value'])
        signals_df.to_csv(filename, index=False)
        print("没有交易动作，已保存空的动作文件")
    
    return signals_df

def plot_trading_signals(index_values, actions, filename="trading_signals.png", invert_y=False):
    """
    绘制指数和交易信号
    """
    if len(actions) == 0:
        print("没有交易信号，无法生成图表")
        return
    
    plt.figure(figsize=(15, 8))
    
    # 绘制指数曲线
    plt.plot(index_values, label='Index Value', color='blue', linewidth=1)
    
    # 标记各类动作
    long_entries = [action for action in actions if action['type'] == 1]
    long_exits = [action for action in actions if action['type'] == 2]
    short_entries = [action for action in actions if action['type'] == 3]
    short_exits = [action for action in actions if action['type'] == 4]
    
    # 做多开仓
    if long_entries:
        entry_points = [action['point'] for action in long_entries]
        entry_values = [action['index_value'] for action in long_entries]
        plt.scatter(entry_points, entry_values, color='red', marker='^', s=100, label='Long Entry', zorder=5)
    
    # 做多平仓
    if long_exits:
        exit_points = [action['point'] for action in long_exits]
        exit_values = [action['index_value'] for action in long_exits]
        plt.scatter(exit_points, exit_values, color='darkred', marker='v', s=100, label='Long Exit', zorder=5)
    
    # 做空开仓
    if short_entries:
        entry_points = [action['point'] for action in short_entries]
        entry_values = [action['index_value'] for action in short_entries]
        plt.scatter(entry_points, entry_values, color='green', marker='^', s=100, label='Short Entry', zorder=5)
    
    # 做空平仓
    if short_exits:
        exit_points = [action['point'] for action in short_exits]
        exit_values = [action['index_value'] for action in short_exits]
        plt.scatter(exit_points, exit_values, color='darkgreen', marker='v', s=100, label='Short Exit', zorder=5)
    
    # 添加交易连线
    long_entry_exit_pairs = list(zip(long_entries, long_exits))
    short_entry_exit_pairs = list(zip(short_entries, short_exits))
    
    for entry, exit in long_entry_exit_pairs:
        plt.plot([entry['point'], exit['point']], 
                 [entry['index_value'], exit['index_value']], 
                 color='green', linewidth=1, alpha=0.7)
    
    for entry, exit in short_entry_exit_pairs:
        plt.plot([entry['point'], exit['point']], 
                 [entry['index_value'], exit['index_value']], 
                 color='red', linewidth=1, alpha=0.7)
    
    plt.title('Trading Signals Based on Model Predictions')
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 根据需要翻转y轴
    if invert_y:
        plt.gca().invert_yaxis()
    
    # 优化x轴显示
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"交易信号图已保存为: {filename}")

def calculate_trading_statistics(actions, index_values):
    """
    计算交易统计信息
    """
    long_entries, long_exits, short_entries, short_exits = convert_actions_to_trading_points(actions)
    
    total_profit = 0.0
    winning_trades = 0
    total_trades = len(long_entries) + len(short_entries)
    
    trade_details = []
    
    # 计算做多交易
    for entry, exit in zip(long_entries, long_exits):
        # 做多：价格上涨时盈利
        profit = exit['index_value'] - entry['index_value']
        total_profit += profit
        if profit > 0:
            winning_trades += 1
        trade_details.append({
            'direction': 'LONG',
            'entry_point': entry['point'],
            'entry_value': entry['index_value'],
            'exit_point': exit['point'],
            'exit_value': exit['index_value'],
            'profit': profit
        })
    
    # 计算做空交易
    for entry, exit in zip(short_entries, short_exits):
        # 做空：价格下跌时盈利
        profit = entry['index_value'] - exit['index_value']
        total_profit += profit
        if profit > 0:
            winning_trades += 1
        trade_details.append({
            'direction': 'SHORT',
            'entry_point': entry['point'],
            'entry_value': entry['index_value'],
            'exit_point': exit['point'],
            'exit_value': exit['index_value'],
            'profit': profit
        })
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'total_profit': total_profit,
        'average_profit': total_profit / total_trades if total_trades > 0 else 0,
        'trade_details': trade_details
    }
def predict_and_generate_signals(csv_file):
    """
    主函数：加载数据、模型预测、生成交易信号
    """
    print(f"正在处理文件: {csv_file}")
    
    # 1. 加载数据
    df = pd.read_csv(csv_file)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 2. 数据预处理
    features, index_values = prepare_data_for_prediction(df)
    print(f"特征维度: {features.shape}")
    
    # 3. 创建序列
    X = create_prediction_sequences(features, SEQ_LEN)
    print(f"预测序列数量: {len(X)}")
    
    if len(X) == 0:
        print("没有足够的数据进行预测")
        return
    
    # 4. 加载模型
    input_dim = X.shape[2]
    model = load_model(input_dim, num_classes=5)  # 五分类
    
    # 5. 模型预测
    print("正在进行模型预测...")
    signals = generate_trading_signals(model, X, index_values)
    print(f"生成 {len(signals)} 个预测信号")
    
    # 6. 分析预测信号统计
    action_counts = {}
    for i in range(5):
        action_counts[i] = sum(1 for s in signals if s['prediction'] == i)
    
    print("各类别预测统计:")
    action_names = ['无操作', '做多开仓', '做多平仓', '做空开仓', '做空平仓']
    for i in range(5):
        count = action_counts[i]
        percentage = count / len(signals) * 100
        print(f"  {action_names[i]}: {count} ({percentage:.1f}%)")
    
    # 7. 识别交易点
    actions = identify_trading_points(signals)
    print(f"识别到 {len(actions)} 个交易动作")
    
    # 8. 应用风险管理
    filtered_actions = apply_risk_management(actions)
    print(f"风险管理后剩余 {len(filtered_actions)} 个动作")
    
    # 9. 保存和可视化结果
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    
    # 保存交易动作
    actions_df = save_trading_signals(filtered_actions, f"{base_filename}_actions.csv")
    
    # 绘制交易信号图
    if index_values is not None and len(filtered_actions) > 0:
        plot_trading_signals(index_values, filtered_actions, f"{base_filename}_signals.png", invert_y=True)
    else:
        print("没有足够的交易信号生成图表")
    
    # 10. 计算交易统计
    if len(filtered_actions) > 0:
        stats = calculate_trading_statistics(filtered_actions, index_values)
        
        print("\n=== 交易统计 ===")
        print(f"总交易数: {stats['total_trades']}")
        print(f"盈利交易数: {stats['winning_trades']}")
        print(f"胜率: {stats['win_rate']*100:.1f}%")
        print(f"总盈利点数: {stats['total_profit']:.2f} 点")
        print(f"平均每笔交易盈利: {stats['average_profit']:.2f} 点")
        
        print("\n=== 交易详情 ===")
        for i, trade in enumerate(stats['trade_details']):
            print(f"交易 {i+1} ({trade['direction']}):")
            print(f"  开仓: 时间点 {trade['entry_point']}, 指数 {trade['entry_value']:.2f}")
            print(f"  平仓: 时间点 {trade['exit_point']}, 指数 {trade['exit_value']:.2f}")
            print(f"  盈利: {trade['profit']:.2f} 点")
            print()
    else:
        print("\n未识别到有效的交易信号")
    
    return actions_df

if __name__ == "__main__":
    # 使用示例
    # 请替换为实际的CSV文件路径
    csv_file_path = "../predict/250813.csv"  # 替换为实际文件路径
    predict_and_generate_signals(csv_file_path)