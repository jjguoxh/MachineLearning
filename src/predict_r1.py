import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

from model import TransformerClassifier, MultiScaleTransformerClassifier
from feature_engineering import add_features
from reml import HybridTradingAgent

# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 30
MODEL_PATH = "../model/best_longtrend_model.pth"
SCALER_PATH = "../model/scaler.pkl"

def load_model(model_path, input_dim, num_classes=5, use_multiscale=False, seq_lengths=[5, 10, 15]):
    """
    加载训练好的模型
    """
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
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("警告: 模型结构与权重不匹配，尝试自动修复...")
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise e
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        raise e
        
    model.eval()
    print("模型加载成功")
    return model

def prepare_data(df, scaler=None):
    """
    准备预测数据
    """
    # 特征工程
    df = add_features(df)
    
    # 准备特征
    exclude_cols = ['label', 'index_value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features = df[feature_cols].values
    
    # 标准化
    if scaler is not None and hasattr(scaler, 'scale_'):
        features = scaler.transform(features)
    else:
        print("警告: 未提供有效的特征标准化器，使用默认标准化")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    return features, df['index_value'].values if 'index_value' in df.columns else None

def create_sequences(features, seq_len=SEQ_LEN):
    """
    创建预测序列
    """
    X = []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i:i + seq_len])
    return np.array(X)

def predict_with_supervised_model(model, X, feature_cols):
    """
    使用监督学习模型进行预测
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        probs = probs.cpu().numpy()
    
    return preds, probs

def plot_trading_signals(index_values, predictions, probabilities=None, title="Trading Signals"):
    """
    绘制交易信号图
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制价格曲线
    ax.plot(index_values, label='Index Value', color='blue', linewidth=1)
    
    # 标记各类动作
    long_entries = []   # 做多开仓 (action=1)
    short_entries = []  # 做空开仓 (action=2)
    holds = []          # 持有 (action=0)
    
    for i, pred in enumerate(predictions):
        point = i + SEQ_LEN - 1  # 序列的最后一个点
        if point < len(index_values):
            if pred == 1:  # 做多开仓
                long_entries.append((point, index_values[point]))
            elif pred == 2:  # 做空开仓
                short_entries.append((point, index_values[point]))
            else:  # 持有
                holds.append((point, index_values[point]))
    
    # 绘制交易信号
    if long_entries:
        x, y = zip(*long_entries)
        ax.scatter(x, y, color='red', marker='^', s=100, label='Long Entry', zorder=5)
    
    if short_entries:
        x, y = zip(*short_entries)
        ax.scatter(x, y, color='green', marker='v', s=100, label='Short Entry', zorder=5)
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Index Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def predict_rl(csv_file, use_hybrid=False, rl_model_path=None):
    """
    使用强化学习模型进行预测
    """
    print(f"正在处理文件: {csv_file}")
    
    # 1. 加载数据
    df = pd.read_csv(csv_file)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 2. 数据预处理
    try:
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        if scaler is None:
            print("未找到scaler文件，将使用数据自带的标准化")
    except Exception as e:
        print(f"加载scaler时出错: {e}")
        scaler = None
    
    features, index_values = prepare_data(df, scaler)
    print(f"特征维度: {features.shape}")
    
    # 3. 创建序列
    X = create_sequences(features, SEQ_LEN)
    print(f"预测序列数量: {len(X)}")
    
    if len(X) == 0:
        print("没有足够的数据进行预测")
        return
    
    # 4. 预测
    if use_hybrid and rl_model_path:
        # 使用混合模型（监督学习+强化学习）
        print("使用混合模型进行预测...")
        agent = HybridTradingAgent(MODEL_PATH, SCALER_PATH)
        # 这里简单地使用监督学习部分进行预测
        # 在实际应用中，您可以加载训练好的RL部分
        predictions, probabilities = predict_with_supervised_model(
            load_model(MODEL_PATH, features.shape[1]), 
            X, 
            df.columns
        )
    else:
        # 仅使用监督学习模型
        print("使用监督学习模型进行预测...")
        model = load_model(MODEL_PATH, features.shape[1])
        predictions, probabilities = predict_with_supervised_model(model, X, df.columns)
    
    # 5. 分析预测结果
    print("\n预测结果统计:")
    unique, counts = np.unique(predictions, return_counts=True)
    action_names = ['持有', '做多开仓', '做空开仓']
    for label, count in zip(unique, counts):
        print(f"  {action_names[label]} ({label}): {count} 个")
    
    # 6. 可视化结果
    if index_values is not None:
        fig = plot_trading_signals(index_values, predictions, probabilities, 
                                  "Trading Signals based on RL-enhanced Predictions")
        plt.savefig(f"{os.path.splitext(csv_file)[0]}_rl_signals.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"交易信号图已保存为: {os.path.splitext(csv_file)[0]}_rl_signals.png")
    
    # 7. 保存预测结果
    results_df = pd.DataFrame({
        'point': np.arange(SEQ_LEN-1, SEQ_LEN-1+len(predictions)),
        'prediction': predictions,
        'action': [action_names[p] for p in predictions]
    })
    
    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            results_df[f'prob_class_{i}'] = probabilities[:, i]
    
    if index_values is not None and len(index_values) >= SEQ_LEN:
        results_df['index_value'] = index_values[SEQ_LEN-1:SEQ_LEN-1+len(predictions)]
    
    output_file = f"{os.path.splitext(csv_file)[0]}_rl_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"预测结果已保存为: {output_file}")
    
    return results_df

def main():
    """
    主函数
    """
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python predict_rl.py <data_file> [--hybrid]")
        print("示例: python predict_rl.py ../data/250814.csv --hybrid")
        return
    
    data_file = sys.argv[1]
    use_hybrid = "--hybrid" in sys.argv
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return
    
    try:
        results = predict_rl(data_file, use_hybrid)
        print("\n预测完成!")
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()