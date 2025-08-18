import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import TransformerClassifier
from feature_engineering import add_features

SEQ_LEN = 60
MODEL_PATH = "E:/SnipingTactics/best_longtrend_model.pth"

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载训练时用的scaler（这里演示先重新fit，需要你保存scaler用joblib等工具）
# 这里为了示例先重新用历史数据拟合scaler
df_train = pd.read_csv("../predict/250806.csv")
df_train = add_features(df_train)
exclude_cols = ['label', 'index_value']
feature_cols = [c for c in df_train.columns if c not in exclude_cols]
scaler = StandardScaler()
scaler.fit(df_train[feature_cols].values)

# 2. 加载模型 - 修正：使用3个类别而不是2个
input_dim = len(feature_cols)
model = TransformerClassifier(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=2, num_classes=3)  # 改为3
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 3. 你新来的最新数据，格式：pd.DataFrame，列名要和训练数据特征名对应，至少有a,b,c,d,index_value列
# 这里假设 new_data_df 是最近至少 SEQ_LEN 秒的原始数据（a,b,c,d,index_value）
# 注意：这里你得保证new_data_df的行数 >= SEQ_LEN，且顺序是时间先后

def preprocess_new_data(new_data_df):
    new_data_df = add_features(new_data_df)
    new_data_df.fillna(0, inplace=True)
    feat = new_data_df[feature_cols].values
    feat_scaled = scaler.transform(feat)
    return feat_scaled

# ... existing code ...

def real_time_predict(model, scaler, recent_data, seq_lengths=[10, 30, 60]):
    """
    实时预测函数，基于最近的数据进行预测
    
    Args:
        model: 训练好的模型
        scaler: 特征标准化器
        recent_data: 最近的数据点 (pandas DataFrame)
        seq_lengths: 序列长度列表
    
    Returns:
        prediction: 预测结果
        probability: 预测概率
    """
    # 确保有足够的数据
    max_seq_len = max(seq_lengths)
    if len(recent_data) < max_seq_len:
        print(f"数据不足，需要至少{max_seq_len}个数据点")
        return None, None
    
    # 只使用最近max_seq_len个数据点
    recent_data = recent_data.tail(max_seq_len).copy()
    
    # 特征工程
    recent_data = add_features(recent_data)
    
    # 准备特征
    exclude_cols = ['label', 'index_value']
    feature_cols = [col for col in recent_data.columns if col not in exclude_cols]
    features = recent_data[feature_cols].values
    
    # 标准化
    features_scaled = scaler.transform(features)
    
    # 创建多尺度序列
    x_dict = {}
    for length in seq_lengths:
        sequence = features_scaled[-length:]
        x_dict[str(length)] = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # 添加batch维度
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(x_dict)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
        
    return prediction.item(), probabilities.cpu().numpy()[0]

# ... existing code ...
# 4. 构造序列输入模型
def predict_trend(new_data_df):
    feat_scaled = preprocess_new_data(new_data_df)
    if len(feat_scaled) < SEQ_LEN:
        raise ValueError(f"输入数据行数少于SEQ_LEN={SEQ_LEN}，无法预测")
    seq_input = feat_scaled[-SEQ_LEN:]  # 取最后SEQ_LEN条
    seq_tensor = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, feat_dim)

    with torch.no_grad():
        output = model(seq_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(prob)
        confidence = prob[pred_class]
        
        # 添加类别映射，便于理解预测结果
        class_mapping = {0: "下跌", 1: "未知", 2: "上涨"}
        pred_label = class_mapping[pred_class]
        
    return pred_class, pred_label, confidence

# 5. 示例用法
if __name__ == "__main__":
    # 示例：读取最新数据并进行预测
    # 请替换为实际的数据文件路径
    try:
        # 读取最新60秒的数据
        new_data_df = pd.read_csv("../predict/250110S.csv")
        
        # 进行预测
        pred_class, pred_label, confidence = predict_trend(new_data_df)
        
        print(f"预测结果:")
        print(f"预测类别索引: {pred_class}")
        print(f"预测标签: {pred_label}")
        print(f"置信度: {confidence:.4f}")
        
        # 输出各类别概率
        feat_scaled = preprocess_new_data(new_data_df)
        seq_input = feat_scaled[-SEQ_LEN:]
        seq_tensor = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(seq_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            
        print(f"\n各类别概率:")
        class_mapping = {0: "下跌", 1: "未知", 2: "上涨"}
        for i, p in enumerate(prob):
            print(f"{class_mapping[i]}: {p:.4f}")
            
    except FileNotFoundError:
        print("请提供最新的数据文件 latest_60_seconds.csv")
    except Exception as e:
        print(f"预测过程中出现错误: {e}")