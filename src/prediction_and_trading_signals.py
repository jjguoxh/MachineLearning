import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from model import TransformerClassifier, MultiScaleTransformerClassifier
from feature_engineering import add_features

# 配置参数
SEQ_LEN = 30
MODEL_PATH = "../model/best_longtrend_model.pth"
SCALER_PATH = "../model/scaler.pkl"
TRAINED_FILES_LOG = "../model/trained_files.log"

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_scaler(scaler_path=SCALER_PATH):
    """
    加载特征标准化器
    """
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("特征标准化器加载成功")
        return scaler
    else:
        print("未找到特征标准化器，需要重新训练模型时生成")
        return None

def load_model(model_path=MODEL_PATH, input_dim=None, num_classes=None, use_multiscale=False, seq_lengths=[10, 30, 60]):
    """
    加载训练好的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    if use_multiscale:
        # 加载多尺度模型
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
        # 加载单尺度模型
        model = TransformerClassifier(
            input_dim=input_dim,
            model_dim=128,
            num_heads=8,
            num_layers=4,
            num_classes=num_classes,
            dropout=0.1
        ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("模型加载成功")
    return model

def preprocess_data(df, scaler=None):
    """
    数据预处理
    """
    # 特征工程
    df = add_features(df)
    
    # 准备特征列
    exclude_cols = ['label', 'index_value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features = df[feature_cols].values
    
    # 标准化
    if scaler is not None:
        features = scaler.transform(features)
    else:
        print("警告：未提供特征标准化器，使用原始特征")
    
    return features, feature_cols

def create_sequence(features, seq_len=SEQ_LEN):
    """
    创建单个序列用于预测
    """
    if len(features) < seq_len:
        raise ValueError(f"数据长度不足，需要至少{seq_len}个数据点")
    
    # 取最后seq_len个数据点
    sequence = features[-seq_len:]
    return sequence

def create_multiscale_sequences(features, seq_lengths=[10, 30, 60]):
    """
    创建多尺度序列用于预测
    """
    max_seq_len = max(seq_lengths)
    if len(features) < max_seq_len:
        raise ValueError(f"数据长度不足，需要至少{max_seq_len}个数据点")
    
    # 取最后max_seq_len个数据点
    features = features[-max_seq_len:]
    
    # 创建多尺度序列
    x_dict = {}
    for length in seq_lengths:
        sequence = features[-length:]
        x_dict[str(length)] = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # 添加batch维度
    
    return x_dict

def predict_single_sequence(model, sequence, use_multiscale=False):
    """
    对单个序列进行预测
    """
    model.eval()
    with torch.no_grad():
        if use_multiscale:
            # 多尺度模型预测
            outputs = model(sequence)
        else:
            # 单尺度模型预测
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            outputs = model(sequence_tensor)
        
        # 计算概率和预测结果
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
        
    return prediction.item(), probabilities.cpu().numpy()[0]

def get_prediction_label(pred_class):
    """
    获取预测标签的可读文本
    """
    class_mapping = {
        0: "无操作",
        1: "做多开仓", 
        2: "做多平仓",
        3: "做空开仓",
        4: "做空平仓"
    }
    return class_mapping.get(pred_class, "未知")

def predict_from_csv(file_path, use_multiscale=False, seq_lengths=[10, 30, 60]):
    """
    从CSV文件进行预测
    """
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 加载标准化器
    scaler = load_scaler()
    
    # 数据预处理
    features, feature_cols = preprocess_data(df, scaler)
    
    # 加载模型
    input_dim = len(feature_cols)
    num_classes = 5  # 根据训练时的类别数调整
    
    model = load_model(
        input_dim=input_dim, 
        num_classes=num_classes, 
        use_multiscale=use_multiscale, 
        seq_lengths=seq_lengths if use_multiscale else None
    )
    
    # 创建序列
    if use_multiscale:
        sequence = create_multiscale_sequences(features, seq_lengths)
    else:
        sequence = create_sequence(features, SEQ_LEN)
    
    # 预测
    pred_class, probabilities = predict_single_sequence(model, sequence, use_multiscale)
    pred_label = get_prediction_label(pred_class)
    
    # 输出结果
    print(f"预测类别索引: {pred_class}")
    print(f"预测标签: {pred_label}")
    print(f"置信度: {max(probabilities):.4f}")
    
    print("\n各类别概率:")
    for i, prob in enumerate(probabilities):
        print(f"{get_prediction_label(i)}: {prob:.4f}")
    
    return pred_class, pred_label, max(probabilities)

def real_time_predict(latest_data_df, use_multiscale=False, seq_lengths=[10, 30, 60]):
    """
    实时预测函数
    """
    # 加载标准化器
    scaler = load_scaler()
    
    # 数据预处理
    features, feature_cols = preprocess_data(latest_data_df, scaler)
    
    # 加载模型
    input_dim = len(feature_cols)
    num_classes = 5  # 根据训练时的类别数调整
    
    model = load_model(
        input_dim=input_dim, 
        num_classes=num_classes, 
        use_multiscale=use_multiscale, 
        seq_lengths=seq_lengths if use_multiscale else None
    )
    
    # 创建序列
    if use_multiscale:
        sequence = create_multiscale_sequences(features, seq_lengths)
    else:
        sequence = create_sequence(features, SEQ_LEN)
    
    # 预测
    pred_class, probabilities = predict_single_sequence(model, sequence, use_multiscale)
    pred_label = get_prediction_label(pred_class)
    
    return pred_class, pred_label, max(probabilities), probabilities

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    # if len(sys.argv) < 2:
    #     print("用法: python predict.py <csv文件路径> [--multiscale]")
    #     print("示例: python predict.py ../data/today.csv")
    #     print("示例: python predict.py ../data/today.csv --multiscale")
    #     sys.exit(1)
    
    file_path = "../predict/250813.csv"
    use_multiscale = "--multiscale" in sys.argv
    
    try:
        print(f"使用{'多尺度' if use_multiscale else '单尺度'}模型进行预测")
        predict_from_csv(file_path, use_multiscale=use_multiscale)
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()