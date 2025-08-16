# config.py
# 全局参数

DATA_CSV = "data.csv"          # 原始数据文件（按时间顺序，从上到下）
SEQ_LEN = 60                     # 历史窗口长度（秒）
PRED_HORIZON = 60                # 预测未来多少秒的方向
LABEL_THRESHOLD = 0.005           # 若为 None 则程序自动用 median(|future_change|) 作为阈值
MODEL_SAVE_PATH = "saved_model.pth"
SCALER_SAVE_PATH = "scaler.save"

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4
VAL_RATIO = 0.15
RANDOM_SEED = 42

CONF_THRESHOLD = 0.90           # 置信度阈值（softmax 最大概率）
COOLDOWN_SEC = 600              # 冷却期（秒），发信号后跳过这么多秒的信号

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"