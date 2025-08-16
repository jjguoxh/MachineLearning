import torch
from data_preprocessing import load_data, create_features
from model import TransformerPredictor
from config import MODEL_SAVE_PATH, COOLDOWN

model = TransformerPredictor(input_dim=4)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

df = load_data()
X = create_features(df)
X_tensor = torch.tensor(X, dtype=torch.float32)

signals = []
last_signal_time = -COOLDOWN
for i in range(len(X_tensor)):
    output = model(X_tensor[i].unsqueeze(0))
    pred_class = torch.argmax(output).item() - 1  # 转回[-1,0,1]
    confidence = torch.max(output).item()
    if pred_class != 0 and (i - last_signal_time) > COOLDOWN:
        signals.append({"index": i, "class": pred_class, "conf": confidence})
        last_signal_time = i

print("总信号数:", len(signals))
print("前5个信号:", signals[:5])
