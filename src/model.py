import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1, max_len=5000):
        super(TransformerClassifier, self).__init__()
        self.model_dim = model_dim
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(model_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(self.relu(x))
        out = self.fc_out(x)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)

class MultiScaleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, seq_lengths, dropout=0.1):
        super(MultiScaleTransformerClassifier, self).__init__()
        self.seq_lengths = seq_lengths
        self.model_dim = model_dim
        
        # 为每个尺度创建输入层
        self.input_fcs = nn.ModuleDict({
            str(length): nn.Linear(input_dim, model_dim) 
            for length in seq_lengths
        })
        
        # 为每个尺度创建位置编码
        self.pos_encoders = nn.ModuleDict({
            str(length): PositionalEncoding(model_dim, length) 
            for length in seq_lengths
        })
        
        # 为每个尺度创建Transformer编码器
        self.transformer_encoders = nn.ModuleDict()
        for length in seq_lengths:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoders[str(length)] = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 融合层
        self.fusion_layer = nn.Linear(model_dim * len(seq_lengths), model_dim)
        
        # 输出层
        self.fc_out = nn.Linear(model_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_dict):
        # 处理多尺度输入
        features = []
        for length in self.seq_lengths:
            x = x_dict[str(length)]
            x = self.input_fcs[str(length)](x)
            x = self.pos_encoders[str(length)](x)
            x = self.transformer_encoders[str(length)](x)
            x = x.mean(dim=1)  # 全局平均池化
            features.append(x)
        
        # 融合多尺度特征
        fused = torch.cat(features, dim=1)
        fused = self.relu(self.fusion_layer(fused))
        fused = self.dropout(fused)
        
        # 输出
        out = self.fc_out(fused)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)