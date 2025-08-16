import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding

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
