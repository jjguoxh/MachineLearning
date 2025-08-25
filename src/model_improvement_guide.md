# 机器学习预测系统改进建议

## 🎯 当前问题诊断

### 1. 可能的问题原因
- **标签质量差**: 标签生成逻辑可能不合理，导致噪声标签
- **特征不足**: 特征工程不够充分，缺少关键的技术指标
- **模型过拟合**: 模型在训练集上表现好，但泛化能力差
- **数据不平衡**: 各类别样本数量不均衡
- **超参数不当**: 学习率、批次大小等参数设置不合理

## 🚀 具体改进方案

### 1. 数据质量改进

#### 1.1 改进标签生成策略
```python
# 使用更稳健的标签生成方法
def improved_label_generation(df, window_size=60, profit_threshold=0.02):
    """
    改进的标签生成：
    1. 使用未来收益率作为标签
    2. 加入风险调整
    3. 考虑交易成本
    """
    labels = []
    for i in range(len(df) - window_size):
        current_price = df['index_value'].iloc[i]
        future_prices = df['index_value'].iloc[i+1:i+window_size+1]
        
        # 计算未来最大收益和最大损失
        max_gain = (future_prices.max() - current_price) / current_price
        max_loss = (current_price - future_prices.min()) / current_price
        
        # 风险调整的标签生成
        if max_gain > profit_threshold and max_gain > max_loss * 1.5:
            labels.append(1)  # 做多
        elif max_loss > profit_threshold and max_loss > max_gain * 1.5:
            labels.append(-1)  # 做空
        else:
            labels.append(0)  # 无操作
    
    return labels
```

#### 1.2 数据清洗和预处理
```python
def advanced_data_cleaning(df):
    """
    高级数据清洗
    """
    # 1. 异常值检测和处理
    for col in ['a', 'b', 'c', 'd', 'index_value']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 2. 缺失值智能填充
    df = df.interpolate(method='linear')
    
    # 3. 数据平滑（去除高频噪声）
    for col in ['a', 'b', 'c', 'd', 'index_value']:
        df[f'{col}_smooth'] = df[col].rolling(window=3, center=True).mean()
    
    return df
```

### 2. 特征工程改进

#### 2.1 增加技术指标特征
```python
def add_technical_indicators(df):
    """
    添加重要的技术指标
    """
    # RSI
    delta = df['index_value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['index_value'].ewm(span=12).mean()
    ema_26 = df['index_value'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # 布林带
    sma_20 = df['index_value'].rolling(window=20).mean()
    std_20 = df['index_value'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
    
    # ATR (平均真实波幅)
    high_low = df['index_value'].rolling(window=2).max() - df['index_value'].rolling(window=2).min()
    df['atr'] = high_low.rolling(window=14).mean()
    
    return df
```

#### 2.2 多时间框架特征
```python
def add_multi_timeframe_features(df):
    """
    添加多时间框架特征
    """
    for window in [5, 10, 20, 50]:
        # 移动平均
        df[f'ma_{window}'] = df['index_value'].rolling(window=window).mean()
        
        # 相对位置
        df[f'position_{window}'] = (df['index_value'] - df[f'ma_{window}']) / df[f'ma_{window}']
        
        # 动量
        df[f'momentum_{window}'] = df['index_value'].pct_change(window)
        
        # 波动率
        df[f'volatility_{window}'] = df['index_value'].pct_change().rolling(window=window).std()
    
    return df
```

### 3. 模型架构改进

#### 3.1 注意力机制改进
```python
class ImprovedTransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        
        # 输入层
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(model_dim)
        
        # Transformer编码器（带残差连接）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',  # 使用GELU激活函数
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # 输入投影和归一化
        x = self.layer_norm1(self.input_fc(x))
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 自适应池化
        x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)
        x = self.adaptive_pool(x).squeeze(-1)  # (batch, features)
        
        # 分类
        return self.classifier(x)
```

#### 3.2 集成学习
```python
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 加权平均
        weighted_output = sum(w * out for w, out in zip(self.weights, outputs))
        return weighted_output
```

### 4. 训练策略改进

#### 4.1 高级损失函数
```python
class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class WeightedFocalLoss(nn.Module):
    """
    结合类别权重和Focal Loss
    """
    def __init__(self, class_weights, alpha=1, gamma=2):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

#### 4.2 学习率调度
```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    余弦退火学习率调度，带预热
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
```

#### 4.3 数据增强
```python
def data_augmentation(X, y, noise_factor=0.01):
    """
    时间序列数据增强
    """
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # 原始数据
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # 添加噪声
        noise = np.random.normal(0, noise_factor, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
        
        # 时间扰动（轻微的时间偏移）
        if len(X[i]) > 2:
            shifted = np.roll(X[i], 1, axis=0)
            augmented_X.append(shifted)
            augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)
```

### 5. 评估和验证改进

#### 5.1 时间序列交叉验证
```python
def time_series_cv(X, y, n_splits=5):
    """
    时间序列交叉验证
    """
    splits = []
    total_len = len(X)
    
    for i in range(n_splits):
        train_end = int(total_len * (i + 1) / (n_splits + 1))
        val_start = train_end
        val_end = int(total_len * (i + 2) / (n_splits + 1))
        
        train_idx = list(range(train_end))
        val_idx = list(range(val_start, val_end))
        
        splits.append((train_idx, val_idx))
    
    return splits
```

#### 5.2 金融指标评估
```python
def evaluate_trading_performance(predictions, prices):
    """
    评估交易性能
    """
    returns = []
    positions = []
    
    for i, pred in enumerate(predictions):
        if pred == 1:  # 买入信号
            positions.append(1)
        elif pred == -1:  # 卖出信号
            positions.append(-1)
        else:
            positions.append(0)
    
    # 计算收益
    for i in range(1, len(positions)):
        if positions[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1] * positions[i-1]
            returns.append(ret)
    
    if returns:
        total_return = np.sum(returns)
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': len(returns)
        }
    
    return None
```

## 📊 实施建议

### 优先级顺序
1. **高优先级**: 改进标签生成和特征工程
2. **中优先级**: 优化模型架构和损失函数
3. **低优先级**: 实施集成学习和高级验证

### 具体实施步骤
1. 首先运行改进版预测脚本 `predict_improved.py` 查看当前效果
2. 根据分析结果重新生成更好的标签
3. 增加技术指标特征
4. 调整模型超参数
5. 使用更好的损失函数和优化器
6. 实施交叉验证评估

### 监控指标
- **模型指标**: 准确率、精确率、召回率、F1分数
- **交易指标**: 总收益率、夏普比率、胜率、最大回撤
- **稳定性指标**: 预测一致性、置信度分布

通过这些改进，预测效果应该会有显著提升！