import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd
from model import TransformerClassifier, MultiScaleTransformerClassifier
from feature_engineering import add_features
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 30
MODEL_PATH = "../model/best_longtrend_model.pth"
SCALER_PATH = "../model/scaler.pkl"

class TradingEnvironment:
    """
    交易环境类，用于强化学习训练
    """
    def __init__(self, data_file, initial_balance=10000, transaction_cost=0.001):
        # 加载数据
        self.df = pd.read_csv(data_file)
        self.df = add_features(self.df)
        
        # 准备特征
        exclude_cols = ['label', 'index_value']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.features = self.df[feature_cols].values
        self.prices = self.df['index_value'].values if 'index_value' in self.df.columns else np.ones(len(self.df))
        self.labels = self.df['label'].values if 'label' in self.df.columns else np.zeros(len(self.df))
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # 环境参数
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: 无仓位, 1: 多头, -1: 空头
        self.position_entry_price = 0
        self.total_reward = 0
        self.max_position_duration = 20  # 最大持仓时间
        self.position_duration = 0  # 当前持仓时间
        
        # 状态维度
        self.observation_space = self.features.shape[1] * SEQ_LEN
        self.action_space = 3  # 0: 持有, 1: 买入, 2: 卖出
        
    def reset(self):
        """
        重置环境
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_entry_price = 0
        self.total_reward = 0
        self.position_duration = 0
        return self._get_state()
    
    def _get_state(self):
        """
        获取当前状态
        """
        if self.current_step < SEQ_LEN:
            return np.zeros(self.observation_space)
        
        # 获取序列特征
        state = self.features[self.current_step-SEQ_LEN+1:self.current_step+1]
        return state.flatten()
    
    def step(self, action):
        """
        执行动作并返回新状态、奖励和是否结束
        """
        # 执行交易动作
        reward = 0
        current_price = self.prices[self.current_step]
        
        # 基础奖励计算
        price_change = current_price - self.prices[self.current_step-1] if self.current_step > 0 else 0
        
        # 根据动作执行交易
        if action == 1 and self.position <= 0:  # 买入/做多
            if self.position < 0:  # 平空仓
                profit = (self.position_entry_price - current_price) 
                reward = profit - (self.position_entry_price + current_price) * self.transaction_cost
                self.balance += profit - (self.position_entry_price + current_price) * self.transaction_cost
            self.position = 1
            self.position_entry_price = current_price
            self.position_duration = 0
            reward -= current_price * self.transaction_cost  # 扣除交易费用
            
        elif action == 2 and self.position >= 0:  # 卖出/做空
            if self.position > 0:  # 平多仓
                profit = (current_price - self.position_entry_price)
                reward = profit - (self.position_entry_price + current_price) * self.transaction_cost
                self.balance += profit - (self.position_entry_price + current_price) * self.transaction_cost
            self.position = -1
            self.position_entry_price = current_price
            self.position_duration = 0
            reward -= current_price * self.transaction_cost  # 扣除交易费用
            
        # 如果持仓，计算持仓盈亏
        elif self.position != 0:
            self.position_duration += 1
            if self.position > 0:  # 多头持仓
                reward = price_change
            else:  # 空头持仓
                reward = -price_change
        
        # 添加持仓时间惩罚，避免频繁交易
        if self.position != 0:
            reward -= 0.01  # 每步持仓成本
            
        # 添加超时平仓机制
        if self.position_duration >= self.max_position_duration:
            if self.position > 0:  # 强制平多
                profit = (current_price - self.position_entry_price)
                reward += profit - (self.position_entry_price + current_price) * self.transaction_cost
                self.balance += profit - (self.position_entry_price + current_price) * self.transaction_cost
            elif self.position < 0:  # 强制平空
                profit = (self.position_entry_price - current_price)
                reward += profit - (self.position_entry_price + current_price) * self.transaction_cost
                self.balance += profit - (self.position_entry_price + current_price) * self.transaction_cost
            self.position = 0
            self.position_entry_price = 0
            self.position_duration = 0
        
        # 添加不必要交易的惩罚
        if action != 0 and ((self.position == 1 and action == 1) or (self.position == -1 and action == 2)):
            reward -= 0.1  # 重复操作惩罚
        
        # 更新总奖励
        self.total_reward += reward
        
        # 移动到下一步
        self.current_step += 1
        
        # 检查是否结束
        done = self.current_step >= len(self.features) - 1
        
        # 获取新状态
        state = self._get_state()
        
        # 添加额外信息
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_reward': self.total_reward,
            'current_price': current_price
        }
        
        return state, reward, done, info

class DQN(nn.Module):
    """
    深度Q网络
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """
    DQN智能体
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma  # 增加gamma值以更重视未来奖励
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 主网络和目标网络
        self.q_network = DQN(state_dim, 512, action_dim).to(DEVICE)
        self.target_network = DQN(state_dim, 512, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.memory = deque(maxlen=50000)  # 增加经验回放大小
        self.batch_size = 128  # 增加批量大小
        
        # 更新目标网络
        self.update_target_network()
        
    def update_target_network(self):
        """
        更新目标网络
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        存储经验
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        根据当前状态选择动作
        """
        # epsilon-greedy策略
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        # 使用Q网络选择最佳动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """
        经验回放训练
        """
        if len(self.memory) < self.batch_size:
            return
        
        # 从经验回放中采样
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(DEVICE)
        actions = torch.LongTensor([e[1] for e in batch]).to(DEVICE)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(DEVICE)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(DEVICE)
        dones = torch.BoolTensor([e[4] for e in batch]).to(DEVICE)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 下一状态的Q值
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 降低epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class HybridTradingAgent:
    """
    混合交易智能体：结合监督学习模型和强化学习策略
    """
    def __init__(self, model_path, scaler_path=None, use_multiscale=False):
        self.use_multiscale = use_multiscale
        
        # 加载scaler或创建新的
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("标准化参数加载成功")
        else:
            self.scaler = StandardScaler()
            print("未找到标准化参数文件，将使用默认标准化")
        
        # 加载监督学习模型
        if os.path.exists(model_path):
            # 先创建一个基础模型来获取输入维度
            dummy_df = pd.DataFrame(columns=['a', 'b', 'c', 'd'])  # 至少包含基础列
            dummy_features = self.prepare_features(dummy_df)
            input_dim = dummy_features.shape[1] if dummy_features.shape[1] > 1 else 88  # 默认使用88
            
            if use_multiscale:
                self.supervised_model = MultiScaleTransformerClassifier(
                    input_dim=input_dim,
                    model_dim=128,
                    num_heads=8,
                    num_layers=4,
                    num_classes=5,
                    seq_lengths=[5, 10, 15]
                ).to(DEVICE)
            else:
                self.supervised_model = TransformerClassifier(
                    input_dim=input_dim,
                    model_dim=128,
                    num_heads=8,
                    num_layers=4,
                    num_classes=5
                ).to(DEVICE)
            
            # 尝试加载模型权重
            try:
                self.supervised_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                print("监督学习模型加载成功")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print("模型结构与权重不匹配，尝试自动修复...")
                    # 尝试不严格加载权重
                    state_dict = torch.load(model_path, map_location=DEVICE)
                    self.supervised_model.load_state_dict(state_dict, strict=False)
                    print("监督学习模型加载完成（部分权重）")
                else:
                    raise e
            self.supervised_model.eval()
        else:
            print("未找到监督学习模型文件")
            
        # 强化学习智能体
        self.rl_agent = None
        
    def prepare_features(self, df):
        """
        准备特征数据
        """
        # 处理空数据框的情况
        if len(df) == 0:
            # 返回一个合适的默认特征维度
            return np.zeros((0, 88))  # 默认88个特征
            
        df = add_features(df)
        exclude_cols = ['label', 'index_value']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features = df[feature_cols].values
        
        # 如果scaler未训练，则先拟合
        if not hasattr(self.scaler, 'scale_'):
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        return features
    
    def get_supervised_action(self, features, current_step):
        """
        获取监督学习模型的预测动作
        """
        if current_step < SEQ_LEN:
            return 0  # 无动作
            
        # 构造序列
        sequence = features[current_step-SEQ_LEN+1:current_step+1]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
        
        # 模型预测
        with torch.no_grad():
            outputs = self.supervised_model(sequence_tensor)
            action = torch.argmax(outputs, dim=1).item()
            
        # 将5类标签映射到3类动作（0: 持有, 1: 买入, 2: 卖出）
        # 0: 无操作, 1: 做多开仓, 2: 做多平仓, 3: 做空开仓, 4: 做空平仓
        if action in [1, 2]:  # 做多相关
            return 1 if action == 1 else 2  # 买入/卖出
        elif action in [3, 4]:  # 做空相关
            return 2 if action == 3 else 1  # 卖出/买入（做空）
        else:
            return 0  # 持有
    
    def train_rl_agent(self, data_file, episodes=100, eval_interval=10):
        """
        训练强化学习智能体
        """
        # 创建交易环境
        env = TradingEnvironment(data_file)
        
        # 初始化RL智能体
        self.rl_agent = DQNAgent(
            state_dim=env.observation_space,
            action_dim=env.action_space,
            gamma=0.99,  # 增加对未来奖励的重视
            lr=1e-3,
            epsilon_decay=0.99  # 调整epsilon衰减速度
        )
        
        # 记录最佳性能
        best_reward = -float('inf')
        
        # 训练循环
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action = self.rl_agent.act(state)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储经验
                self.rl_agent.remember(state, action, reward, next_state, done)
                
                # 更新状态
                state = next_state
                total_reward = info['total_reward']
                
                # 经验回放
                self.rl_agent.replay()
            
            # 更新目标网络
            if episode % 5 == 0:
                self.rl_agent.update_target_network()
                
            # 评估模型性能
            if episode % eval_interval == 0:
                eval_reward = self._evaluate_agent(data_file)
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.rl_agent.epsilon:.4f}, Eval Reward: {eval_reward:.2f}")
                
                # 保存最佳模型
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    # 保存模型（可选）
            else:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.rl_agent.epsilon:.4f}")
    
    def _evaluate_agent(self, data_file, eval_episodes=1):
        """
        评估智能体性能
        """
        if self.rl_agent is None:
            return 0
            
        env = TradingEnvironment(data_file)
        total_rewards = []
        
        for _ in range(eval_episodes):
            state = env.reset()
            done = False
            while not done:
                # 使用评估模式（不探索）
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.rl_agent.q_network(state_tensor)
                action = np.argmax(q_values.cpu().data.numpy())
                
                state, reward, done, info = env.step(action)
                
            total_rewards.append(info['total_reward'])
            
        return np.mean(total_rewards)
    
    def get_hybrid_action(self, features, current_step, position):
        """
        获取混合动作（监督学习 + 强化学习）
        """
        # 获取监督学习模型建议
        supervised_action = self.get_supervised_action(features, current_step)
        
        # 如果没有训练RL智能体，直接使用监督学习动作
        if self.rl_agent is None:
            return supervised_action
            
        # 获取当前状态
        if current_step < SEQ_LEN:
            state = np.zeros(self.rl_agent.state_dim)
        else:
            state = features[current_step-SEQ_LEN+1:current_step+1].flatten()
        
        # 获取RL智能体动作
        rl_action = self.rl_agent.act(state)
        
        # 融合策略：基于置信度和状态的智能融合
        # 当RL智能体置信度高时，更多依赖RL；否则依赖监督学习
        if self.rl_agent.epsilon < 0.1:  # RL训练充分时
            # 使用RL动作
            return rl_action
        else:
            # 更多依赖监督学习
            return supervised_action

def train_reinforcement_learning_strategy(data_file, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """
    训练强化学习交易策略
    """
    print("开始训练强化学习交易策略...")
    
    # 创建混合智能体
    agent = HybridTradingAgent(model_path, scaler_path)
    
    # 训练RL智能体
    print("训练强化学习智能体...")
    agent.train_rl_agent(data_file, episodes=100)
    
    print("强化学习策略训练完成!")
    return agent

def backtest_hybrid_strategy(data_file, agent):
    """
    回测混合策略
    """
    print("开始回测混合策略...")
    
    # 创建交易环境
    env = TradingEnvironment(data_file)
    state = env.reset()
    done = False
    
    # 记录交易历史
    trades = []
    balances = [env.balance]
    
    while not done:
        # 获取混合动作
        action = agent.get_hybrid_action(agent.prepare_features(env.df), env.current_step, env.position)
        
        # 执行动作
        state, reward, done, info = env.step(action)
        
        # 记录信息
        balances.append(info['balance'])
        if action != 0:  # 如果执行了交易动作
            trades.append({
                'step': env.current_step,
                'action': action,
                'price': info['current_price'],
                'balance': info['balance'],
                'position': info['position']
            })
    
    print(f"回测完成!")
    print(f"初始资金: {env.initial_balance}")
    print(f"最终资金: {env.balance}")
    print(f"总收益率: {(env.balance - env.initial_balance) / env.initial_balance * 100:.2f}%")
    print(f"交易次数: {len(trades)}")
    
    return {
        'initial_balance': env.initial_balance,
        'final_balance': env.balance,
        'total_return': (env.balance - env.initial_balance) / env.initial_balance * 100,
        'num_trades': len(trades),
        'trades': trades,
        'balances': balances
    }

# 添加一个函数来保存scaler
def save_scaler(scaler, scaler_path=SCALER_PATH):
    """
    保存scaler到文件
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"标准化参数已保存到: {scaler_path}")

if __name__ == "__main__":
    # 示例用法
    # data_file = "../data/250814.csv"  # 替换为实际数据文件路径
    # trained_agent = train_reinforcement_learning_strategy(data_file)
    # results = backtest_hybrid_strategy(data_file, trained_agent)
    # print("回测结果:", results)
    # save_scaler(trained_agent.scaler)
    # 示例用法
    import glob
    
    # 获取所有CSV文件
    data_files = glob.glob("../data/*.csv")
    
    if not data_files:
        print("未找到任何CSV文件")
        exit()
    
    # 使用第一个文件训练
    print(f"找到 {len(data_files)} 个CSV文件")
    print("使用第一个文件进行训练...")
    training_file = data_files[0]
    trained_agent = train_reinforcement_learning_strategy(training_file)
    
    # 对所有文件进行回测
    all_results = []
    for data_file in data_files:
        print(f"正在回测文件: {data_file}")
        results = backtest_hybrid_strategy(data_file, trained_agent)
        all_results.append({
            'file': data_file,
            'results': results
        })
        print("回测结果:", results)
    
    # 保存scaler
    save_scaler(trained_agent.scaler)