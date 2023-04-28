import torch
import torch.nn as nn

# 定义技术指标状态神经网络
class TechnicalNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TechnicalNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 定义历史行为状态神经网络
class BehaviorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehaviorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# 定义混合神经网络
class CombinedActorNet(nn.Module):
    def __init__(self, input_dim_technical, input_dim_behavior, output_dim_technical, output_dim_behavior, action_dim):
        super(CombinedActorNet, self).__init__()
        self.input_dim_technical = input_dim_technical
        self.input_dim_behavior = input_dim_behavior
        # 初始化技术指标状态预测模型和历史行为状态预测模型
        self.indicator_model = TechnicalNet(input_dim_technical, output_dim_technical)
        self.behavior_model = BehaviorNet(input_dim_behavior, output_dim_behavior)
        self.fc1 = nn.Linear(output_dim_technical + output_dim_behavior, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        x1 = state[:, :self.input_dim_technical]
        x2 = state[:, self.input_dim_technical:]
        indicator_output = self.indicator_model(x1)
        behavior_output = self.behavior_model(x2)
        x = torch.cat((indicator_output, behavior_output), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x



