import torch
import torch.nn as nn

import numpy as np

def pad_gt_obs(gt_features, max_gt_num=5):
    """填充GT特征到固定数量，返回掩码"""
    batch_size = len(gt_features)
    padded = np.zeros((batch_size, max_gt_num, 3))
    masks = np.zeros((batch_size, max_gt_num))
    for i in range(batch_size):
        valid_num = len(gt_features[i])
        padded[i, :valid_num] = gt_features[i][:max_gt_num]
        masks[i, :valid_num] = 1.0
    return torch.tensor(padded, dtype=torch.float32), torch.tensor(masks, dtype=torch.bool)

class Actor(nn.Module):
    def __init__(self, gt_features_dim: int, max_gt_num: int, other_features_dim: int, action_dim: int, hidden_dim: int):
        super(Actor, self).__init__()
        # 处理其他观测
        self.other_fc = nn.Sequential(
            nn.Linear(other_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GT特征处理
        self.gt_encoder = nn.Linear(gt_features_dim, hidden_dim)  # 将每个GT特征编码为hidden_dim维
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )

        # 合并处理
        self.merge_fc = nn.Linear(hidden_dim + hidden_dim, hidden_dim)  # 其他特征 + GT注意力输出
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, 
                gt_features: torch.Tensor, 
                other_features: torch.Tensor, 
                mask: torch.Tensor,  
                h_in: torch.Tensor):
        """
        Args:
            other_obs: [batch_size, other_obs_dim]
            gt_obs: [batch_size, max_gt_num, 3] （填充后的GT特征）
            h_in: GRU隐藏状态 [batch_size, hidden_dim]
        """
        # 处理其他观测
        other_features = self.other_fc(other_features)
        
        # 处理GT特征
        batch_size = gt_features.shape[0]
        gt_encoded = self.gt_encoder(gt_features)

        # 注意力层
        attn_output, _ = self.attention(
            query=gt_encoded,
            key=gt_encoded,
            value=gt_encoded,
            key_padding_mask=mask
        )

        # 取平均或最后一个维度作为固定长度向量
        pooled_gt = attn_output.mean(dim=1)  

        # 合并特征
        merged = torch.cat([other_features, pooled_gt], dim=1)
        merged = self.merge_fc(merged)
        
        # GRU处理
        h_out = self.gru(merged, h_in)
        action = self.tanh(self.fc_out(h_out))
        
        return action, h_out

class CentralCritic(nn.Module):
    def __init__(self, 
                 n_agents: int,          # 智能体数量（如无人机数量）
                 gt_features_dim: int,   # 每个GT特征维度（如3）
                 max_gt_num: int,        # 最大GT数量
                 other_features_dim: int,# 单个智能体其他观测维度
                 action_dim: int,        # 单个智能体动作维度
                 hidden_dim: int = 64):
        super(CentralCritic, self).__init__()
        
        # 单个智能体的GT特征处理
        self.gt_encoder = nn.Linear(gt_features_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )
        
        # 单个智能体的其他特征处理
        self.other_fc = nn.Sequential(
            nn.Linear(other_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 全局聚合层
        self.global_merge = nn.Sequential(
            nn.Linear(hidden_dim * 2 * n_agents, hidden_dim),  # 每个智能体的合并特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作处理
        self.action_fc = nn.Linear(action_dim * n_agents, hidden_dim)
        
        # Q值输出
        self.q_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, 
               all_gt_features: torch.Tensor,  # [n_agents, batch_size, max_gt_num, 3]
               all_other_features: torch.Tensor,  # [n_agents, batch_size, other_features_dim]
               all_masks: torch.Tensor,          # [n_agents, batch_size, max_gt_num]
               all_actions: torch.Tensor,        # [n_agents, batch_size, action_dim]
               ):
        """
        Args:
            all_gt_features: 所有智能体的GT特征（需转置为[n_agents, batch_size, ...]）
            all_other_features: 所有智能体的其他观测
            all_masks: 所有智能体的掩码
            all_actions: 所有智能体的动作
        Returns:
            q_value: [batch_size, 1]
        """
        batch_size = all_other_features.shape[1]
        agent_outputs = []
        
        # 处理每个智能体的GT特征
        for agent_id in range(n_agents):
            gt_features = all_gt_features[agent_id]  # [batch_size, max_gt_num, 3]
            mask = all_masks[agent_id]
            
            # 编码GT特征（与Actor相同逻辑）
            gt_encoded = self.gt_encoder(gt_features)
            attn_output, _ = self.attention(
                query=gt_encoded,
                key=gt_encoded,
                value=gt_encoded,
                key_padding_mask=mask
            )
            pooled_gt = attn_output.mean(dim=1)  # [batch, hidden_dim]
            
            # 处理其他观测
            other_encoded = self.other_fc(all_other_features[agent_id])
            
            # 合并单个智能体的特征
            agent_output = torch.cat([other_encoded, pooled_gt], dim=1)
            agent_outputs.append(agent_output)
        
        # 聚合所有智能体的特征
        global_state = torch.cat(agent_outputs, dim=1)  # [batch, hidden_dim * 2 * n_agents]
        global_state = self.global_merge(global_state)
        
        # 处理所有动作
        all_actions = all_actions.view(batch_size, -1)  # [batch, n_agents * action_dim]
        action_encoded = self.action_fc(all_actions)
        
        # 合并全局状态和动作
        merged = torch.cat([global_state, action_encoded], dim=1)
        q_value = self.q_out(merged)
        
        return q_value

if __name__ == "__main__":
    # torch.manual_seed(10)
    # torch.cuda.manual_seed(10)
    
    # # 环境返回原始GT特征列表（变长）
    # raw_gt_features = [
    #     [[1.0, 2.0, 0.5], [3.0, 4.0, 0.3]],  # 第一个样本有2个GT
    #     [[5.0, 6.0, 0.7], [7.0, 8.0, 0.2], [9.0, 10.0, 0.1]]  # 第二个样本有3个GT
    # ]
    
    # # 填充并生成掩码
    # padded_gt, mask = pad_gt_obs(raw_gt_features, max_gt_num=5)
    
    # print("填充后的GT特征:", padded_gt)
    # print("掩码:", mask)

    # # 其他观测（假设为固定维度向量）
    # other_obs = torch.randn(2, 100)  # batch_size=2，其他观测维度=100

    # # 初始化Actor并前向传播
    # actor = Actor(
    #     gt_features_dim=3,
    #     max_gt_num=5,
    #     other_features_dim=100,
    #     action_dim=10,
    #     hidden_dim=64
    # )

    # action, _ = actor(
    #     gt_features=padded_gt,
    #     other_features=other_obs,
    #     mask=mask,
    #     h_in=torch.zeros(2, 64)
    # )

    # print(action)

    n_agents = 2  # 无人机数量
    hidden_dim = 64
    action_dim = 10

    critic = CentralCritic(
        n_agents=n_agents,
        gt_features_dim=3,
        max_gt_num=5,
        other_features_dim=100,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )

    # 假设有2个智能体，每个的观测和动作：
    all_gt_features = torch.stack([padded_gt1, padded_gt2], dim=0)  # [2, batch_size, 5, 3]
    all_other_features = torch.stack([obs1, obs2], dim=0)          # [2, batch_size, 100]
    all_masks = torch.stack([mask1, mask2], dim=0)                 # [2, batch_size, 5]
    all_actions = torch.stack([action1, action2], dim=0)           # [2, batch_size, 10]

    q_value = critic(
        all_gt_features,
        all_other_features,
        all_masks,
        all_actions
    )
    print("全局Q值:", q_value)  # [batch_size, 1]