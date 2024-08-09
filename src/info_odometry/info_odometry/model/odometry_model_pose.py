import torch
from torch import nn
from torch.nn import functional as F


class OdometryModelPose(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu'):
        """
        embedding_size: not used (for code consistency in main.py)
        """
        # use posterior_states for pose prediction (since prior_states already contains pose information)
        super().__init__()
        self.device = args.device

        self.act_fn = getattr(F, activation_function)
        self.dropout = getattr(F, 'dropout')
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2_lin = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc2_ang = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc3_lin = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.fc4_trans_x = nn.Linear(2 * hidden_size, hidden_size)
        self.fc5_trans_x = nn.Linear(hidden_size, 1)
        self.fc3_trans_y = nn.Linear(2 * hidden_size, 1)
        self.fc3_rot = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.fc4_rot = nn.Linear(2 * hidden_size, 1)

    def forward(self, state):
        hidden = self.act_fn(self.dropout(self.fc1(state), 0.5))
        hidden_lin = self.act_fn(self.dropout(self.fc2_lin(hidden), 0.5))
        hidden_ang = self.act_fn(self.dropout(self.fc2_ang(hidden), 0.5))
        trans_x = self.act_fn(self.dropout(self.fc3_lin(hidden_lin), 0.5))
        trans_x = self.act_fn(self.dropout(self.fc4_trans_x(trans_x), 0.5))
        trans_x = self.fc5_trans_x(trans_x)
        trans_y = self.fc3_trans_y(hidden_ang)
        zero_trans = torch.zeros([trans_x.shape[0], 1]).to(self.device)
        rot = self.act_fn(self.dropout(self.fc3_rot(hidden_ang), 0.5))
        rot = self.fc4_rot(rot)
        zero_rot = torch.zeros([rot.shape[0], 2]).to(self.device)

        return torch.cat([trans_x, trans_y, zero_trans, zero_rot, rot], dim=1)
