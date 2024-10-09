from torch import nn
import torch
from .info_model import flownet_featsize
from torch.nn import functional as F


class SymbolicEncoder(nn.Module):
    def __init__(self, args, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(flownet_featsize[args.flowfeat_size_dataset], embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    # @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, activation_function='relu', batch_norm=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = img_conv(batch_norm, 6, 16, kernel_size=7, stride=2)
        self.conv2 = img_conv(batch_norm, 16, 32, kernel_size=5, stride=2)
        self.conv3 = img_conv(batch_norm, 32, 64, kernel_size=3, stride=2)
        self.conv4 = img_conv(batch_norm, 64, 128, kernel_size=3, stride=2)
        self.conv5 = img_conv(batch_norm, 128, 256, kernel_size=3, stride=2)
        self.conv6 = img_conv(batch_norm, 256, 256, kernel_size=3, stride=2)
        self.conv6_1 = img_conv(batch_norm, 256, 256, kernel_size=3, stride=2)

        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    # @jit.script_method
    def forward(self, observation):
        # observation: [batch, 6, H, W]
        hidden = self.conv1(observation)
        hidden = self.conv2(hidden)
        hidden = self.conv3(hidden)
        hidden = self.conv4(hidden)
        hidden = self.conv5(hidden)
        hidden = self.conv6(hidden)
        hidden = self.conv6_1(hidden)
        pdb.set_trace()
        hidden = hidden.view(hidden.size()[0], -1)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu'):
    """
    hidden_size: not used (for code consistency in main.py)
    """
    if symbolic:  # use FlowNet2S features
        return SymbolicEncoder(args, embedding_size, activation_function)
    else:  # train from scratch
        return VisualEncoder(embedding_size, activation_function)