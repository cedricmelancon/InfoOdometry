from torch import nn
import torch
from .info_model import flownet_featsize


class SymbolicObservationModel(nn.Module):
    def __init__(self, args, belief_size, state_size, embedding_size, activation_function='relu',
                 observation_type='visual'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        if observation_type == 'visual':
            self.fc3 = nn.Linear(embedding_size, flownet_featsize[args.flowfeat_size_dataset])
        elif observation_type == 'imu':
            self.fc3 = nn.Linear(embedding_size, 6 * 11)  # for each frame-pair

    # @jit.script_method
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation




class VisualObservationModel(nn.Module):
    def __init__(self, belief_size, state_size, embedding_size, activation_function='relu', batch_norm=False):
        raise NotImplementedError('need to check the final output image size')
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.upconv6 = img_upconv(batch_norm, 256, 256)
        self.upconv5 = img_upconv(batch_norm, 256, 128)
        self.upconv4 = img_upconv(batch_norm, 128, 64)
        self.upconv3 = img_upconv(batch_norm, 64, 32)
        self.upconv2 = img_upconv(batch_norm, 32, 16)
        self.upconv1 = img_upconv(batch_norm, 16, 6)
        # self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        # self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        # self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        # self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    # @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.upconv6(hidden)
        hidden = self.upconv5(hidden)
        hidden = self.upconv4(hidden)
        hidden = self.upconv3(hidden)
        hidden = self.upconv2(hidden)
        observation = self.upconv1(hidden)
        return observation


def ObservationModel(symbolic, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu',
                     observation_type='visual'):
    """
    hidden_size: not used (for code consistency in main.py)
    """
    if symbolic:  # use Flownet2S features
        return SymbolicObservationModel(args, belief_size, state_size, embedding_size, activation_function,
                                        observation_type)
    else:  # train from scratch
        if observation_type != 'visual': raise ValueError('error: observation must be visual for symbolic being False')
        return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)