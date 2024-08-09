from torch import nn
from torch.nn import functional as F

flownet_featsize = {
    'kitti': 1024 * 5 * 19,
    'euroc': 1024 * 8 * 12,
    'mit': 81920
}


class OdometryModelEncoder(nn.Module):
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
