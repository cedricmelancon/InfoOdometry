try:
    from flownet import FlowNetS
    from flownet.submodules import *
except:
    from .flownet import FlowNetS
    from .flownet.submodules import *
'Parameter count = 162,518,834'


class FlowNet2S(FlowNetS.FlowNetS):
    def __init__(self, rgb_max, batchNorm=False, div_flow=20):
        super(FlowNet2S, self).__init__(input_channels=6, batchNorm=batchNorm)
        self.rgb_max = 255.0
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        out_conv6_1 = self.conv6_1(out_conv6)

        return out_conv6_1