import os
import torch

try:
    from info_odometry.flownet import FlowNetS
    from info_odometry.flownet.submodules import *
except:
    from .info_odometry.flownet import FlowNetS
    from .info_odometry.flownet.submodules import *


class OdometryModelFlownet2S(FlowNetS.FlowNetS):
    def __init__(self, args, batch_norm=False):
        super(OdometryModelFlownet2S, self).__init__(input_channels=6, batchNorm=batch_norm)

        self.rgb_max = 255.0
        self.flownet_model_name = args.flownet_model

        if self.flownet_model_name != 'none' and args.img_prefeat == 'none':
            if args.train_img_from_scratch:
                raise ValueError('if --flownet_model -> --train_img_from_scratch should not be used')

    def load_model(self, model_path):
        assert os.path.exists(model_path)

        #resume_ckp = '/data/results/ckp/pretrained_flownet/{}_checkpoint.pth.tar'.format(self.flownet_model_name)
        flow_ckp = torch.load(model_path)
        self.load_state_dict(flow_ckp['state_dict'])

    def forward(self, inputs):
        rgb_mean = (inputs.contiguous()
                    .view(inputs.size()[:2] + (-1,))
                    .mean(dim=-1)
                    .view(inputs.size()[:2] + (1, 1, 1,)))
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
