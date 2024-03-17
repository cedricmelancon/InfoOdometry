#!/bin/bash

mkdir -p ckp/tmp/src/
mkdir -p ckp/pretrained_flownet/
cp /data/FlowNet2-S_checkpoint.pth.tar ckp/pretrained_flownet/FlowNet2S_checkpoint.pth.tar
mkdir tb_dir
mkdir eval

bash flownet_install.sh