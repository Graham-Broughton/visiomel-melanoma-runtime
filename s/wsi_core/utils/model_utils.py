from collections import OrderedDict

import torch
import torch.nn as nn


class ResNetFeatureExtractor(nn.Module):
    def __init__(self,backbone_net):
        super(ResNetFeatureExtractor,self).__init__()
        self.allowed_module_keys = [
            "conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4"
        ]
        backbone_net = [
            (k,backbone_net._modules[k]) for k in self.allowed_module_keys
        ]
        self.backbone_net = nn.Sequential(OrderedDict(backbone_net))

    def forward(self,inp):
        val = self.backbone_net(inp)
        val = torch.mean(val,dim=[2,3])
        return val

    def convert_state_dict(self,state_dict):
        return OrderedDict(
            [
                (k, v)
                for k, v in state_dict.items()
                if k.split(".")[0] in self.allowed_module_keys
            ]
        )

    def load_state_dict(self,state_dict):
        state_dict = self.convert_state_dict(state_dict)
        self.backbone_net.load_state_dict(state_dict)
