from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from model.swinunetr import SwinUNETR
from model.unet import UNet3D
import warnings

warnings.filterwarnings('ignore')
from collections import OrderedDict


class Universal_classify_modelV2(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, class_num, backbone='swinunetr', encoding='rand_embedding',
                 task='segment', split_patch=False, frozen_backbone=False, use_feat=False):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.split_patch = split_patch
        self.backbone_name = backbone
        self.task = task
        self.use_feat = use_feat
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      feature_size=48,
                                      drop_rate=0.0,
                                      attn_drop_rate=0.0,
                                      dropout_path_rate=0.0,
                                      use_checkpoint=False,
                                      )
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 48),
                nn.ReLU(inplace=True),
                nn.Conv3d(48, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'unet':
            self.backbone = UNet3D()
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                # nn.GroupNorm(16, 512),
                # nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                # nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )

        self.encoding = encoding

        self.classify_head = nn.Conv3d((256 + 256) * 3, 4, kernel_size=1, stride=1, padding=0)

        # self.class_num = class_num

        if frozen_backbone:
            self.frozen_params()

    def frozen_params(self):
        for n, p in self.backbone.named_parameters():
            p.requires_grad = False

    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def forward(self, x=None, **kwargs):
        '''
        :param organ_x: type list, [(bs, 1, h, w, d), ...]
        :param organ_id: type list, [0, 1, 2, 3]
        :return:
        '''

        if self.use_feat:
            x_feat = kwargs.get('feat').flatten(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            dec4, out = self.backbone(x)
            x_feat = self.GAP(dec4)
        classify_logits = self.classify_head(x_feat)

        return classify_logits


if __name__ == '__main__':
    model = Universal_model(img_size=(96, 96, 96),
                            in_channels=1,
                            out_channels=4,
                            class_num=4,
                            backbone='unet',
                            encoding='word_embedding',
                            task='classify'
                            )
    state_dict = torch.load('../pretrained_model/swinunetr.pth', map_location='cpu')
    x = torch.randn((1, 1, 96, 96, 32))

    out = model([x] * 4)
    print(out.shape)
    # store_dict = {}
    # for key, value in state_dict.items():
    #     name = '.'.join(key.split('.')[1:])
    #     # if 'organ_embedding' in key:
    #     #     value = value[[5, 0, 2, 1]]
    #     store_dict[name] = value
    # msg = model.load_state_dict(store_dict, strict=False)
    # print(msg)
    # m = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=1, groups=2)
    # print(m.weight.shape)

