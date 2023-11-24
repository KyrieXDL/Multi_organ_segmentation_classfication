from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from model.swinunetr import SwinUNETR
from model.unet import UNet3D
from model.resnet import resnet50
import warnings

warnings.filterwarnings('ignore')
from collections import OrderedDict


class Universal_classify_model(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, class_num, backbone='swinunetr', encoding='rand_embedding',
                 task='segment', split_patch=False, frozen_backbone=False, task_type='single', use_text_prompt=False,
                 share_weight=True, use_global=False, use_2d_encoder=False):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.split_patch = split_patch
        self.backbone_name = backbone
        self.task = task
        self.task_type = task_type
        self.use_text_prompt = use_text_prompt
        self.share_weight = share_weight
        self.share_head = share_weight
        self.use_global = use_global
        self.use_2d_encoder = use_2d_encoder
        self.feat_dim = 768
        # self.backbone = backbone

        if backbone == 'swin':
            self.backbone = SwinUNETR(img_size=img_size,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      feature_size=48,
                                      drop_rate=0.0,
                                      attn_drop_rate=0.0,
                                      dropout_path_rate=0.0,
                                      use_checkpoint=True,
                                      use_decoder='segment' in task
                                      )
            self.GAP = self.build_gap(use_text_prompt or use_global)
            if not share_weight:
                for i in range(class_num - 1):
                    swin = SwinUNETR(img_size=img_size,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     feature_size=48,
                                     drop_rate=0.0,
                                     attn_drop_rate=0.0,
                                     dropout_path_rate=0.0,
                                     use_checkpoint=True,
                                     use_decoder='segment' in task)
                    gap = self.build_gap(use_text_prompt or use_global)
                    setattr(self, f'backbone{i + 1}', swin)
                    setattr(self, f'GAP{i + 1}', gap)

            feat_dim = self.feat_dim if not use_text_prompt else 512
        elif backbone == 'unet':
            self.backbone = UNet3D(use_decoder='segment' in task)
            self.GAP = self.build_gap(use_text_prompt or use_global)

            if not share_weight:
                for i in range(class_num - 1):
                    unet = UNet3D(use_decoder='segment' in task)
                    gap = self.build_gap(use_text_prompt or use_global)
                    setattr(self, f'backbone{i + 1}', unet)
                    setattr(self, f'GAP{i + 1}', gap)
            feat_dim = 512

        if use_2d_encoder:
            self.encoder_2d = resnet50()
            self.proj_2d = nn.Linear(2048, 512)
            feat_dim += 512
            # state_dict = torch.load('../pretrained_model/resnet50.pth')
            # msg = model.load_state_dict(state_dict, strict=False)
            # print(msg)

        if use_global:
            self.global_backbone = UNet3D(use_decoder='segment' in task)
            self.global_GAP = self.build_gap(use_text_prompt or use_global)

        self.encoding = encoding

        if 'classify' in task:
            if task_type == 'single':
                self.classify_conv = nn.Conv3d(512 * class_num, 512, kernel_size=1, stride=1, padding=0)
                self.classify_head = nn.Conv3d(256 + 256, class_num, kernel_size=1, stride=1, padding=0)
            else:
                if not self.share_head:
                    self.classify_head = nn.ModuleList(
                        [nn.Conv3d(feat_dim, 1, kernel_size=1, stride=1, padding=0) for _ in range(class_num)])
                else:
                    self.classify_head = nn.Conv3d(feat_dim, 1, kernel_size=1, stride=1, padding=0)

        if use_text_prompt:
            if self.encoding == 'rand_embedding':
                self.organ_embedding = nn.Embedding(out_channels, 256)
            elif self.encoding == 'word_embedding':
                self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
                self.text_to_vision = nn.Linear(512, 256)
        self.class_num = class_num

        if frozen_backbone:
            self.frozen_params()

    def build_gap(self, use_text_prompt):
        if use_text_prompt:
            if self.backbone_name == 'unet':
                return nn.Sequential(
                    nn.GroupNorm(16, 512),
                    nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
                )
            elif self.backbone_name == 'swin':
                return nn.Sequential(
                    nn.GroupNorm(16, 768),
                    nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
                )
        else:
            if self.backbone_name == 'unet':
                return nn.Sequential(
                    nn.GroupNorm(16, 512),
                    nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                    # nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
                )
            elif self.backbone_name == 'swin':
                return nn.Sequential(
                    # nn.GroupNorm(16, self.feat_dim),
                    # nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                    # nn.Conv3d(768, 768, kernel_size=(3, 3, 1), stride=1, padding=0)
                    # nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
                )
        raise ValueError

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

    def forward(self, organ_x, global_x=None):
        '''
        :param organ_x: type list, [(bs, 1, h, w, d), ...]
        :param organ_id: type list, [0, 1, 2, 3]
        :return:
        '''
        if self.task_type == 'single':
            all_feature = []
            for x_in, organ_id in zip(organ_x, [0, 1, 2, 3]):
                dec4, out = self.backbone(x_in)
                x_feat = self.GAP(dec4)
                all_feature.append(x_feat)

            all_feature = torch.cat(all_feature, dim=1)
            # print(all_feature.shape)
            all_feature = self.classify_conv(all_feature)
            output = self.classify_head(all_feature).squeeze_(-1).squeeze_(-1).squeeze_(-1)

            # output
        else:
            all_pred = []
            global_feat = None
            if self.use_global:
                feat = self.global_backbone(global_x)
                global_feat = self.global_GAP(feat)

            for x_in, organ_id in zip(organ_x, [0, 1, 2, 3]):
                organ_pred = self.extract_feature(x_in, organ_id, global_feat)
                all_pred.append(organ_pred)
            output = torch.cat(all_pred, dim=1)
        # print('output ', output.shape)
        return output

    def extract_feature(self, x_in, organ_id, global_feat=None):
        # print('extract feat')
        if self.share_weight:
            dec4, out = self.backbone(x_in)
            x_feat = self.GAP(dec4)
        else:
            names = ['backbone', 'backbone1', 'backbone2', 'backbone3']
            gap_names = ['GAP', 'GAP1', 'GAP2', 'GAP3']
            backbone = getattr(self, names[organ_id - 1])
            gap = getattr(self, gap_names[organ_id - 1])
            dec4, out = backbone(x_in)
            x_feat = gap(dec4)
            # print("x_feat ", x_feat.shape)
        if self.use_2d_encoder:
            feat_list = []
            # print(x_in.shape)
            for i in range(x_in.shape[-1]):
                x_in_slice = x_in[..., i].repeat(1, 3, 1, 1)
                slice_feat = self.encoder_2d(x_in_slice)
                feat_list.append(slice_feat)
            slice_feat = torch.stack(feat_list, dim=1)
            slice_feat = self.proj_2d(slice_feat.mean(1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x_feat = torch.cat([x_feat, slice_feat], dim=1)

        if global_feat is not None:
            x_feat = torch.cat([x_feat, global_feat], dim=1)

        if self.use_text_prompt:
            if self.encoding == 'rand_embedding':
                task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
            elif self.encoding == 'word_embedding':
                task_encoding = F.relu(self.text_to_vision(self.organ_embedding[organ_id: organ_id + 1]))
                task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        # x_feat = self.GAP(dec4)
        b = x_feat.shape[0]
        classify_logits_array = []
        for i in range(b):
            ## text emb & image encoder feat
            if self.use_text_prompt:
                x_cond = torch.cat([x_feat[i].unsqueeze(0), task_encoding], 1)
            else:
                x_cond = x_feat[i].unsqueeze(0)
            ## classify head
            if self.share_weight:
                classify_logits = self.classify_head(x_cond)
            else:
                classify_logits = self.classify_head[organ_id](x_cond)
            # print('classify_logits ', classify_logits.shape)
            classify_logits = classify_logits.squeeze_(-1).squeeze_(-1).squeeze_(-1).squeeze(1)
            classify_logits_array.append(classify_logits)

        classify_out = torch.stack(classify_logits_array, dim=0)
        # print('classify_out ', classify_out.shape)
        return classify_out


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

