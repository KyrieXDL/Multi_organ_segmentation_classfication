from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from model.swinunetr import SwinUNETR
from model.unet import UNet3D


# from model.DiNTS import TopologyInstance, DiNTS
# from model.Unetpp import BasicUNetPlusPlus


class Universal_model(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, class_num=4, backbone='swinunetr',
                 encoding='rand_embedding', task='segment', split_patch=False, frozen_backbone=False):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.split_patch = split_patch
        self.backbone_name = backbone
        self.task = task

        if self.backbone_name == 'swin':
            self.backbone = SwinUNETR(img_size=img_size,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      feature_size=48,
                                      drop_rate=0.0,
                                      attn_drop_rate=0.0,
                                      dropout_path_rate=0.0,
                                      use_checkpoint=False,
                                      use_decoder='segment' in task
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
        elif self.backbone_name == 'unet':
            self.backbone = UNet3D(use_decoder='segment' in task)
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise ValueError

        self.encoding = encoding

        if 'segment' in task:
            weight_nums, bias_nums = [], []
            weight_nums.append(8 * 8)
            weight_nums.append(8 * 8)
            weight_nums.append(8 * 1)
            bias_nums.append(8)
            bias_nums.append(8)
            bias_nums.append(1)
            self.weight_nums = weight_nums
            self.bias_nums = bias_nums
            self.controller = nn.Conv3d(256 + 256, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0)

        if 'classify' in task:
            self.classify_head = nn.Conv3d(256 + 256, 1, kernel_size=1, stride=1, padding=0)

        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.class_num = class_num

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

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]] = 1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)  # num_class
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[
                        :num_layers]  # 卷积权重 torch.Size([num_class, 8*8]), torch.Size([num_class, 8*8]), torch.Size([num_class, 8*1])
        bias_splits = params_splits[
                      num_layers:]  # 卷积偏置 torch.Size([num_class, 8]), torch.Size([num_class, 8]), torch.Size([num_class, 1])

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features  # torch.Size([1, num_class*8, 96, 96, 96])
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def unfold_images(self, images, patch_size=(96, 96), patch_depth=48):
        batch_size = images.shape[0]
        x = images.squeeze(1)

        n = x.shape[-1] // patch_depth
        if x.shape[-1] % patch_depth:
            n += 1
        patches = []
        for i in range(n):
            if i < n - 1:
                tmp_x = x[..., :patch_depth].permute(0, 3, 1, 2)
            else:
                tmp_x = x[..., -patch_depth:].permute(0, 3, 1, 2)
            output = F.unfold(tmp_x, kernel_size=patch_size, stride=patch_size)
            patches.append(output)
        patches = torch.cat(patches, dim=-1)
        num_patches = patches.shape[-1]
        patches = patches.permute(0, 2, 1).view(batch_size * num_patches, -1)
        x = patches.view(-1, patch_depth, *patch_size).permute(0, 2, 3, 1).unsqueeze(1)
        return x

    def forward(self, image, **kwargs):
        x_in = image
        dec4, out = self.backbone(x_in)
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.encoding == 'word_embedding':
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
            # task_encoding = F.relu(self.text_to_vision(self.organ_embedding[[0, 1, 2, 5]]))
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x_feat = self.GAP(dec4)
        b = x_feat.shape[0]
        segment_logits_array, classify_logits_array = [], []
        for i in range(b):
            ## text emb & image encoder feat
            x_cond = torch.cat([x_feat[i].unsqueeze(0).repeat(self.class_num, 1, 1, 1, 1), task_encoding],
                               1)  # torch.Size([num_class, 256 + 256, 1, 1, 1])

            ## classify head
            if 'classify' in self.task:
                classify_logits = self.classify_head(x_cond)
                classify_logits = classify_logits.squeeze_(-1).squeeze_(-1).squeeze_(-1).squeeze(1)
                classify_logits_array.append(classify_logits)

            ## segment head
            ## condition conv params
            if 'segment' in self.task:
                params = self.controller(x_cond)  # torch.Size([num_class, num_weight + num_bias, 1, 1, 1])
                params.squeeze_(-1).squeeze_(-1).squeeze_(-1)  # torch.Size([num_class, num_weight + num_bias)

                ##condition conv
                head_inputs = self.precls_conv(out[i].unsqueeze(0))  # torch.Size([1, 8, 96, 96, 96])
                head_inputs = head_inputs.repeat(self.class_num, 1, 1, 1, 1)  # torch.Size([num_class, 8, 96, 96, 96])
                N, _, D, H, W = head_inputs.size()
                head_inputs = head_inputs.reshape(1, -1, D, H, W)  # torch.Size([1, num_class*8, 96, 96, 96])
                weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

                logits = self.heads_forward(head_inputs, weights, biases, N)
                segment_logits_array.append(logits.reshape(1, -1, D, H, W))  # torch.Size([1, num_class, 96, 96, 96])

        if 'segment' in self.task:
            segment_out = torch.cat(segment_logits_array, dim=0)

        if 'classify' in self.task:
            classify_out = torch.stack(classify_logits_array, dim=0)

        if 'segment' in self.task and 'classify' in self.task:
            return segment_out, classify_out
        elif 'segment' in self.task:
            return segment_out
        else:
            return classify_out


if __name__ == '__main__':
    model = Universal_model(img_size=(96, 96, 96),
                            in_channels=1,
                            out_channels=4,
                            backbone='unet',
                            encoding='word_embedding'
                            )

    x = torch.randn((2, 1, 96, 128, 128))
    out = model(x)
    print(out.shape)

    # m = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=1, groups=2)
    # print(m.weight.shape)
