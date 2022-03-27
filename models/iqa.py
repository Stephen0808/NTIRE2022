import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.rcan_channel_attention import ResidualGroup, ResidualGroupXY
from timm.models.vision_transformer import Block
from torch import nn
from einops import rearrange, repeat
import numpy as np

def extract_feature(save_output):
    # print(save_output.outputs[3].size())
    feat = torch.cat(
    (
        # save_output.outputs[0][:, 1:],
        # save_output.outputs[1][:, 1:],
        # save_output.outputs[2][:, 1:],
        # save_output.outputs[3][:, 1:],
        # save_output.outputs[4][:, 1:],
        save_output.outputs[5][:, 1:],
        save_output.outputs[6][:, 1:],
        save_output.outputs[7][:, 1:],
        save_output.outputs[8][:, 1:],
        # save_output.outputs[9][:, 1:],
    ),
    dim=2)

    return feat

# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []

# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        # print(self.base)
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size //(8**0.5))
        middle_filter = Filter(size, size //(8**0.5), size //(4**0.5))
        high_filter = Filter(size, size //(8**0.5), size*2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 224, 224]
        # 4 kernel
        y_list = [x]
        for i in range(2):
            x_pass = self.filters[i](x_freq)  # [N, 3, 224, 224]
            
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 224, 224]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 9, 224, 224]
        return out

class RB(nn.Module):
    def __init__(self, embed_dim):
        super(RB, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        _x = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = 0.2 * x + _x
        return x


class IQA(nn.Module):
    def __init__(self, in_channels=3, embed_dim=72, num_outputs=1, dim_mlp_head=512,
                    patch_size=8, drop=0.1, patches_resolution=(18, 18), depths=[2, 2], window_size=9,
                    dim_mlp=512, num_heads=[4, 4], num_decoder_layers=2, img_size=224, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.FAD = FAD_Head(224)
        # save intermediate layers
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        
        self.conv_first = nn.Conv2d(768*4, 768, 1, 1, 0)
        self.conv_second = nn.Conv2d(768*2, 768, 1, 1, 0)
        self.merge = nn.Conv2d(768*2, 768, 1, 1, 0)

        self.rb1 = RB(768)
        self.rb2 = RB(768)
        self.rb3 = RB(768)
        
        self.fc_score = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(768, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(768, num_outputs),
            nn.Sigmoid()
        )
    
    def _weak_details(self, x, h, w):
        B, C, H, W = x.shape
        _x_weak = F.interpolate(x, size=[h, w], mode="bilinear", align_corners=False)
        return _x_weak

    def forward_resuidalblock(self, inp):
        self.vit.eval()
        # img_cat = self.FAD(inp)
        # x, x_low, x_high = torch.chunk(img_cat, 3, 1)

        x = self.vit(inp)
        x = extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # x_high = self.vit(x_high)
        # x_high = extract_feature(self.save_output)
        # self.save_output.outputs.clear()

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        # x_high = rearrange(x_high, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        x = self.conv_first(x)
        # x_high = self.conv_second(x_high)
        # x_merge = torch.cat((x, x_high), dim=1)
        # x = self.merge(x_merge)
    
        _x = x
        x = self.rb1(x)
        # x += _x
        x = self.rb2(x)
        # x += _x
        x = self.rb3(x)
        # x = 0.2 * x + _x
        x += _x

        x = rearrange(x, 'b c h w -> b (h w) c')
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, s.unsqueeze(0)), 0)
        return score
    
    def forward(self, x):
        x = self.forward_resuidalblock(x)
        return x
