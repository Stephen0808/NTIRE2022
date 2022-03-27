import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

from timm.models.vision_transformer import Block
from models.swin_135 import SwinTransformer
from torch import nn
from einops import rearrange


class RealCA(nn.Module):
    def __init__(self, num_fc, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(num_fc, num_fc)
        self.c_k = nn.Linear(num_fc, num_fc)
        self.c_v = nn.Linear(num_fc, num_fc)
        self.norm_fact = 1 / math.sqrt(num_fc)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class IQA(nn.Module):
    def __init__(self, embed_dim=768, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_channel_attn=4, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)

        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.conv_first = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp
        )

        self.real_ca = nn.ModuleList()
        for i in range(num_channel_attn):
            ca = RealCA((img_size // patch_size) ** 2)
            self.real_ca.append(ca)

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, num_outputs),
            nn.ReLU()
        )

        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, num_outputs),
            nn.Sigmoid()
        )
    
    def extract_feature(self, save_output):
        x7 = save_output.outputs[7][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x6 = save_output.outputs[6][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        for ca in self.real_ca:
            x = ca(x)
        x = rearrange(x, 'b c (h w) -> b (h w) c', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        x = self.conv_first(x)
        x = self.swintransformer(x)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, s.unsqueeze(0)), 0)
        return score
