from tkinter import X
from tokenize import group
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
sys.path.append('/home/ysd21/VIT')
from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super(FeedForward, self).__init__()
        self.norm2 = LayerNorm(dim)
        hidden_features = int(dim*2)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        _x = x
        x = self.norm2(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x + _x


class MDTA(nn.Module):
    def __init__(self, dim, num_heads) -> None:
        super(MDTA, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b,c,h,w = x.shape
        _x = x
        x = self.norm1(x)
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out + _x

class BasicLayer(nn.Module):
    def __init__(self, dim, heads) -> None:
        super(BasicLayer, self).__init__()
        self.attn = MDTA(dim, heads)
        self.feedforward = FeedForward(dim)

    def forward(self, x):
        x = self.attn(x)
        # x = self.feedforward(x)
        return x
class IQCA(nn.Module):
    def __init__(self, channel, hw):
        super().__init__()
        self.norm1 = nn.LayerNorm(channel)
        self.norm2 = nn.LayerNorm(channel)
        self.dconv_q = nn.ModuleList([
                    nn.Conv2d(channel, channel, 1, 1), 
                    nn.Conv2d(channel, channel, 3, 1, 1, groups=channel),
                    nn.Conv2d(channel, channel, 1, 1)
        ])
        self.dconv_k = nn.ModuleList([
                    nn.Conv2d(channel, channel, 1, 1), 
                    nn.Conv2d(channel, channel, 3, 1, 1, groups=channel),
                    nn.Conv2d(channel, channel, 1, 1)
        ])
        self.dconv_v = nn.ModuleList([
                    nn.Conv2d(channel, channel, 1, 1),
                    nn.Conv2d(channel, channel, 3, 1, 1, groups=channel),
                    nn.Conv2d(channel, channel, 1, 1)
        ])

        self.score = nn.ModuleList([
                    nn.Conv2d(channel, channel, 1, 1),
                    nn.Conv2d(channel, channel, 3, 1, 1, groups=channel),
                    nn.Conv2d(channel, 4*channel, 1, 1)
        ])

        self.gate = nn.ModuleList([
                    nn.Conv2d(channel, channel, 1, 1),
                    nn.Conv2d(channel, channel, 3, 1, 1, groups=channel),
                    nn.Conv2d(channel, 4*channel, 1, 1)
        ])

        self.downchannel = nn.Conv2d(4*channel, channel, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(hw, hw)
        self.proj_drop = nn.Dropout(0.1)
        self.gelu = nn.GELU()



    def forward(self, x):
        B, C, H, W = x.shape
        _x = x
        x = self.norm1(x)
        q = self.dconv_q(x).flatten(2)
        k = self.dconv_k(x).flatten(2)
        v = self.dconv_v(x).flatten(2)
        attn = q.transpose(-2, -1) @ k
        attn = self.proj_drop(attn)
        attn = self.softmax(attn) / ((H*W)**0.5)
        x = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = _x + x

        _x = x
        x = self.norm2(X)
        score = self.score(x)
        gate = self.gelu(self.gate(x))
        x = score * gate
        x = self.downchannel(x)
        x = _x + x
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
                    img_size=224, num_channel_attn=4, ca_scale=0.13, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channel_attn = num_channel_attn
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)

        # save intermediate layers
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
            embed_dim=embed_dim*4,
            window_size=window_size,
            dim_mlp=dim_mlp
        )
        self.ca_scale = ca_scale
        self.iqca = nn.ModuleList()
        for _ in range(self.num_channel_attn):
            ca = IQCA(embed_dim, self.patches_resolution[0] * self.patches_resolution[1])
            self.iqca.append(ca)
            
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

        self.MDTA = nn.ModuleList()
        for _ in range(num_channel_attn):
            rca = BasicLayer(embed_dim*4, 4)
            self.MDTA.append(rca)

    def extract_feature(self, save_output):
        # x3 = save_output.outputs[3][:, 1:]
        # x5 = save_output.outputs[5][:, 1:]
        # x7 = save_output.outputs[7][:, 1:]
        
        # x2 = save_output.outputs[2][:, 1:]
        # x4 = save_output.outputs[4][:, 1:]
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        # x10 = save_output.outputs[10][:, 1:]
        # x11 = save_output.outputs[11][:, 1:]
        # return torch.cat((x8, x9, x10, x11), dim=2)
        return torch.cat((x6, x7, x8, x9), dim=2)

    def forward(self, x):
        _x = self.vit(x)
        self.vit.eval()
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()        
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        # for ca in self.MDTA:
        #     x = ca(x)
       
        
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        
        x = self.swintransformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        x = self.conv_first(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            w = torch.clamp(w, 0, 0.7)
            s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, s.unsqueeze(0)), 0)
        return score


if __name__ == '__main__':
    MDSA = IQA(768, 4)
    x = torch.randn((8, 3, 224, 224))
    out = MDSA(x)
    # out = FN(out)
    print(out, out.size())