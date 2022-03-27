import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange
import numpy as np
from transformers import Transformer
from posencode import PositionEmbeddingSine


# utils
def DCT_mat(size):
	m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
	return m

def generate_filter(start, end, size):
	return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
	return 2. * torch.sigmoid(x) - 1.

# Global attention
class GA(nn.Module):
	def __init__(self) -> None:
		super(GA, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv1 = nn.Conv2d(768*4, 768*4, 3, 1, 1)

	def forward(self, x, x_high):
		_x = x
		x_high = self.conv1(x)
		x = self.avg_pool(x)
		x_high = x * x_high
		return _x + 0.2 * x_high

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
		# 2 kernel
		y_list = [x]
		for i in range(2):
			x_pass = self.filters[i](x_freq)  # [N, 3, 224, 224]
			
			y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 224, 224]
			y_list.append(y)
		out = torch.cat(y_list, dim=1)    # [N, 9, 224, 224]
		return out


class IQCA(nn.Module):
	def __init__(self, num_fc):
		super().__init__()
		self.c_q = nn.Linear(num_fc, num_fc)
		self.c_k = nn.Linear(num_fc, num_fc)
		self.c_v = nn.Linear(num_fc, num_fc)
		self.softmax = nn.Softmax(dim=-1)
		self.proj = nn.Linear(num_fc, num_fc)
		self.proj_drop = nn.Dropout(0.1)

	def forward(self, x):
		B, C, N = x.shape
		q = self.c_q(x)
		k = self.c_k(x)
		v = self.c_v(x)

		attn = q @ k.transpose(-2, -1)
		attn = self.softmax(attn) / (N ** 0.5)
		attn = self.proj_drop(attn)
		score = (attn @ v).transpose(1, 2).reshape(B, C, N)

		score = self.proj_drop(score)
		return score

class FCA(nn.Module):
	def __init__(self, num_fc):
		super().__init__()
		self.c_q = nn.Linear(num_fc, num_fc)
		self.c_k = nn.Linear(num_fc, num_fc)
		self.c_v = nn.Linear(num_fc, num_fc)
		self.softmax = nn.Softmax(dim=-1)
		self.proj = nn.Linear(num_fc, num_fc)
		self.proj_drop = nn.Dropout(0.1)

	def forward(self, x, x_filter):
		B, C, N = x.shape
		q = self.c_q(x_filter)
		k = self.c_k(x)
		v = self.c_v(x)

		attn = q @ k.transpose(-2, -1)
		attn = self.softmax(attn) / (N ** 0.5)
		attn = self.proj_drop(attn)
		score = (attn @ v).transpose(1, 2).reshape(B, C, N)

		score = self.proj_drop(score)
		return score

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
		self.FAD = FAD_Head(224)
		self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
		# self.FCA = FCA(784)
		self.globalattn = GA()
		# save intermediate layers
		self.save_output = SaveOutput()
		hook_handles = []
		for layer in self.vit.modules():
			if isinstance(layer, Block):
				handle = layer.register_forward_hook(self.save_output)
				hook_handles.append(handle)



		self.transformer = Transformer(d_model=768,nhead=4,
									   num_encoder_layers=2,
									   dim_feedforward=4*768,
									   normalize_before=False,
									   dropout = 0.1)
		

		self.position_embedding = PositionEmbeddingSine(768 // 2, normalize=True)
		self.pos_enc = self.position_embedding(torch.ones(1, 768, 28, 28*2)).cuda()



		self.conv_first = nn.Conv2d(embed_dim * 4, embed_dim, 3, 1, 1)
		self.conv_second = nn.Conv2d(embed_dim * 4, embed_dim, 3, 1, 1)



		self.swintransformer = SwinTransformer(
			patches_resolution=self.patches_resolution,
			depths=depths,
			num_heads=num_heads,
			embed_dim=embed_dim,
			window_size=window_size,
			dim_mlp=dim_mlp
		)





		self.ca_scale = ca_scale
		self.iqca = nn.ModuleList()
		for i in range(self.num_channel_attn):
			ca = IQCA(self.patches_resolution[0] * self.patches_resolution[1])
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


	def extract_feature_high(self, save_output):
		x0 = save_output.outputs[0][:, 1:]
		x1 = save_output.outputs[1][:, 1:]
		x2 = save_output.outputs[2][:, 1:]
		x3 = save_output.outputs[3][:, 1:]
		# x5 = save_output.outputs[5][:, 1:]
		# x7 = save_output.outputs[7][:, 1:]
		
		# x2 = save_output.outputs[2][:, 1:]
		# x4 = save_output.outputs[4][:, 1:]
		# x6 = save_output.outputs[6][:, 1:]
		# x8 = save_output.outputs[8][:, 1:]
		# x9 = save_output.outputs[9][:, 1:]
		# x10 = save_output.outputs[10][:, 1:]
		# x11 = save_output.outputs[11][:, 1:]
		# return torch.cat((x8, x9, x10, x11), dim=2)
		return torch.cat((x0, x1, x2, x3), dim=2)

	def forward(self, inp):
		
		# print(inp.size())
		img_cat = self.FAD(inp)
		x, x_low, x_high = torch.chunk(img_cat, 3, 1)
		x_high = x_low   # convert
		x = self.vit(x)
		x = self.extract_feature(self.save_output)
		self.save_output.outputs.clear()
		# print(x.size())
		
		x_high = self.vit(x_high)
		x_high = self.extract_feature_high(self.save_output)
		self.save_output.outputs.clear()

		x = rearrange(x, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
		x_high = rearrange(x_high, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)

		x = self.conv_first(x)
		x_high = self.conv_second(x_high)

		x = rearrange(x, 'b c h w-> b (h w) c', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
		x_high = rearrange(x_high, 'b c h w-> b (h w) c', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)


		x = torch.cat((x, x_high), dim=1)
		# print(x.size())
		# x = rearrange(x, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size, w=2*self.img_size // self.patch_size)

		pos_enc = self.pos_enc.repeat(x.shape[0],1,1,1).contiguous()
		x = self.transformer(x, pos_enc)
		# print(x.size())
		# x = rearrange(x, 'b c h w-> b (h w) c', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
		# x = x[:, :28*28]
		# print(x.size())
		# print(x.size())
		# x_high = rearrange(x_high, 'b (h w) c -> b c (h w)', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)

		# _x = self.FCA(x, x_high)
		# x = _x + x
		
		# x = self.globalattn(x, x_high)
		# x = x + x_high
		# print(x.size())
		x = rearrange(x, 'b (h w) c-> b c h w', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
		
		x = self.swintransformer(x)
		x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)

		for ca in self.iqca:
		    _x = x
		    x = ca(x)
		    x = x + _x

		x = rearrange(x, 'b c (h w) -> b (h w) c', h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
		score = torch.tensor([]).cuda()
		for i in range(x.shape[0]):
			f = self.fc_score(x[i])
			w = self.fc_weight(x[i])
			s = torch.sum(f * w) / torch.sum(w)
			score = torch.cat((score, s.unsqueeze(0)), 0)
		return score


if __name__ == '__main__':
	model = IQA()
	print(model)