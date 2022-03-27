import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.eval_process_image import ToTensor, Normalize
# from utils.process_image import Normalize
from data.ntire2022 import NTIRE2022
from tqdm import tqdm
from utils.process_image import crop_image

device = torch.device("cuda:{}".format(0))

def five_point_crop(idx, d_img):
	new_h = 224
	new_w = 224
	b, c, h, w = d_img.shape
	if idx == 0:
		top = 0
		left = 0
	elif idx == 1:
		top = 0
		left = w - new_w
	elif idx == 2:
		top = h - new_h
		left = 0
	elif idx == 3:
		top = h - new_h
		left = w - new_w
	elif idx == 4:
		center_h = h // 2
		center_w = w // 2
		top = center_h - new_h // 2
		left = center_w - new_w // 2
	d_img_org = crop_image(top, left, 224, img=d_img)
	return d_img_org

def setup_seed(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	# np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def eval_epoch(config, net, test_loader):
	with torch.no_grad():
		net.eval()
		name_list = []
		pred_list = []
		with open(config.valid_path + '/output.txt', 'w') as f:
			for data in tqdm(test_loader):
				d_img_org = data['d_img_org'].to(device)
				# print(d_img_org.size())
				d_img_org = d_img_org.squeeze(0)
				c, h, w = d_img_org.size()
				pred_scores = 0
				for i in range(5):
					x_d = data['d_img_org'].cuda()
					x_d = five_point_crop(i, d_img=x_d)
					# print(x_d.size())
					# print(net(x_d).size())
					pred_scores += net(x_d)

				# for i in range(20):
				# 	top = np.random.randint(0, h - 224)
				# 	left = np.random.randint(0, w - 224)
					
				# 	img_ = d_img_org[:, top: top+224, left: left+224].unsqueeze(0)
				# 	# print(img.size())
				# 	img_ = torch.as_tensor(img_.to(device))
				# 	pred = net(img_)
				# 	# print(pred.size())
				# 	pred_scores = pred_scores + pred.cpu().tolist()
				# 	# print(pred_scores)
				# pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 20)), axis=1)
				# print(pred_scores)
				pred_scores = pred_scores.cpu().numpy()
				pred_scores /= 5
				d_name = data['d_name']
				
				name_list.extend(d_name)
				pred_list.extend(pred_scores)
				# print(pred_list)

			for i in range(len(name_list)):
				print(name_list[i] + ',' + str(pred_list[i]))
				f.write(name_list[i] + ',' + str(pred_list[i]) + '\n')
			print(len(name_list))
		f.close()


def sort_file(file_path):
	f2 = open(file_path, "r")
	lines = f2.readlines()
	ret = []
	for line in lines:
		line = line[:-1]
		ret.append(line)
	ret.sort()

	with open(config.valid_path + '/output.txt', 'w') as f:
		for i in ret:
			f.write(i + '\n')


if __name__ == '__main__':
	cpu_num = 1
	os.environ['OMP_NUM_THREADS'] = str(cpu_num)
	os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
	os.environ['MKL_NUM_THREADS'] = str(cpu_num)
	os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
	os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
	torch.set_num_threads(cpu_num)

	setup_seed(20)

	# config file
	config = Config({
		# dataset path
		"db_name": "PIPAL",                                                     # database name [ PIPAL | LIVE | CSIQ | TID2013 ]
		"val_ref_path": "/mnt/data_16TB/ysd21/IQA/NTIRE2022_NR_Valid_Dis/",
		"val_dis_path": "/mnt/data_16TB/ysd21/IQA/NTIRE2022_NR_Valid_Dis/",

		# optimization
		"batch_size": 1,
		"learning_rate": 1e-4,
		"weight_decay": 1e-5,
		"n_epoch": 300,
		"T_max": 50,                        # cosine learning rate period (iteration)
		"eta_min": 0,                        # mininum learning rate

		# device
		"num_workers": 8,
		
		# load & save checkpoint
		"valid_path": "./final_test_output",
		"model_path": "/home/ysd21/VIT/output/models/epoch_X"
	})


	logging.info(config)
	
	# data load
	val_dataset = NTIRE2022(
		ref_path=config.val_ref_path,
		dis_path=config.val_dis_path,
		transform=transforms.Compose(
			[
				# Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
				Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
				ToTensor()
			]
		),
	)
	val_loader = DataLoader(
		dataset=val_dataset,
		batch_size=config.batch_size,
		num_workers=config.num_workers,
		drop_last=True,
		shuffle=False
	)

	net = torch.load(config.model_path)['model']
	net = net.to(device)

	# train & validation
	losses, scores = [], []
	eval_epoch(config, net, val_loader)

	sort_file(config.valid_path + 'output.txt')
		