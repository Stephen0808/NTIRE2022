import os
import torch
import numpy as np
import cv2 


class PIPAL(torch.utils.data.Dataset):
    def __init__(self, ref_path, dis_path, txt_file_name, transform):
        super(PIPAL, self).__init__()
        self.ref_path = ref_path
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        # count = 0
        ref_files_data, dis_files_data, score_data, cls_data = [], [], [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                ref = dis[:5] + '.bmp'
                score = float(score)
                ref_files_data.append(ref)
                dis_files_data.append(dis)
                score_data.append(score)
                cls_data.append(int(line.split('_')[1]))
                # count += 1
                # if count == 116:
                #     break
        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'r_img_list': ref_files_data, 'd_img_list': dis_files_data, 'score_list': score_data,'class': cls_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['r_img_list'])
    
    def __getitem__(self, idx):
        # r_img: H x W x C -> C x H x W
        r_img_name = self.data_dict['r_img_list'][idx]
        r_img = cv2.imread(os.path.join(self.ref_path, r_img_name), cv2.IMREAD_COLOR)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        r_img = np.array(r_img).astype('float32') / 255
        r_img = np.transpose(r_img, (2, 0, 1))

        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        
        score = self.data_dict['score_list'][idx]
        sample = {
            'r_img_org': r_img,
            'd_img_org': d_img,
            'score': score,
            
        }
        if self.transform:
            sample = self.transform(sample)
        sample['class'] = int(d_img_name.split('_')[1])
        return sample
