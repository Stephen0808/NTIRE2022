import os
import cv2
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
import sys
sys.path.append('/home/ysd21/VIT/models')
from torchvision import transforms
from torch.utils.data import DataLoader
from models.iqa_135 import IQA
from config import Config
from utils.process_image import RandCrop, ToTensor, RandHorizontalFlip, Normalize, crop_image
from scipy.stats import spearmanr, pearsonr
from data.pipal_nr import PIPAL
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm


# os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def five_point_crop(idx, d_img, config):
    new_h = config.crop_size
    new_w = config.crop_size
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
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)

    return d_img_org


""" train model """
def train_epoch(config, epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # print(net)
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    cls_epoch = []
    preds, labs = [[] for i in range(7)], [[] for i in range(7)]
    clshash = {
        '0':0,
        '1':0,
        '2':0,
        '3':0,
        '4':0,
        '5':0,
        '6':0
    }
    clsnum = {
        '0':12,
        '1':16,
        '2':10,
        '3':24,
        '4':13,
        '5':14,
        '6':27
    }
    for data in tqdm(train_loader):
        x_d = data['d_img_org'].cuda()
        labels = data['score']
        cls = data['class']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        cls = torch.squeeze(cls.type(torch.LongTensor)).cuda()
        pred_d = net(x_d)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        cls_batch_numpy = cls.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        cls_epoch = np.append(cls_epoch, cls_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    print(cls_epoch.shape)
    pred_epoch, labels_epoch, cls_epoch = np.squeeze(pred_epoch), np.squeeze(labels_epoch), np.squeeze(cls_epoch)
    for idx in range(23200):
        for i in range(7):
            if cls_epoch[idx] == i:
                clshash[str(i)] += abs(pred_epoch[idx]-labels_epoch[idx])
                preds[i].append(pred_epoch[idx])
                labs[i].append(labels_epoch[idx])
    for key in clshash:
        for key2 in clsnum:
            if key == key2:
                clshash[key] /= clsnum[key2]
    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    logging.info('CLS LOSS: train epoch:{} / 00:{:.4} / 01:{:.4} / 02:{:.4} / 03:{:.4} /04:{:.4} /05:{:.4} / 06:{:.4}'.format(epoch + 1, clshash['0'], clshash['1'], clshash['2'], clshash['3'], clshash['4'], clshash['5'], clshash['6']))
    for i in range(7):
        rho_s, _ = spearmanr(np.array(preds[i]), np.array(labs[i]))
        rho_p, _ = pearsonr(np.array(preds[i]), np.array(labs[i]))
        logging.info('train epoch:{} CLS:{}, SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, i, rho_s, rho_p))
    
    
    return ret_loss, rho_s, rho_p


""" validation """
def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d)

            pred /= config.num_avg_val
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_s, rho_p


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
        "train_ref_path": "/mnt/data_16TB/wth22/IQA_dataset/PIPAL/Train_Ref/",
        "train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/PIPAL/Train_Distort/",       
        "val_ref_path": "/mnt/data_16TB/wth22/IQA_dataset/PIPAL/Val_Ref/",
        "val_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/PIPAL/Val_Distort/",
        "train_txt_file_name": "./data/PIPAL_train.txt",
        "val_txt_file_name": "./data/PIPAL_val.txt",

        # optimization
        "batch_size": 2,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 100,
        "val_freq": 1,
        "T_max": 100,                        # cosine learning rate period (iteration)
        "eta_min": 0,                        # mininum learning rate
        "momentum": 0.9,
        "num_avg_val": 5,
        "crop_size": 224,

        # device
        "num_workers": 8,

        # model
        "patch_size": 8,
        "img_size": 224,
        "in_channels": 3,
        "embed_dim": 768,
        "dim_mlp_head": 768,
        "num_outputs": 1,
        
        # load & save checkpoint
        "model_name": "modelname",
        "snap_path": "./output/models/",               # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard/"
    })

    config.snap_path += config.model_name
    config.tensorboard_path += config.model_name
    config.log_file = config.model_name + config.log_file

    set_logging(config)
    logging.info(config)
    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    writer = SummaryWriter(config.tensorboard_path)

    # data load
    train_dataset = PIPAL(
        ref_path=config.train_ref_path,
        dis_path=config.train_dis_path,
        txt_file_name=config.train_txt_file_name,
        transform=transforms.Compose(
            [
                RandCrop(config.crop_size, 1),
                Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
                # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                RandHorizontalFlip(),
                ToTensor()
            ]
        ),
    )
    val_dataset = PIPAL(
        ref_path=config.val_ref_path,
        dis_path=config.val_dis_path,
        txt_file_name=config.val_txt_file_name,
        transform=transforms.Compose([Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]), ToTensor()]),
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )

    net = IQA()
    net = nn.DataParallel(net)
    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(config, epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

            if rho_s > best_srocc or rho_p > best_plcc:
                best_srocc = rho_s
                best_plcc = rho_p
                # save weights
                model_name = "epoch{}".format(epoch + 1)
                model_save_path = os.path.join(config.snap_path, model_name)
                torch.save({
                    'epoch': epoch,
                    'model': net,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss
                }, model_save_path)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))