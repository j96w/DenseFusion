import os
import copy
import random
import argparse
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

from data_controller import SegDataset
from loss import Loss
from segnet import SegNet as segnet
import sys
sys.path.append("..")
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/home/data1/jeremy/YCB_Video_Dataset', help="dataset root dir (''YCB_Video Dataset'')")
parser.add_argument('--batch_size', default=3, help="batch size")
parser.add_argument('--n_epochs', default=600, help="epochs to train")
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help="learning rate")
parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--model_save_path', default='trained_models/', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--resume_model', default='', help="resume model name")
opt = parser.parse_args()

if __name__ == '__main__':
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = SegDataset(opt.dataset_root, '../datasets/ycb/dataset_config/train_data_list.txt', True, 5000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))
    test_dataset = SegDataset(opt.dataset_root, '../datasets/ycb/dataset_config/test_data_list.txt', False, 1000)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))

    model = segnet()
    model = model.cuda()

    if opt.resume_model != '':
        checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
        model.load_state_dict(checkpoint)
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = Loss()
    best_val_cost = np.Inf
    st_time = time.time()

    for epoch in range(1, opt.n_epochs):
        model.train()
        train_all_cost = 0.0
        train_time = 0
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))

        for i, data in enumerate(dataloader, 0):
            rgb, target = data
            rgb, target = Variable(rgb).cuda(), Variable(target).cuda()
            semantic = model(rgb)
            optimizer.zero_grad()
            semantic_loss = criterion(semantic, target)
            train_all_cost += semantic_loss.item()
            semantic_loss.backward()
            optimizer.step()
            logger.info('Train time {0} Batch {1} CEloss {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), train_time, semantic_loss.item()))
            if train_time != 0 and train_time % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'model_current.pth'))
            train_time += 1

        train_all_cost = train_all_cost / train_time
        logger.info('Train Finish Avg CEloss: {0}'.format(train_all_cost))
        
        model.eval()
        test_all_cost = 0.0
        test_time = 0
        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        for j, data in enumerate(test_dataloader, 0):
            rgb, target = data
            rgb, target = Variable(rgb).cuda(), Variable(target).cuda()
            semantic = model(rgb)
            semantic_loss = criterion(semantic, target)
            test_all_cost += semantic_loss.item()
            test_time += 1
            logger.info('Test time {0} Batch {1} CEloss {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_time, semantic_loss.item()))

        test_all_cost = test_all_cost / test_time
        logger.info('Test Finish Avg CEloss: {0}'.format(test_all_cost))

        if test_all_cost <= best_val_cost:
            best_val_cost = test_all_cost
            torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'model_{}_{}.pth'.format(epoch, test_all_cost)))
            print('----------->BEST SAVED<-----------')
