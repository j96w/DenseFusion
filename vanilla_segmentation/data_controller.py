import torch
import numpy as np
from PIL import Image
import numpy.ma as ma
import torch.utils.data as data
import copy
from torchvision import transforms
import scipy.io as scio
import torchvision.datasets as dset
import random
import scipy.misc
import scipy.io as scio
import os
from PIL import ImageEnhance
from PIL import ImageFilter

class SegDataset(data.Dataset):
    def __init__(self, root_dir, txtlist, use_noise, length):
        self.path = []
        self.real_path = []
        self.use_noise = use_noise
        self.root = root_dir
        input_file = open(txtlist)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.path.append(copy.deepcopy(input_line))
            if input_line[:5] == 'data/':
                self.real_path.append(copy.deepcopy(input_line))
        input_file.close()

        self.length = length
        self.data_len = len(self.path)
        self.back_len = len(self.real_path)

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.back_front = np.array([[1 for i in range(640)] for j in range(480)])

    def __getitem__(self, idx):
        index = random.randint(0, self.data_len - 10)

        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.path[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.path[index]))
        if not self.use_noise:
            rgb = np.array(Image.open('{0}/{1}-color.png'.format(self.root, self.path[index])).convert("RGB"))
        else:
            rgb = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, self.path[index])).convert("RGB")))

        if self.path[index][:8] == 'data_syn':
            rgb = Image.open('{0}/{1}-color.png'.format(self.root, self.path[index])).convert("RGB")
            rgb = ImageEnhance.Brightness(rgb).enhance(1.5).filter(ImageFilter.GaussianBlur(radius=0.8))
            rgb = np.array(self.trancolor(rgb))
            seed = random.randint(0, self.back_len - 10)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, self.path[seed])).convert("RGB")))
            back_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.path[seed])))
            mask = ma.getmaskarray(ma.masked_equal(label, 0))
            back = np.transpose(back, (2, 0, 1))
            rgb = np.transpose(rgb, (2, 0, 1))
            rgb = rgb + np.random.normal(loc=0.0, scale=5.0, size=rgb.shape)
            rgb = back * mask + rgb
            label = back_label * mask + label
            rgb = np.transpose(rgb, (1, 2, 0))
            #scipy.misc.imsave('embedding_final/rgb_{0}.png'.format(index), rgb)
            #scipy.misc.imsave('embedding_final/label_{0}.png'.format(index), label)
            
        if self.use_noise:
            choice = random.randint(0, 3)
            if choice == 0:
                rgb = np.fliplr(rgb)
                label = np.fliplr(label)
            elif choice == 1:
                rgb = np.flipud(rgb)
                label = np.flipud(label)
            elif choice == 2:
                rgb = np.fliplr(rgb)
                rgb = np.flipud(rgb)
                label = np.fliplr(label)
                label = np.flipud(label)
                

        obj = meta['cls_indexes'].flatten().astype(np.int32)
        obj = np.append(obj, [0], axis=0)
        target = copy.deepcopy(label)

        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = self.norm(torch.from_numpy(rgb.astype(np.float32)))
        target = torch.from_numpy(target.astype(np.int64))

        return rgb, target


    def __len__(self):
        return self.length

