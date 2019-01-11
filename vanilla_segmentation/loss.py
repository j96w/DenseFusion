from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import copy
import math

CEloss = nn.CrossEntropyLoss()

def loss_calculation(semantic, target):
    bs = semantic.size()[0]
    pix_num = 480 * 640

    target = target.view(bs, -1).view(-1).contiguous()
    semantic = semantic.view(bs, 22, pix_num).transpose(1, 2).contiguous().view(bs * pix_num, 22).contiguous()
    semantic_loss = CEloss(semantic, target)

    return semantic_loss


class Loss(_Loss):

    def __init__(self):
        super(Loss, self).__init__(True)

    def forward(self, semantic, target):
        return loss_calculation(semantic, target)