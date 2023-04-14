import pandas as pd
import numpy as np
import os
import random
import shutil
import time
import argparse
import PIL
import cv2
from fastai.vision.all import *
from fastai.callback.mixup import CutMix

from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from types import SimpleNamespace

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self,MODEL,NORM,pre=True):
        super().__init__()
        if MODEL == 'ResX50':
          ##resnext model
          m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
          print('loaded ResNext model')
        elif MODEL == 'ResS50':
          ##resnest model
          m = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pre)
          print('loaded ResNest model')
        blocks = [*m.children()]
        enc = blocks[:-2]
        self.enc = nn.Sequential(*enc)
        C = blocks[-1].in_features
        head = [AdaptiveConcatPool2d(),
                Flatten(), #bs x 2*C
                nn.Linear(2*C,512),
                Mish()
                ]
        if NORM == 'GN':
          head.append(nn.GroupNorm(32,512))
          print('Group Norm')
        elif NORM == 'BN':
          head.append(nn.BatchNorm1d(512))
          print('Batch Norm')

        head.append(nn.Dropout(0.5))
        head.append(nn.Linear(512,NUM_CLASSES-1))
        self.head = nn.Sequential(*head)

    def forward(self, *x):
        shape = x[0].shape
        n = shape[1]## n_tiles
        x = torch.stack(x,1).view(-1,shape[2],shape[3],shape[4])
        x = self.enc(x)
        shape = x.shape
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
        x = self.head(x)
        return x


def acc(inp, targ):
    pred = torch.sigmoid(inp).sum(1).round()
    return (pred==targ).float().mean()
