import pandas as pd
import numpy as np
import os
import random
from PIL import Image
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TRAIN_LABELS = './data/train_labels.csv'
TRAIN = './data/train.csv'
DIR_WORKSPACE = './workspace/'
DIR_MODEL = f'{DIR_WORKSPACE}/models/'  # models output directory
os.makedirs(DIR_MODEL, exist_ok=True)

N_MAX = 80
N = 16
sz = 256
bs = 1


class EvalDataset(Dataset):
    def __init__(self, dir_test, seed, N_MAX, N, sz):
        self.dir_test = dir_test
        self.seed = seed
        self.N_MAX = N_MAX
        self.N = N
        self.sz = sz
        self.dftest = self.get_dftest(dir_test)
        self.data = []
        self.metas = []
        self.labels = []
        self.dftrain = self.dftrain.groupby('tissue_id').agg({
            'tile_id': list, 'path': list, 'filename': 'first', 'melanoma_history': 'mean', 'sex': 'mean', 'age': 'mean', 'n_tiles': 'mean'
        }).reset_index()
        self.dftest = self.dftest.sample(frac=1)

    def __len__(self):
        return len(self.dftest)

    def __getitem__(self, index):
        item = self.dftest.iloc[index:index+1, :]

        meta = item[['sex', 'age', 'melanoma_history']].values
        data = item[['path', 'tissue_id', 'tile_id']]
        data = data.explode('path', ignore_index=True)

        avail_imgs = len(data)
        ixs = []
        choices = min(avail_imgs, self.N_MAX)

        if choices < self.N:
            ixs = list(np.random.choice(range(choices), size=self.N, replace=True))
        else:
            ixs = list(np.random.choice(range(choices), size=self.N, replace=False))
        ixs = sorted(list(ixs))
        images = torch.stack([self.load_img(data.loc[x, 'path']).to(torch.float32) for x in ixs])
        images /= 255.
        metadata = torch.from_numpy(meta)

        return images, metadata

    def get_dftest(self, dir_test):
        test = pd.read_csv(dir_test / 'test_metadata.csv')

        df = pd.DataFrame()
        test = test.sort_values(by='filename').reset_index(drop=True)
        test['tissue_id'] = test.filename.str.split('.').str[0].values

        df = test.set_index('tissue_id')
        files = sorted({p[:p.rindex('_')] for p in os.listdir(dir_test)})

        df = df.loc[files].reset_index().sort_values(
            by=['tissue_id']).reset_index(drop=True)

        print(f'training dataset: {df.shape[0]} samples')

        fnames = [p for p in os.listdir(dir_test) if p.split('.')[1] == 'jpeg']
        df1 = pd.DataFrame(fnames).rename(columns={0: 'tissue_id'})

        df1['path'] = dir_test + df1['tissue_id']
        df1['tissue_id'] = df1.tissue_id.str.rsplit('_', n=1, expand=True)[0]
        df1['tile_id'] = df1['path'].str.split(
            '_').str[-1].str.split('.', expand=True)[0].astype(np.int16)
        print(f'testing dataset: {df1.shape[0]} tiles')

        n_tiles = []
        for ix, row in df.iterrows():
            fn = row.tissue_id
            tiles = [f for f in fnames if fn in f]
            n_tiles.append(len(tiles))
        df['n_tiles'] = n_tiles
        df = df[df.n_tiles > 0]

        df_test = pd.merge(df, df1, on='tissue_id').sort_values(
            by=['tissue_id', 'tile_id']).reset_index(drop=True)

        df_test = df_test.merge(test, on='filename')
        nums = df_test.select_dtypes(include='number').columns
        df_test[nums] = df_test[nums].astype(np.float32)
        return df_test

    def load_img(self, path):
        img = Image.open(path)
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize([self.sz, self.sz])
        if (img.width, img.height) != (self.sz, self.sz):
            img = resize(img)
        return to_tensor(img)


class Dataset(Dataset):
    def __init__(self, dir_train, nfolds, seed, fold, N_MAX, N, sz, isval=True):
        self.dir_train = dir_train
        self.nfolds = nfolds
        self.seed = seed
        self.fold = fold
        self.N_MAX = N_MAX
        self.N = N
        self.sz = sz
        self.df_labels = pd.read_csv(TRAIN_LABELS)
        self.dftrain = self.get_dftrain(dir_train)
        self.data = []
        self.metas = []
        self.labels = []
        self.dftrain = self.dftrain.groupby('tissue_id').agg({
            'tile_id': list, 'path': list, 'filename': 'first', 'melanoma_history': 'mean', 'sex': 'mean', 'age': 'mean', 'is_valid': 'mean', 'n_tiles': 'mean', 'split': 'mean', 'relapse': 'mean'
        }).reset_index()
        if isval:
            self.dftrain = self.dftrain[self.dftrain['is_valid'] == 1]
        else:
            self.dftrain = self.dftrain[self.dftrain['is_valid'] == 0]
        self.dftrain = self.dftrain.sample(frac=1)

    def __len__(self):
        return len(self.dftrain)

    def __getitem__(self, index):
        item = self.dftrain.iloc[index:index+1, :]

        label = item['relapse'].values
        meta = item[['sex', 'age', 'melanoma_history']].values
        data = item[['path', 'tissue_id', 'tile_id']]
        data = data.explode('path', ignore_index=True)

        avail_imgs = len(data)
        ixs = []
        choices = min(avail_imgs, self.N_MAX)

        if choices < self.N:
            ixs = list(np.random.choice(range(choices), size=self.N, replace=True))
        else:
            ixs = list(np.random.choice(range(choices), size=self.N, replace=False))
        ixs = sorted(list(ixs))
        images = torch.stack([self.load_img(data.loc[x, 'path']).to(torch.float32) for x in ixs])
        images /= 255.
        #images = read_image(data.loc[ixs[0], 'path'])
        metadata = torch.from_numpy(meta)
        labels = torch.from_numpy(label)

        return images, metadata, labels

    def get_dftrain(self, dir_train):
        df_labels = self.df_labels
        train = pd.read_csv(TRAIN)

        df = pd.DataFrame()
        df_labels = df_labels.sort_values(by='filename').reset_index(drop=True)
        df_labels['tissue_id'] = df_labels.filename.str.split('.').str[0].values

        df = df_labels.set_index('tissue_id')
        files = sorted({p[:p.rindex('_')] for p in os.listdir(dir_train)})

        df = df.loc[files].reset_index().sort_values(
            by=['tissue_id']).reset_index(drop=True)

        splits = StratifiedKFold(
            n_splits=self.nfolds, random_state=self.seed, shuffle=True)
        splits = list(splits.split(df, df['relapse']))

        folds_splits = np.zeros(len(df)).astype(int)
        for i in range(self.nfolds):
            folds_splits[splits[i][1]] = i

        df['split'] = folds_splits
        print(f'training dataset: {df.shape[0]} samples')

        fnames = [p for p in os.listdir(dir_train) if p.split('.')[1] == 'jpeg']
        df1 = pd.DataFrame(fnames).rename(columns={0: 'tissue_id'})

        df1['path'] = dir_train + df1['tissue_id']
        df1['tissue_id'] = df1.tissue_id.str.rsplit('_', n=1, expand=True)[0]
        df1['tile_id'] = df1['path'].str.split(
            '_').str[-1].str.split('.', expand=True)[0].astype(np.int16)
        print(f'training dataset: {df1.shape[0]} tiles')

        n_tiles = []
        for ix, row in df.iterrows():
            fn = row.tissue_id
            tiles = [f for f in fnames if fn in f]
            n_tiles.append(len(tiles))
        df['n_tiles'] = n_tiles
        df = df[df.n_tiles > 0]

        df_train = pd.merge(df, df1, on='tissue_id').sort_values(
            by=['tissue_id', 'tile_id']).reset_index(drop=True)
        df_train['is_valid'] = 0
        df_train.loc[df_train.split == self.fold, 'is_valid'] = 1

        df_train = df_train.merge(train, on='filename')
        nums = df_train.select_dtypes(include='number').columns
        df_train[nums] = df_train[nums].astype(np.float32)
        return df_train

    def load_img(self, path):
        img = Image.open(path)
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize([self.sz, self.sz])
        if (img.width, img.height) != (self.sz, self.sz):
            img = resize(img)
        return to_tensor(img)
