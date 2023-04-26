import pandas as pd
import numpy as np
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from loguru import logger

from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATA = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) / 'data'
DIR_WORKSPACE = './workspace/'
# DIR_MODEL = f'{DIR_WORKSPACE}/models/'  # models output directory
# os.makedirs(DIR_MODEL, exist_ok=True)


def process_df(df: pd.DataFrame):
    df['age'] = df['age'].str.slice(1, 3).astype(np.float32)
    df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
    df['sex'] = df['sex'].replace(1, 0).replace(2, 1).astype(np.float32)
    df['melanoma_history'] = df['melanoma_history'].replace({'YES': 1, "NO": 0}).fillna(-1).astype(np.float32)
    return df


class EvalDataset(Dataset):
    def __init__(self, dir_test, dir_meta, seed, N_MAX, N, sz):
        self.dir_test = dir_test
        self.seed = seed
        self.N_MAX = N_MAX
        self.N = N
        self.sz = sz
        dftest = self.get_dftest(dir_test, dir_meta)
        df1 = dftest.groupby('tissue_id').agg({'tile_id': list, 'path': list})
        df2 = dftest.drop(['path', 'tile_id'], axis=1).groupby('tissue_id').agg(lambda x: x.head(1))
        dftest = pd.merge(df1, df2, on='tissue_id').reset_index().drop('body_site', axis=1)
        dftest = pd.concat([dftest, pd.get_dummies(dftest['body_site']).astype(np.float32)], axis=1)
        self.dftest = dftest.sample(frac=1)

    def __len__(self):
        return len(self.dftest)

    def __getitem__(self, index):
        item = self.dftest.iloc[index:index+1, :]

        meta = item[[
            'sex', 'age', 'melanoma_history', 'thigh', 'trunc', 'face', 'forearm', 'arm', 'leg', 'hand',
            'foot', 'sole', 'finger', 'neck', 'toe', 'seat', 'scalp', 'nail', 'trunk', 'lower limb/hip',
            'hand/foot/nail', 'head/neck', 'upper limb/shoulder'
        ]].values
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

        return images, metadata, data['tissue_id']

    @staticmethod
    def get_dftest(dir_test, meta_dir):
        test = pd.read_csv(meta_dir / 'test_metadata.csv')
        test = process_df(test)

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

        #df_test = df_test.merge(test, on='filename')
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
        self.N_MAX = N_MAX
        self.N = N
        self.sz = sz
        dftrain = self.get_dftrain(dir_train, nfolds, seed, fold)
        df1 = dftrain.groupby('tissue_id').agg({'tile_id': list, 'path': list})
        df2 = dftrain.drop(['path', 'tile_id'], axis=1).groupby('tissue_id').agg(lambda x: x.head(1))
        dftrain = pd.merge(df1, df2, on='tissue_id').reset_index()
        dftrain = pd.concat([dftrain, pd.get_dummies(dftrain['body_site']).astype(np.float32)], axis=1).drop('body_site', axis=1)
        if isval:
            dftrain = dftrain[dftrain['is_valid'] == 1]
        else:
            dftrain = dftrain[dftrain['is_valid'] == 0]
        self.dftrain = dftrain.sample(frac=1)

    def __len__(self):
        return len(self.dftrain)

    def __getitem__(self, index):
        item = self.dftrain.iloc[index:index+1, :]

        label = item['relapse'].values
        meta = item[[
            'sex', 'age', 'melanoma_history', 'thigh', 'trunc', 'face', 'forearm', 'arm', 'leg', 'hand',
            'foot', 'sole', 'finger', 'neck', 'toe', 'seat', 'scalp', 'nail', 'trunk', 'lower limb/hip',
            'hand/foot/nail', 'head/neck', 'upper limb/shoulder'
        ]].values
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
        labels = torch.from_numpy(label)

        return images, metadata, labels

    @staticmethod
    def get_dftrain(dir_train, nfolds, seed, fold):
        df_labels = pd.read_csv(DATA / 'train_labels.csv')
        train = pd.read_csv(DATA / 'train_metadata.csv')
        train = process_df(train)

        df = pd.DataFrame()
        df_labels = df_labels.sort_values(by='filename').reset_index(drop=True)
        df_labels['tissue_id'] = df_labels.filename.str.split('.').str[0].values

        df = df_labels.set_index('tissue_id')
        files = sorted({p[:p.rindex('_')] for p in os.listdir(dir_train)})

        df = df.loc[files].reset_index().sort_values(
            by=['tissue_id']).reset_index(drop=True)

        splits = StratifiedKFold(
            n_splits=nfolds, random_state=seed, shuffle=True)
        splits = list(splits.split(df, df['relapse']))

        folds_splits = np.zeros(len(df)).astype(int)
        for i in range(nfolds):
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
        df_train.loc[df_train.split == fold, 'is_valid'] = 1

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
