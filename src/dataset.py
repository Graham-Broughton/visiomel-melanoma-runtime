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

# parser = argparse.ArgumentParser()
# parser.add_argument("--train_labels")
# parser.add_argument("--model_id")
# args = parser.parse_args()


TRAIN_LABELS = '../data/train_labels.csv'
DIR_WORKSPACE = './workspace/'
DIR_MODEL = f'{DIR_WORKSPACE}/models/'  # models output directory
os.makedirs(DIR_MODEL, exist_ok=True)

NUM_CLASSES = 1
nfolds = 5


class config:
    seed = 42
    fold = 0
    N_MAX = 128
    N = 72
    sz = 384
    bs = 8


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_dftrain(dir_train):
    df_labels = pd.read_csv(TRAIN_LABELS)
    df = pd.DataFrame()
    df_labels = df_labels.sort_values(by='filename').reset_index(drop=True)
    df_labels['tissue_id'] = df_labels.filename.str.split('.').str[0].values

    df = df_labels.set_index('tissue_id')
    files = sorted({p[:p.rindex('_')] for p in os.listdir(dir_train)})

    df = df.loc[files].reset_index().sort_values(
        by=['tissue_id']).reset_index(drop=True)
    splits = StratifiedKFold(
        n_splits=nfolds, random_state=config.seed, shuffle=True)
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
    df_train.loc[df_train.split == config.fold, 'is_valid'] = 1
    return df_train


# imagenet_stats
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean = torch.tensor(mean)
std = torch.tensor(std)


class TissueTiles(tuple):
    @classmethod
    def create(cls, fns): return cls(tuple(PILImage.create(f) for f in fns))


def TissueTilesBlock():
    return TransformBlock(type_tfms=TissueTiles.create, batch_tfms=None)


class TissueGetItems():
    def __init__(self, path_col, tissue_id_col, label_col, max_nth_tile, shuffle=False):
        self.path = path_col
        self.tissue_id = tissue_id_col
        self.label = label_col  # tissue label
        self.shuffle = shuffle
        self.max_nth_tile = max_nth_tile

    def __call__(self, df):
        data = []
        for tissue_id in progress_bar(df[self.tissue_id].unique()):
            tiles = df[(df[self.tissue_id] == tissue_id) &
                       (df['tile_id'] < self.max_nth_tile)]
            fns = tiles[self.path].tolist()

            lbl = tiles[self.label].values[0]
            data.append([*fns, lbl])
        return data


def create_batch(data):
    xs, ys = [], []
    for d in data:
        img = d[0]
        n_available = len(img)
        ixs = []
        choice_range = min(n_available, config.N_MAX)
        if n_available < config.N:
            ixs = list(np.random.choice(
                range(choice_range), size=config.N, replace=True))
        else:
            ixs = list(np.random.choice(
                range(choice_range), size=config.N, replace=False))
        ixs = sorted(list(ixs))
        # div and normalize
        img = tuple([((img[i]/255.) - mean[..., None, None]) /
                    std[..., None, None] for i in ixs])
        xs.append(img)
        ys.append(d[1])

    xs = torch.cat([TensorImage(torch.cat([im[None]
                   for im in x], dim=0))[None] for x in xs], dim=0)
    ys = torch.cat([y[None] for y in ys], dim=0)
    return TensorImage(xs), TensorCategory(ys)


def show_tile_batch(dls, max_rows=4, max_cols=5):
    xb, yb = dls.one_batch()
    fig, axes = plt.subplots(
        ncols=max_cols, nrows=max_rows, figsize=(12, 6), dpi=120)
    for i in range(max_rows):
        xs, ys = xb[i], yb[i]
        for j, x in enumerate(xs):
            if j == max_cols:
                break
            x = x.cpu()
            x = mean[..., None, None] + x * std[..., None, None]
            ax = axes[i, j] # type: ignore
            ax.imshow(x.permute(1, 2, 0).numpy())
            ax.set_title(ys.item())
            ax.axis('off')


class SequenceTfms(Transform):
    def __init__(self, tfms):
        self.tfms = tfms

    def encodes(self, x: TensorImage):

        bs, seq_len, ch, rs, cs = x.shape
        x = x.view(bs, seq_len*ch, rs, cs)
        # print(f'x={x.shape}')
        x = compose_tfms(x, self.tfms)
        x = x.view(bs, seq_len, ch, rs, cs)
        return x


class BatchTfms(Transform):
    def __init__(self, tfms):
        self.tfms = tfms

    def encodes(self, x: TensorImage):
        bs, seq_len, ch, rs, cs = x.shape
        x = x.view(bs*seq_len, ch, rs, cs)
        x = compose_tfms(x, self.tfms)

        x = x.view(bs, seq_len, ch, rs, cs)
        return x


def get_dataloader(df_train):
    def splitter(lst):
        df = pd.DataFrame([l[0] for l in lst]).rename(columns={0: 'path'})
        df = pd.merge(df, df_train).drop_duplicates(
            subset=['tissue_id']).sort_values(by=['tissue_id']).reset_index(drop=True)
        train = df[df.is_valid != 1].index.to_list()
        valid = df[df.is_valid == 1].index.to_list()
        return train, valid

    fix_seed(config.seed)
    affine_tfms, light_tfms = aug_transforms(flip_vert=False, max_rotate=25.0, max_zoom=1.1, max_warp=0.,
                                             p_affine=0.75, p_lighting=0., xtra_tfms=None)

    def get_x(t):
        return t[:-1]

    def get_y(t):
        return t[-1]
    dblock = DataBlock(
        blocks=(TissueTilesBlock, CategoryBlock),
        get_items=TissueGetItems('path', 'tissue_id', 'relapse', config.N_MAX),
        get_x=get_x,
        get_y=get_y,
        splitter=splitter,
        item_tfms=Resize(config.sz),
        batch_tfms=[SequenceTfms([affine_tfms])]
    )

    dls = dblock.dataloaders(df_train, bs=config.bs, create_batch=create_batch)
    return dls
