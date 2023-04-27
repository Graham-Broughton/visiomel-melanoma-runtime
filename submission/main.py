from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd

import subprocess
import random
import os
from tqdm.auto import tqdm

import torch
import torch.utils.data as data_utils

from config import CFG, NET
import src.dataloading
from src.model import Attention
from src.utils import get_N_MAX


# ROOT = Path("./")
# DATA_ROOT = Path("data/")
# MODEL_ROOT = Path("assets")
ROOT = Path("/code_execution")
DATA_ROOT = Path("/code_execution/data/")
MODEL_ROOT = Path("/code_execution/assets")
S_PIXELS = [48, 64]
PAGES = [0, 1, 2, 3]
FOLDS = list(range(5))


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_model(starting_pixels, page, fold, size, meta_shape):
    logger.info(f"Loading model: {starting_pixels}-{page}-{fold}")
    path = f'{MODEL_ROOT}/{starting_pixels}-{page}-{fold}.pt'
    state = torch.load(path)
    model = Attention(input_D=size, meta_shape=meta_shape)
    model.load_state_dict(state)
    return model


def test(model, device, test_loader):
    model.eval()
    predictions = []
    tissue_ids = []
    logger.info("Starting Inference")
    with torch.no_grad():
        for (data, meta, tissue_id) in tqdm(test_loader):
            data = torch.squeeze(data)

            data, meta = data.to(device), meta.to(device)

            Y_prob, Y_hat, A = model(data, meta)

            predictions.append(Y_prob)
            tissue_ids.append(tissue_id)

    logger.info("Finished Inference")
    return predictions, tissue_ids


def main():
    # load sumission format
    submission_format = pd.read_csv(DATA_ROOT / "submission_format.csv", index_col=0)

    logger.info("Starting Preprocessing")
    out = subprocess.run(
        ['python', 'src/preprocess.py', '--dir_input_tif', f'{DATA_ROOT}', '--file_meta', f'{DATA_ROOT}/test_metadata.csv', '--dir_output', 'processed'],
        capture_output=True, text=True)
    #logger.info(f'subprocess info: {out}')

    logger.info("Finished Preprocessing")
    device = torch.device("cuda")

    logger.info("Getting Data Ready")
    N_MAX, sz, N = get_N_MAX(int(NET.SP), int(NET.PAGE))
    dataset = src.dataloading.EvalDataset(f'{ROOT}/workspace/tiles/processed/{NET.SP}/{NET.PAGE}/', DATA_ROOT, CFG.SEED, N_MAX, N, sz)
    meta_shape = dataset[0][1].shape[1]
    loader = data_utils.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    dfs = []
    for fold in FOLDS:
        model = load_model(NET.SP, NET.PAGE, fold, sz, meta_shape).to(device)
        preds, ids = test(model, device, loader)
        dfs.append(pd.DataFrame(columns={'tissue_id': ids, "predictions": preds}))

    logger.info("Finished all folds")
    df = pd.concat(dfs, axis=0)
    df = df.groupby('tissue_id').mean()
    df.index = df.index.rename('filename') + '.tif'
    df = df.reindex(submission_format.index)
    df = df.rename({'predictions': 'relapse'}, axis=1).reset_index()
    df.to_csv("submission.csv")


if __name__ == '__main__':
    fix_seed(CFG.SEED)
    main()
