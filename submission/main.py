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


# ROOT = Path("./submission")
# DATA_ROOT = Path("submission/data/")
# MODEL_ROOT = Path("submission/assets")
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
    logger.info("Starting Inference")
    with torch.no_grad():
        for (data, meta) in tqdm(test_loader):
            data = torch.squeeze(data)

            data, meta = data.to(device), meta.to(device)

            Y_prob, Y_hat, A = model(data, meta)

            predictions.append(Y_prob.cpu().squeeze(0).squeeze(0).numpy())

    logger.info("Finished Inference")
    return predictions


def main():
    # load sumission format
    submission_format = pd.read_csv(DATA_ROOT / "submission_format.csv", index_col=0)

    logger.info("Starting Preprocessing")
    subprocess.run(
        ['python', 'src/preprocess.py', '--dir_input_tif', f'{DATA_ROOT}', '--file_meta', f'{DATA_ROOT}/test_metadata.csv', '--dir_output', 'processed'],
        check=True)

    logger.info("Finished Preprocessing")
    device = torch.device("cuda")

    logger.info("Getting Data Ready")
    N_MAX, sz, N = get_N_MAX(int(NET.SP), int(NET.PAGE))
    dataset = src.dataloading.EvalDataset(f'{ROOT}/workspace/tiles/processed/{NET.SP}/{NET.PAGE}/', DATA_ROOT, CFG.SEED, N_MAX, N, sz, submission_format)
    meta_shape = dataset[0][1].shape[1]
    loader = data_utils.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    dfs = []
    for fold in FOLDS:
        model = load_model(NET.SP, NET.PAGE, fold, sz, meta_shape).to(device)
        preds = test(model, device, loader)
        dfs.append(pd.DataFrame({"predictions": preds}, index=submission_format.index))

    logger.info("Finished all folds")
    df = pd.concat(dfs, axis=0).reset_index()
    df = df.groupby('filename').mean()
    df = df.reindex(submission_format.index)
    df = df.rename({'predictions': 'relapse'}, axis=1).reset_index()
    df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    fix_seed(CFG.SEED)
    main()
