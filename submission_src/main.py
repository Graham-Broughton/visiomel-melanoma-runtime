import pickle
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd

import subprocess
import random
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from sklearn.metrics import accuracy_score

import src.dataloading
from src.model import Attention
from src.utils import get_N_MAX


DATA_ROOT = Path("/code_execution/data/")
MODEL_ROOT = Path("/code_execution/assets/")
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


def process_df(df: pd.DataFrame):
    df['age'] = df['age'].str.slice(1, 3).astype(np.float32)
    df['age'] = (df['age'] - df['age'].min()) / (df['age'] - df['age'].min()).max()
    df['sex'] = df['sex'].replace(1, 0).replace(2, 1).astype(np.float32)
    df['melanoma_history'] = df['melanoma_history'].replace({'YES': 1, "NO": 0}).fillna(-1).astype(np.float32)
    return df


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data, meta, label) in enumerate(test_loader):
            bag_label = label
            data = torch.squeeze(data)

            data, bag_label, meta = Variable(data), Variable(bag_label), Variable(meta)
            data, bag_label, meta = data.to(device), bag_label.to(device), meta.to(device)

            loss, error, Y_hat, attention_weights = model.calculate_all(data, meta, bag_label)
            test_loss += loss.data[0]
            test_error += error

            predictions.append(int(Y_hat))
            labels.append(int(bag_label))

        test_error /= len(test_loader)
        test_loss /= len(test_loader)
        acc = accuracy_score(labels, predictions)

    print(
        f'\nValid Set, Loss: {test_loss.cpu().numpy()[0]:.4f}, Error: {test_error:.4f}, Accuracy: {acc:.2f}'
    )
    return test_loss, test_error, acc


def main():
    # load sumission format
    submission_format = pd.read_csv(DATA_ROOT / "submission_format.csv", index_col=0)

    # load test_metadata
    test_metadata = pd.read_csv(DATA_ROOT / "test_metadata.csv", index_col=0)

    subprocess.run(
        ['python', 'src/preprocess.py', '--dir_input_tif', DATA_ROOT, '--file_meta', DATA_ROOT / 'test_metadata.csv', '--dir_output', 'processed'],
        check=True)
