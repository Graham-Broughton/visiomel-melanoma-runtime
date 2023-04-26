import os
import argparse
import numpy as np
import pickle
import loguru
import random
import wandb

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data_utils
from sklearn.metrics import accuracy_score
import dotenv
from tqdm.auto import tqdm

import dataloading
from model import Attention
from utils import get_N_MAX

import sys
sys.path.append('/home/broug/Desktop/visiomel-melanoma-runtime/')
from config import CFG, NET

dotenv.load_dotenv()


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(model, device, train_loader, valid_loader, optimizer, epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    predictions = []
    labels = []
    print('epoch = ', epoch)
    for batch_idx, (data, meta, label) in tqdm(enumerate(train_loader)):
        bag_label = label
        data = torch.squeeze(data)
        data, bag_label, meta = data.to(device), bag_label.to(device), meta.to(device)

        loss, error, Y_hat, attention_weights = model.calculate_all(data, meta, bag_label)
        wandb.log({'train loss': loss.item()})

        train_loss += loss.data[0]
        loss /= NET.ACC_STEPS
        train_error += error
        # Reset gradients

        # Keep track of predictions and labels to calculate accuracy after each epoch
        predictions.append(int(Y_hat))
        labels.append(int(bag_label))
        # Backward pass
        loss.backward()
        # Update model weights

        if ((batch_idx + 1) % NET.ACC_STEPS) or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % 25 == 0:
            print('batch_idx = ', batch_idx)

    # Calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_loss = train_loss.cpu().numpy()

    train_error /= len(train_loader)
    train_error = train_error.cpu().numpy()
    valid_loss, valid_error, vacc = test(model, device, valid_loader)

    tacc = accuracy_score(labels, predictions) * 100

    print(
        f'Train Set, Epoch: {epoch}, Loss: {train_loss[0]:.4f}, Error: {train_error:.4f}, Accuracy: {tacc:.2f}'
    )
    return train_loss[0], train_error, tacc, valid_loss, valid_error, vacc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    predictions = []
    labels = []
    with torch.no_grad():
        for _, (data, meta, label) in enumerate(test_loader):
            bag_label = label
            data = torch.squeeze(data)

            #data, bag_label, meta = Variable(data), Variable(bag_label), Variable(meta)
            data, bag_label, meta = data.to(device), bag_label.to(device), meta.to(device)

            loss, error, Y_hat, attention_weights = model.calculate_all(data, meta, bag_label)
            test_loss += loss.data[0].item()
            test_error += error.item()

            predictions.append(int(Y_hat))
            labels.append(int(bag_label))

            wandb.log({"val loss": loss})

        test_error /= len(test_loader)
        test_loss /= len(test_loader)
        acc = accuracy_score(labels, predictions) * 100

    print(
        f'\nValid Set, Loss: {test_loss:.4f}, Error: {test_error:.4f}, Accuracy: {acc:.2f}'
    )
    return test_loss, test_error, acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Histopathology MIL')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='For displaying SMDataParallel-specific logs')
    parser.add_argument('--data-path', type=str, default='workspace/tiles/data/processed/')
    parser.add_argument('--start-pixels', type=int, default=48)

    # Model checkpoint location
    parser.add_argument('--model-dir', type=str, default='workspace/models/')
    parser.add_argument('--page', type=int, default=3)
    parser.add_argument('--nfolds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)

    args = parser.parse_args()
    # args.world_size = dist.get_world_size()
    # args.rank = rank = dist.get_rank()
    # args.local_rank = local_rank = dist.get_local_rank()
    # args.batch_size //= args.world_size // 8
    # args.batch_size = max(args.batch_size, 1)
    model_name = f'{args.start_pixels}-{args.page}-{args.fold}'
    data_path = f'{args.data_path}{args.start_pixels}/{args.page}/'

    # if args.verbose:
    #     print('Hello from rank', rank, 'of local_rank',
    #             local_rank, 'in world size of', args.world_size)

    if not torch.cuda.is_available():
        raise Exception("Must run SMDataParallel on CUDA-capable devices.")

    args.N_MAX, args.sz, args.N = get_N_MAX(args.start_pixels, args.page)

    fix_seed(args.seed)
    wandb.login()
    wandb.init(project="visiomel", job_type=model_name, group=f'page: {args.page}, batch_acc')
    wandb.config = dict(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        sp=args.start_pixels
    )

    train_ds = dataloading.Dataset(data_path, args.nfolds, args.seed, args.fold, args.N_MAX, args.N, args.sz, isval=False)
    meta_shape = train_ds[0][1].shape[1]
    val_ds = dataloading.Dataset(data_path, args.nfolds, args.seed, args.fold, args.N_MAX, args.N, args.sz)

    train_dl = data_utils.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = data_utils.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda")
    model = Attention(input_D=args.sz, meta_shape=meta_shape).to(device)
    # torch.cuda.set_device(local_rank)
    # model.cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)

    print('Start Training')
    wandb.watch(model, log="all")
    history = {}
    for epoch in range(1, args.epochs):
        train_loss, train_error, tacc, valid_loss, valid_error, vacc = \
            train(model, device, train_dl, val_dl, optimizer, epoch)
        history[epoch] = {
            'train_loss': train_loss,
            'train_error': train_error,
            'tacc': tacc,
            'valid_loss': valid_loss,
            'valid_error': valid_error,
            'vacc': vacc
        }
    wandb.finish()

    print("Saving the model...")
    torch.save(model.state_dict(), f'{args.model_dir}{model_name}.pt')
    pickle.dump(history, open(f'{args.model_dir}{model_name}_history.pkl', 'wb'))


if __name__ == '__main__':
    main()
