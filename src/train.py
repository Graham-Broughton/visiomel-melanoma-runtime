import os
import argparse
import time
import numpy as np
import pandas as pd
import pickle
import loguru
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data_utils
import PIL
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset import get_dftrain, get_dataloader
from model import Attention, Meta


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
    for batch_idx, (data, label) in enumerate(train_loader):
        if batch_idx % 25 == 0:
            print('batch_idx = ', batch_idx)
        bag_label = label
        data = torch.squeeze(data)
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        data, bag_label = data.to(device), bag_label.to(device)

        # Reset gradients
        optimizer.zero_grad()
        # Calculate loss
        loss, error, Y_hat, attention_weights = model.calculate_all(data, bag_label)
        train_loss += loss.data[0]
        train_error += error

        # Keep track of predictions and labels to calculate accuracy after each epoch
        predictions.append(int(Y_hat))
        labels.append(int(bag_label))
        # Backward pass
        loss.backward()
        # Update model weights
        optimizer.step()

    # Calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    valid_loss, valid_error, vacc = test(model, device, valid_loader)

    tacc = accuracy_score(labels, predictions) * 100

    print(
        f'Train Set, Epoch: {epoch}, Loss: {train_loss.cpu().numpy()[0]:.4f}, Error: {train_error:.4f}, Accuracy: {tacc:.2f}'
    )
    return train_loss.cpu().numpy()[0], train_error.numpy(), tacc, valid_loss.cpu().numpy()[0], valid_error.numpy(), vacc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label
            data = torch.squeeze(data)

            data, bag_label = Variable(data), Variable(bag_label)
            data, bag_label = data.to(device), bag_label.to(device)

            loss, error, Y_hat, attention_weights = model.calculate_all(data, bag_label)
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
    # Training settings
    parser = argparse.ArgumentParser(description='Histopathology MIL')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
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

    parser.add_argument('--image-size', type=int, default=384)

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

    fix_seed(args.seed)

    dataframe = get_dftrain(data_path, args.nfolds, args.seed, args.fold)

    loader = get_dataloader(dataframe)
    train_dataloader = loader.train
    valid_dataloader = loader.valid

    device = torch.device("cuda")
    model = Attention(input_D=args.image_size).to(device)
    # torch.cuda.set_device(local_rank)
    # model.cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)

    print('Start Training')
    history = {}
    for epoch in range(1, args.epochs):
        train_loss, train_error, tacc, valid_loss, valid_error, vacc = \
            train(model, device, train_dataloader, valid_dataloader, optimizer, epoch)
        history[epoch] = {
            'train_loss': train_loss,
            'train_error': train_error,
            'tacc': tacc,
            'valid_loss': valid_loss,
            'valid_error': valid_error,
            'vacc': vacc
        }

    print("Saving the model...")
    torch.save(model.state_dict(), f'{args.model_dir}{model_name}.pt')
    pickle.dump(history, open(f'{args.model_dir}{model_name}_history.pkl', 'wb'))


if __name__ == '__main__':
    main()
