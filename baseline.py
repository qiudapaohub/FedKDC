import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from Models import *
from arguments import args_parser
from utils import get_dataset, get_network
from local_train import LocalUpdate, DatasetSplit, test_inference

# 单个模型的训练
def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    logger = SummaryWriter('../logs')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------data preparation--------
    train_dataset, test_dataset, user_groups = get_dataset(args)  # user_groups的最后一类作为proxy_data
    train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[1]),
                                batch_size=args.batch_size, shuffle=True)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # --------model preparation--------
    model = get_network(args.model6, args).to(device)
    model.train()

    # --------criterion preparation--------
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # Set optimizer for the local updates
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.5, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    # start train
    epoch_loss = []
    acc_max = 0
    for epoch in range(args.epochs):
        # if epoch <= 30:
        #     args.lr = 0.0003
        # elif epoch <= 60:
        #     args.lr = 0.00015
        # elif epoch <= 90:
        #     args.lr = 0.0001
        # else:
        #     args.lr = 0.00005

        model.train()
        batch_loss = []
        correct_train = 0
        total_num_train = 0
        # Set optimizer for the local updates
        # if epoch == 0:
        #     optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay = 4e-5, momentum = 0.9)
        # elif epoch == 70:
        #     optimizer = optim.SGD(model.parameters(), lr=args.lr/10, weight_decay = 4e-5, momentum = 0.9)
        # elif epoch == 140:
        #     optimizer = optim.SGD(model.parameters(), lr=args.lr/100, weight_decay = 4e-5, momentum = 0.9)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()

            log_probs, soft_output, output = model(images, t=args.temp)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            _, pred_labels = torch.max(log_probs, dim=1)
            pred_labels = pred_labels.view(-1)
            correct_train += torch.sum(torch.eq(pred_labels, labels)).item()
            total_num_train += len(labels)
        acc_train = correct_train / total_num_train

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        scheduler.step()  # 更新学习率

        # test

        model.eval()
        total_num = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()

            log_probs, _, _ = model(images, t=args.temp)
            _, pred_labels = torch.max(log_probs, dim=1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total_num += len(labels)
        acc = correct / total_num
        if acc > acc_max:
            acc_max = acc

        print('| Epoch : {} | Epoch loss : {} | acc_train: {}| acc_test: {} |max_acc: {}'.format(epoch, epoch_loss[epoch], acc_train, acc, acc_max))


if __name__ == '__main__':
    main()
