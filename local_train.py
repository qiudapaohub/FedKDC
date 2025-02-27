#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idx, logger, data_dom):
        self.args = args
        self.logger = logger
        self.trainloader, self.preloader = self.train_val_test(dataset, idx)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.data_dom = data_dom

    def train_val_test(self, dataset, idx):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        trainloader = DataLoader(dataset[idx], batch_size=self.args.batch_size, shuffle=True)
        preloader = DataLoader(dataset[idx], batch_size=self.args.batch_size, shuffle=True)

        return trainloader, preloader

    def update_weights(self, model, num_clint, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        lr = self.args.lr[num_clint]
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.5, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.loc_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.data_dom == '2d':
                    images = images.reshape(images.size(0), -1)

                model.zero_grad()
                log_probs, soft_output, output = model(images, t=1)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print(
                        '| Global round: {} | Clint Num : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, num_clint, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                           100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_prox(self, model, global_model, num_clint, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        mu = 0.01

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr[num_clint],
                                        momentum=0.5, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr[num_clint],
                                         weight_decay=1e-4)

        for iter in range(self.args.loc_epochs):
            batch_loss = []
            # set_label = set()
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # for i in labels:
                #     set_label.add(i.item())  #打印每个client内的类别数量

                model.zero_grad()
                if self.data_dom == '2d':
                    images = images.reshape(images.size(0), -1)
                else:
                    images = images.reshape(images.size(0), 1, -1)

                log_probs, soft_output, output = model(images, t=1)

                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss = self.criterion(soft_output, labels) + (mu / 2) * proximal_term
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print(
                        '| Global round: {} | Clint Num : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, num_clint, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                           100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print(set_label)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def prediction(self, model):
        model.eval()
        soft_pre = []
        with torch.no_grad():  # 在评估模式下，不需要梯度计算
            for batch_idx, (images, labels) in enumerate(self.preloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.data_dom == '2d':
                    images = images.reshape(images.size(0), -1)

                # Inference
                _, batch_soft_pre, _ = model(images, t=4.0)
                soft_pre.append(batch_soft_pre)

        # 将 soft_pre 中的每个元素连接成一个张量
        pre_clint = torch.cat(soft_pre, dim=0)
        return pre_clint

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


def test_inference(args, model, test_dataset, device, data_dom):
    """ Returns the test accuracy, loss, and confusion matrix.
    """
    model.eval()
    total, correct = 0.0, 0.0
    batch_loss_list = []

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    outputs = []
    labels_list = []  # Store true labels for confusion matrix
    preds_list = []   # Store predicted labels for confusion matrix

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            if data_dom == '2d':
                images = images.reshape(images.size(0), -1)

            # Inference
            log_outputs, _, output = model(images, t=args.temp)
            batch_loss = criterion(output, labels)
            batch_loss_list.append(batch_loss.item())

            # Prediction
            _, pred_labels = torch.max(log_outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            # Collect predictions and true labels
            outputs.append(output)
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(pred_labels.cpu().numpy())

    accuracy = correct / total
    outputs = torch.cat(outputs, dim=0)
    loss = sum(batch_loss_list) / len(batch_loss_list)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels_list, preds_list)

    return accuracy, outputs, loss, conf_matrix



def prox_inference(args, model, testloader, device, clint_num, epoch, data_dom):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    clients_pred = []

    criterion = nn.NLLLoss().to(device)

    outputs = []
    label = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            if data_dom[clint_num] == '2d':
                images = images.reshape(images.size(0), -1)

            # Inference
            log_outputs, _, output = model(images, t=args.temp)
            batch_loss = criterion(log_outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(log_outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            outputs.append(output)
            label.append(labels)
            clients_pred.append(F.one_hot(pred_labels, num_classes=args.num_classes))

    accuracy = correct/total
    outputs = torch.cat(outputs, dim=0)
    return accuracy, outputs, clients_pred
