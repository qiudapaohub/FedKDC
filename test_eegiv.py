import copy

import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_network
from arguments import args_parser
import torch.nn as nn
import numpy as np
import h5py
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn import preprocessing
import pandas as pd

# 修改后的 write_acc 函数
def write_acc(epoch, acc_list, target_file_path):
    # 确保保存文件夹存在
    if not os.path.exists(target_file_path):
        os.makedirs(target_file_path)

    # 定义文件路径
    file_path = os.path.join(target_file_path, 'output.csv')

    # 将 acc_list 转换为 DataFrame，每个列表项是某个 epoch 的结果
    acc_df = pd.DataFrame(acc_list)

    if not os.path.exists(file_path):
        # 如果文件不存在，写入数据并保存列名（表头）
        acc_df.to_csv(file_path, index=False, header=True)
    else:
        # 如果文件存在，按列追加新数据，不保存表头
        existing_df = pd.read_csv(file_path)
        # 追加新的列
        combined_df = pd.concat([existing_df, acc_df], axis=1)
        combined_df.to_csv(file_path, index=False, header=True)


def main():
    target_file_path = 'D:/python object/federate learning/FedVC_eeg/result_test_crocul/result_test_F_G/EEG_cnn_2'

    all_acc = []  # 存储所有循环的测试 acc，每个循环的 acc 是一个列表
    for client_num in range(1, 5):
        # 文件路径
        train_file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/FRA/FRA{}.h5'.format(client_num)
        test_file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/GER/GER1.h5'

        # 读取训练集数据
        with h5py.File(train_file_path, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        # train_eeg = preprocessing.scale(eeg_data)
        train_label = labels

        train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
        train_label_tensor = torch.tensor(train_label, dtype=torch.long)

        train_dataset = TensorDataset(train_eeg_tensor, train_label_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 读取测试集数据
        with h5py.File(test_file_path, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        test_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        # test_eeg = preprocessing.scale(eeg_data)
        test_label = labels

        test_eeg_tensor = torch.tensor(test_eeg, dtype=torch.float32)
        test_label_tensor = torch.tensor(test_label, dtype=torch.long)

        test_dataset = TensorDataset(test_eeg_tensor, test_label_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        # 模型相关部分
        args = args_parser()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        model = get_network(args.model9, args).to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0000006, momentum=0.5, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.000002, weight_decay=args.weight_decay)

        epoch_loss = []
        acc_max = 0
        acc_epoch_list = []  # 记录每个 epoch 的 acc

        for epoch in range(args.epochs):
            model.train()
            batch_loss = []
            correct_train = 0
            total_num_train = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                log_probs, _, output = model(images, t=1)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                _, pred_labels = torch.max(log_probs, dim=1)
                pred_labels = pred_labels.view(-1)
                correct_train += torch.sum(torch.eq(pred_labels, labels)).item()
                total_num_train += len(labels)
            acc_train = correct_train / total_num_train

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # 测试阶段
            model_p = copy.deepcopy(model)
            model_p.eval()
            total_num = 0
            correct = 0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(device), labels.to(device)
                    model.zero_grad()

                    log_probs, _, _ = model_p(images, t=args.temp)
                    _, pred_labels = torch.max(log_probs, dim=1)
                    pred_labels = pred_labels.view(-1)
                    correct += torch.sum(torch.eq(pred_labels, labels)).item()
                    total_num += len(labels)
                acc = correct / total_num
                acc_epoch_list.append(acc)  # 记录当前 epoch 的 acc
                if acc > acc_max:
                    acc_max = acc

            print(
                '| Epoch : {} | Epoch loss : {} | acc_train: {}| acc_test: {} |max_acc: {}'.format(epoch, epoch_loss[epoch],
                                                                                                   acc_train, acc, acc_max))

        # 将当前 client 的 acc 列表加入 all_acc
        all_acc.append(acc_epoch_list)

    # 将所有循环的 acc 写入同一个 CSV 文件，形成四列
    write_acc(range(args.epochs), all_acc, target_file_path)


if __name__ == '__main__':
    main()