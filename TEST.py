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




# 自定义数据集类
class EEGDataset(Dataset):
    def __init__(self, h5_file, indices):
        self.h5_file = h5_file
        self.indices = indices
        self.file = h5py.File(self.h5_file, 'r')  # 在初始化时打开文件

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        eeg_data = self.file['eeg_data'][index].astype(np.float32)
        label = self.file['labels'][index]

        # 转换label为int
        if isinstance(label, np.ndarray):
            label = label.item()
        label = int(label)

        return torch.tensor(eeg_data), torch.tensor(label, dtype=torch.long)

    def __del__(self):
        self.file.close()  # 在析构函数中关闭文件


def write_acc(round, acc, i):
    # target_file_path = 'D:/python object/federate learning/FedVC_update/result_FedAvg/CIFAR10/K{}/acc.txt'.format(args.choose_classes)
    target_file_path = 'D:/python object/federate learning/FedVC_eeg/result_test/EEG_cnn_{}'.format(i)
    if not os.path.exists(target_file_path):
        os.makedirs(target_file_path)
    with open(os.path.join(target_file_path, 'acc.txt'), 'a') as target_file:
        target_file.write("|---- Current Epoch: {}   Test Accuracy: {} \n".format((round + 1), acc))

# ---------------------------------------------------SEED------------------------------------------------
def main():
    # 文件路径
    input_file_path = 'D:/python object/federate learning\FedVC_eeg\test1_14.h5'

    # 读取HDF5文件
    with h5py.File(input_file_path, 'r') as f:
        eeg_data = f['eeg_data'][:]
        labels = f['labels'][:]

    # 生成数据索引并随机打乱
    indices = np.arange(eeg_data.shape[0])
    np.random.shuffle(indices)

    # 选取10%的数据作为训练集，3%的数据作为测试集
    train_size = int(0.09 * len(indices))
    test_size = int(0.09 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]

    # 选取训练集和测试集
    # train_eeg = eeg_data[train_indices]
    train_eeg = eeg_data[train_indices].reshape(-1, 1, 62, 400)  # 10152
    train_labels = labels[train_indices]
    # test_eeg = eeg_data[test_indices]
    test_eeg = eeg_data[test_indices].reshape(-1, 1, 62, 400)   # 4060
    test_labels = labels[test_indices]

    # 转换为PyTorch张量
    train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_eeg_tensor = torch.tensor(test_eeg, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    # 创建TensorDataset
    train_dataset = TensorDataset(train_eeg_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_eeg_tensor, test_labels_tensor)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    args = args_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    # -----------------------------------------------------------------------------
    for i in range(3):
        # --------------model pre------------------
        if i == 0:
            model = get_network(args.model8, args).to(device)
        elif i == 1:
            model = get_network(args.model9, args).to(device)
        else:
            model = get_network(args.model10, args).to(device)

        model.train()
        # ---------------criterion-------------------
        criterion = nn.CrossEntropyLoss().to(device)
        # ------------------optimizer---------------
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        epoch_loss = []
        acc_max = 0
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

            # test
            model.eval()
            total_num = 0
            correct = 0
            with torch.no_grad():
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

            print('| Epoch : {} | Epoch loss : {} | acc_train: {}| acc_test: {} |max_acc: {}'.format(epoch, epoch_loss[epoch],
                                                                                                     acc_train, acc, acc_max))
            write_acc(epoch, acc, i)

if __name__ == '__main__':
    main()

# 示例: 显示客户端 0 的 DataLoader 中的数据形状
# for data, labels in clients_dataloaders[0]:
#     print(f"Data shape: {data.shape}")
#     print(f"Labels shape: {labels.shape}")
#     break  # 只显示一个批次的数据
# for data, labels in proxy_dataloader:
#     print(f"Data shape: {data.shape}")
#     print(f"Labels shape: {labels.shape}")
#     break  # 只显示一个批次的数据

