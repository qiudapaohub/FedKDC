import sys
import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar10_iid, cifar10_noniid, seed_iid, SEED_noniid
import h5py
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
from sklearn import preprocessing

class EEGDataset(Dataset):
    def __init__(self, data_file, indices):
        self.data_file = data_file
        self.indices = indices

        with h5py.File(self.data_file, 'r') as f:
            self.eeg_data_shape = f['eeg_data'].shape
            self.labels_shape = f['labels'].shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        with h5py.File(self.data_file, 'r') as f:
            eeg_data = f['eeg_data'][index]
            label = f['labels'][index]
        eeg_data = eeg_data.reshape(1, 62, 400)
        return torch.tensor(eeg_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    global test_dataset, train_dataset, user_groups
    if args.dataset == 'cifar10':
        data_dir = '../FedVC/data/cifar10/'

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])


        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,   # (50000, 32, 32)
                                       transform=transform_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,   # (10000, 32, 32)
                                      transform=transform_test)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar10_iid(train_dataset, args.num_clint + 1)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar10_noniid(train_dataset, args.num_clint, args.choose_classes)

    elif args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../FedVC/data/mnist/'
        else:
            data_dir = '../FedVC/data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        if  args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_clint + 1)    # 最后一类作为proxy_data
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_clint)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_clint, args.choose_classes)

    elif args.dataset == 'seed':
        # 读取test HDF5文件
        test_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_FEATURE/TEST3.0/FEATUREtest15.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        test_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        test_labels = labels
        # 转为pytorch张量
        test_eeg_tensor = torch.tensor(test_eeg, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
        # 创建TensorDataset
        test_dataset = TensorDataset(test_eeg_tensor, test_labels_tensor)


        # 读取proxy HDF5文件
        proxy_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_FEATURE/TEST3.0/proxy1_5_12.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        proxy_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        proxy_labels = labels
        # 转为pytorch张量
        proxy_eeg_tensor = torch.tensor(proxy_eeg, dtype=torch.float32)
        proxy_labels_tensor = torch.tensor(proxy_labels, dtype=torch.long)
        # 创建TensorDataset
        proxy_dataset = TensorDataset(proxy_eeg_tensor, proxy_labels_tensor)


        # 读取train h5文件
        train_dataset = {}  # 用于存放7个client的数据加一个proxy的数据的字典
        if args.iid == 1:
            file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_FEATURE/TESTIID'
            file_names = [f'client{i}.h5' for i in range(1, 8)]

            for i in range(len(file_names)):
                client_eeg = {}
                file_path_full = os.path.join(file_path, file_names[i])
                file_path_full = file_path_full.replace(os.sep, '/')
                with h5py.File(file_path_full, 'r') as f:
                    eeg_data = f['eeg_data'][:]
                    labels = f['labels'][:]
                train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
                train_labels = labels
                train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
                train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
                train_dataset_i = TensorDataset(train_eeg_tensor, train_labels_tensor)

                train_dataset[i] = train_dataset_i
            train_dataset[len(file_names)] = proxy_dataset
        else:
            file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_FEATURE/TEST{}'.format(
                args.alpha_d)
            file_names = [f'client{i}_3.h5' for i in range(1, 8)]

            for i in range(len(file_names)):
                file_path_full = os.path.join(file_path, file_names[i])
                file_path_full = file_path_full.replace(os.sep, '/')
                with h5py.File(file_path_full, 'r') as f:
                    eeg_data = f['eeg_data'][:]
                    labels = f['labels'][:]
                train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
                train_labels = labels

                train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
                train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
                train_dataset[i] = TensorDataset(train_eeg_tensor, train_labels_tensor)
            train_dataset[len(file_names)] = proxy_dataset

    elif args.dataset == 'seediv':
        # 读取test HDF5文件
        test_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV/SEEDIV_FEATURE/TESTIID_3Session/FEATUREtest15_3session.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        test_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        test_labels = labels
        # 转为pytorch张量
        test_eeg_tensor = torch.tensor(test_eeg, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
        # 创建TensorDataset
        test_dataset = TensorDataset(test_eeg_tensor, test_labels_tensor)

        # 读取proxy HDF5文件
        proxy_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV/SEEDIV_FEATURE/TESTIID_3Session/proxy_all_3session.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        proxy_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        proxy_labels = labels
        # 转为pytorch张量
        proxy_eeg_tensor = torch.tensor(proxy_eeg, dtype=torch.float32)
        proxy_labels_tensor = torch.tensor(proxy_labels, dtype=torch.long)
        # 创建TensorDataset
        proxy_dataset = TensorDataset(proxy_eeg_tensor, proxy_labels_tensor)

        # 读取train h5文件
        train_dataset = {}  # 用于存放7个client的数据加一个proxy的数据的字典
        if args.iid == 1:
            file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV/SEEDIV_FEATURE/TESTIID_3Session'
            file_names = [f'client{i}.h5' for i in range(1, 8)]

            for i in range(len(file_names)):
                client_eeg = {}
                file_path_full = os.path.join(file_path, file_names[i])
                file_path_full = file_path_full.replace(os.sep, '/')
                with h5py.File(file_path_full, 'r') as f:
                    eeg_data = f['eeg_data'][:]
                    labels = f['labels'][:]
                train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
                train_labels = labels
                train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
                train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
                train_dataset_i = TensorDataset(train_eeg_tensor, train_labels_tensor)

                train_dataset[i] = train_dataset_i
            train_dataset[len(file_names)] = proxy_dataset
        else:
            file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV/SEEDIV_FEATURE/TEST{}'.format(
                args.alpha_d)
            file_names = [f'client{i}_3.h5' for i in range(1, 8)]

            for i in range(len(file_names)):
                file_path_full = os.path.join(file_path, file_names[i])
                file_path_full = file_path_full.replace(os.sep, '/')
                with h5py.File(file_path_full, 'r') as f:
                    eeg_data = f['eeg_data'][:]
                    labels = f['labels'][:]
                train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
                train_labels = labels

                train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
                train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
                train_dataset[i] = TensorDataset(train_eeg_tensor, train_labels_tensor)
            train_dataset[len(file_names)] = proxy_dataset

    elif args.dataset == 'crocul1':
        # 读取test HDF5文件
        test_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/CHI/CHI1.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        test_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        test_labels = labels
        # 转为pytorch张量
        test_eeg_tensor = torch.tensor(test_eeg, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
        # 创建TensorDataset
        test_dataset = TensorDataset(test_eeg_tensor, test_labels_tensor)

        # 读取proxy HDF5文件
        proxy_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/CHI/CHI_proxy.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        proxy_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        proxy_labels = labels
        # 转为pytorch张量
        proxy_eeg_tensor = torch.tensor(proxy_eeg, dtype=torch.float32)
        proxy_labels_tensor = torch.tensor(proxy_labels, dtype=torch.long)
        # 创建TensorDataset
        proxy_dataset = TensorDataset(proxy_eeg_tensor, proxy_labels_tensor)

        # 读取train h5文件
        train_dataset = {}  # 用于存放7个client的数据加一个proxy的数据的字典

        file_path_FRA = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/FRA'
        file_path_GER = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/GER'
        file_names_FRA = [f'FRA{i}.h5' for i in range(1, 5)]
        file_names_GER = [f'GER{i}.h5' for i in range(1, 5)]

        for i in range(len(file_names_FRA)):
            client_eeg = {}
            # FRA
            file_path_full = os.path.join(file_path_FRA, file_names_FRA[i])
            file_path_full = file_path_full.replace(os.sep, '/')
            with h5py.File(file_path_full, 'r') as f:
                eeg_data = f['eeg_data'][:]
                labels = f['labels'][:]
            train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
            train_labels = labels
            train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            train_dataset_i = TensorDataset(train_eeg_tensor, train_labels_tensor)

            train_dataset[i] = train_dataset_i
            # GER
            file_path_full = os.path.join(file_path_GER, file_names_GER[i])
            file_path_full = file_path_full.replace(os.sep, '/')
            with h5py.File(file_path_full, 'r') as f:
                eeg_data = f['eeg_data'][:]
                labels = f['labels'][:]
            train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
            train_labels = labels
            train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            train_dataset_i = TensorDataset(train_eeg_tensor, train_labels_tensor)

            train_dataset[i+4] = train_dataset_i

        train_dataset[args.num_clint] = proxy_dataset

    elif args.dataset == 'crocul2':
        # 读取test HDF5文件
        test_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/FRA/FRA1.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        test_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        test_labels = labels
        # 转为pytorch张量
        test_eeg_tensor = torch.tensor(test_eeg, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
        # 创建TensorDataset
        test_dataset = TensorDataset(test_eeg_tensor, test_labels_tensor)

        # 读取proxy HDF5文件
        proxy_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/FRA/FRA_proxy.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        proxy_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        proxy_labels = labels
        # 转为pytorch张量
        proxy_eeg_tensor = torch.tensor(proxy_eeg, dtype=torch.float32)
        proxy_labels_tensor = torch.tensor(proxy_labels, dtype=torch.long)
        # 创建TensorDataset
        proxy_dataset = TensorDataset(proxy_eeg_tensor, proxy_labels_tensor)

        # 读取train h5文件
        train_dataset = {}  # 用于存放7个client的数据加一个proxy的数据的字典

        file_path_CHI = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/CHI'
        file_path_GER = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/GER'
        file_names_CHI = [f'CHI{i}.h5' for i in range(1, 5)]
        file_names_GER = [f'GER{i}.h5' for i in range(1, 5)]

        for i in range(len(file_names_CHI)):
            client_eeg = {}
            # CHI
            file_path_full = os.path.join(file_path_CHI, file_names_CHI[i])
            file_path_full = file_path_full.replace(os.sep, '/')
            with h5py.File(file_path_full, 'r') as f:
                eeg_data = f['eeg_data'][:]
                labels = f['labels'][:]
            train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
            train_labels = labels
            train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            train_dataset_i = TensorDataset(train_eeg_tensor, train_labels_tensor)

            train_dataset[i] = train_dataset_i
            # GER
            file_path_full = os.path.join(file_path_GER, file_names_GER[i])
            file_path_full = file_path_full.replace(os.sep, '/')
            with h5py.File(file_path_full, 'r') as f:
                eeg_data = f['eeg_data'][:]
                labels = f['labels'][:]
            train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
            train_labels = labels
            train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            train_dataset_i = TensorDataset(train_eeg_tensor, train_labels_tensor)

            train_dataset[i+4] = train_dataset_i

        train_dataset[args.num_clint] = proxy_dataset

    elif args.dataset == 'crocul3':
        # 读取test HDF5文件
        test_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/GER/GER1.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        test_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        test_labels = labels
        # 转为pytorch张量
        test_eeg_tensor = torch.tensor(test_eeg, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
        # 创建TensorDataset
        test_dataset = TensorDataset(test_eeg_tensor, test_labels_tensor)

        # 读取proxy HDF5文件
        proxy_dataset = {}
        test_data_dir = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/GER/GER_proxy.h5'
        with h5py.File(test_data_dir, 'r') as f:
            eeg_data = f['eeg_data'][:]
            labels = f['labels'][:]
        proxy_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
        proxy_labels = labels
        # 转为pytorch张量
        proxy_eeg_tensor = torch.tensor(proxy_eeg, dtype=torch.float32)
        proxy_labels_tensor = torch.tensor(proxy_labels, dtype=torch.long)
        # 创建TensorDataset
        proxy_dataset = TensorDataset(proxy_eeg_tensor, proxy_labels_tensor)

        # 读取train h5文件
        train_dataset = {}  # 用于存放7个client的数据加一个proxy的数据的字典

        file_path_CHI = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/CHI'
        file_path_FRA = 'D:/python object/federate learning/FedVC_eeg/data/SEED_CFG/Train_2/FRA'
        file_names_CHI = [f'CHI{i}.h5' for i in range(1, 5)]
        file_names_FRA = [f'FRA{i}.h5' for i in range(1, 5)]

        for i in range(len(file_names_CHI)):
            client_eeg = {}
            # CHI
            file_path_full = os.path.join(file_path_CHI, file_names_CHI[i])
            file_path_full = file_path_full.replace(os.sep, '/')
            with h5py.File(file_path_full, 'r') as f:
                eeg_data = f['eeg_data'][:]
                labels = f['labels'][:]
            train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
            train_labels = labels
            train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            train_dataset_i = TensorDataset(train_eeg_tensor, train_labels_tensor)

            train_dataset[i] = train_dataset_i
            # FRA
            file_path_full = os.path.join(file_path_FRA, file_names_FRA[i])
            file_path_full = file_path_full.replace(os.sep, '/')
            with h5py.File(file_path_full, 'r') as f:
                eeg_data = f['eeg_data'][:]
                labels = f['labels'][:]
            train_eeg = preprocessing.scale(eeg_data).reshape(-1, 1, 310)
            train_labels = labels
            train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            train_dataset_i = TensorDataset(train_eeg_tensor, train_labels_tensor)

            train_dataset[i+4] = train_dataset_i

        train_dataset[args.num_clint] = proxy_dataset


    user_groups = 1

    return train_dataset, test_dataset, user_groups

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def get_network(net_name, args):
    if net_name == 'vgg16':
        from Models.vgg import vgg16_bn
        net = vgg16_bn(args)
    elif net_name == 'vgg13':
        from Models.vgg import vgg13_bn
        net = vgg13_bn(args)
    elif net_name == 'vgg11':
        from Models.vgg import vgg11_bn
        net = vgg11_bn(args)
    elif net_name == 'vgg19':
        from Models.vgg import vgg19_bn
        net = vgg19_bn(args)
    elif net_name == 'resnet18':
        from Models.resnet import seresnet18
        net = seresnet18(args.input_channels)
    elif net_name == 'resnet34':
        from Models.resnet import seresnet34
        net = seresnet34(args.input_channels)
    elif net_name == 'resnet50':
        from Models.resnet import seresnet50
        net = seresnet50(args.input_channels)
    elif net_name == 'shufflenet':
        from Models.shufflenet import shufflenet
        net = shufflenet()
    elif net_name == 'shufflenetv2':
        from Models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(args.input_channels, class_num=args.num_classes)
    elif net_name == 'mobilenet':
        from Models.mobilenet import mobilenet
        net = mobilenet(args.input_channels, class_num=args.num_classes)
    elif net_name == 'mobilenetv2':
        # from Models.mobilenetv2_new import mobilenetv2
        # net = mobilenetv2(args.input_channels, output_size=args.num_classes)
        from Models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(args.input_channels, class_num=args.num_classes)
    elif net_name == 'cnn_mnist':
        from Models.cnn import cnn_mnist
        net = cnn_mnist()
    elif net_name == 'cnn_cifar10':
        from Models.cnn import cnn_cifar10
        net = cnn_cifar10()
    elif net_name == 'EEG_CNN_1':
        from Models.EEG_CNN_1 import DeprNet
        net = DeprNet(4)
    elif net_name == 'EEG_CNN_2':
        from Models.EEG_CNN_2 import DeprNet
        net = DeprNet(4)
    elif net_name == 'EEG_CNN_3':
        from Models.EEG_CNN_3 import DeprNet
        net = DeprNet(4)
    elif net_name == 'EEGNet':
        from Models.EEGnet1 import EEGnet1
        net = EEGnet1()
    elif net_name == 'dnn':
        from Models.dnn import dnn
        net = dnn()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net