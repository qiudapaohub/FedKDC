import os
import copy
import random

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from Models import *
from arguments import args_parser
from utils import get_dataset, get_network, average_weights
from local_train import LocalUpdate, DatasetSplit, test_inference, prox_inference
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def _make_criterion(alpha=0, T=2.0, mode='cse'):
    def criterion(outputs, targets, labels):
        if mode == 'cse':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif mode == 'mse':
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = nn.MSELoss()(_p, _q) / 2
        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        _hard_loss = F.cross_entropy(outputs, labels)
        loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
        return loss

    return criterion


def single_leader_train(args, device, num, dataset, leader_model,
                        follower_models, follower_idx, optimizer, criterion, leader_idx):
    leader_model.train()
    for kd_epoch in range(args.KD_epochs):
        kd_epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(dataset):
            batch_follower_logit = []  # 所有follower在一个bs上的logit
            batch_follower_output = []
            images, labels = images.to(device), labels.to(device)
            leader_model.zero_grad()
            optimizer.zero_grad()

            # get batch_leader_logit
            _, batch_leader_soft_pre, batch_leader_output = leader_model(images, t=args.temp)

            # get batch_follower_pre_mean
            with torch.no_grad():
                for fol_num in range(len(follower_idx)):
                    follower_model = follower_models[fol_num].eval()
                    _, batch_follower_soft_pre, batch_follower_output = follower_model(images, t=args.temp)
                    batch_follower_logit.append(batch_follower_output)

            stacked_tensor = torch.stack(batch_follower_logit)
            batch_follower_logit_mean = torch.mean(stacked_tensor, dim=0) # 这里使用均值法聚合logit，后续应该要改进
            # batch_follower_logit_mean = F.softmax(batch_follower_logit_mean / 0.5, dim=1)

            # print(batch_follower_logit_mean)

            # criterion
            loss = criterion(batch_leader_output, batch_follower_logit_mean, labels)
            # loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            kd_epoch_loss += loss.item()
        print('| Leader Num : {} | Leader Epoch : {} | \tLoss: {:.6f}'.format(
            leader_idx[num], kd_epoch, kd_epoch_loss / len(dataset)))
    # return leader_model.state_dict()


def single_follower_train(args, device, fol_idx, dataset, leader_models,  belonged_leader_idx,
                          follower_model, optimizer, criterion):
    follower_model.train()
    for kd_epoch in range(args.KD_epochs):
        kd_epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(dataset):
            batch_leader_logit = []  # 所有leader在一个bs上的logit
            images, labels = images.to(device), labels.to(device)
            follower_model.zero_grad()
            optimizer.zero_grad()

            # get batch_follower_logit
            _, batch_follower_soft_pre, batch_follower_output = follower_model(images, t=args.temp)

            # get batch_leader_logit_mean
            with torch.no_grad():
                for lea_num in range(len(belonged_leader_idx)):
                    leader_model = leader_models[lea_num].eval()
                    _, batch_leader_soft_pre, batch_leader_output = leader_model(images, t=args.temp)
                    batch_leader_logit.append(batch_leader_output)
            stacked_tensor = torch.stack(batch_leader_logit)
            batch_leader_logit_mean = torch.mean(stacked_tensor, dim=0)

            # criterion
            loss = criterion(batch_follower_output, batch_leader_logit_mean, labels)
            # loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            kd_epoch_loss += loss.item()
        print('| Follower Num : {} | Leader Epoch : {} | \tLoss: {:.6f}'.format(
            fol_idx, kd_epoch, kd_epoch_loss / len(dataset)))


def print_dis_image(outputs, label, epoch):
    print('----printing images....----')
    # t-SNE可视化
    staked_output = torch.cat(outputs, dim=0)
    # label = torch.cat(label, dim=0)
    staked_output = staked_output.to('cpu')
    label = label.to('cpu')
    output_np = staked_output.detach().numpy()
    label = label.detach().numpy()
    # output_np = staked_output.detach().cpu().numpy()
    # label = label.detach().cpu().numpy()

    # 使用 t-SNE 将高维数据降到二维，便于可视化
    tsne = TSNE(n_components=2, random_state=0)
    output_tsne = tsne.fit_transform(output_np)

    # 获取类别列表
    unique_labels = np.unique(label)

    # 绘制 t-SNE 可视化图像
    plt.figure(figsize=(15, 12))
    # 指定字体
    # plt.rcParams['font.family'] = 'Arial'

    # 遍历每个类别并绘制数据点
    for lab in unique_labels:
        # 获取当前类别的数据索引
        idx = label == lab
        # 获取当前类别的 t-SNE 降维结果
        idx_flat = idx.flatten()
        nonzero_indices = np.nonzero(idx_flat)  # 获取非零元素的索引
        output_label = output_tsne[nonzero_indices]
        # 绘制当前类别的数据点，并指定不同的颜色
        plt.scatter(output_label[:, 0], output_label[:, 1], s=0.3)

    # 添加图例
    plt.legend()
    # 添加标题和标签
    plt.title('t-SNE Visualization of Epoch{} Output'.format(epoch))
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.savefig('E{}_tsne_visualization.png'.format(epoch))
    # 显示图形
    plt.show()
    print('-----images printed-----')


def print_matrix(args, train_dataset, user_groups, fl_local_model_list, device, epoch, test_dataset, data_dom):
    # -------绘制相似度邻接矩阵------
    # 得到所有clint使用proxy_dataset的输出
    clients_pred_dic = {idx: [] for idx in range(args.num_clint)}
    clients_pred_dic2 = {idx: [] for idx in range(args.num_clint)}
    acc_prox = []
    with torch.no_grad():
        for idx in range(args.num_clint):
            # data
            proxy_dataset = DataLoader(train_dataset[args.num_clint],
                                       batch_size=args.batch_size, shuffle=False)
            testloader = DataLoader(test_dataset, batch_size=128,
                                    shuffle=False)

            model = fl_local_model_list[idx].eval()
            test_acc, outputs_clint, clients_pred = prox_inference(args, model, testloader, device, idx, epoch, data_dom)
            acc_prox.append(test_acc)
            # for batch_idx, (images, labels) in enumerate(proxy_dataset):
            #     images, labels = images.to(device), labels.to(device)
            #
            #     log_outputs, softmax_output, output = model(images, t=args.temp)
            #     pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #     _, pred = torch.max(log_outputs, 1)

            clients_pred_dic[idx] = clients_pred
            # clients_pred_dic2[idx].append(pred)
        print("|---- Current Epoch: {}   proxy Accuracy: ".format(epoch + 1), acc_prox)

    # A字典中每一个元素应该是一个clint的输出 (nun_sample, num_classes)
    A = {idx: torch.cat(clients_pred_dic[idx], dim=0) for idx in
         range(args.num_clint)}
    # B = {idx: torch.cat(clients_pred_dic2[idx], dim=0) for idx in
    #      range(args.num_clint)}

    # 得到所有clint与其他clint的相似度
    # clients_similarity中每个元素是第i个clint与其他10个clint的相似度
    clients_similarity = {idx: [] for idx in range(args.num_clint)}

    for idx1 in range(args.num_clint):
        for idx2 in range(args.num_clint):
            A1_norm = torch.norm(A[idx1].type(torch.cuda.FloatTensor), 'fro')
            A2_norm = torch.norm(A[idx2].type(torch.cuda.FloatTensor), 'fro')
            A1_A2 = A1_norm * A2_norm
            sum_sim = (A[idx1] * A[idx2]).sum()
            # intersection_tensor = torch.logical_and(A[idx1].unsqueeze(1) == A[idx2], torch.ones_like(A[idx2], dtype=torch.bool))
            # num_sim = torch.sum(intersection_tensor).item()
            sim = (sum_sim / A1_A2).item()
            clients_similarity[idx1].append(sim)
    # mat_sim是相似度矩阵
    mat_sim = np.zeros([args.num_clint, args.num_clint])
    for i in range(args.num_clint):
        mat_sim[i, :] = np.array(clients_similarity[i])

    # 画图
    save_folder = 'D:/python object/federate learning/FedVC_eeg/result_FedMD/SEEDIV/f{}'.format(args.alpha_d)
    # 确保保存文件夹存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # 设置颜色映射范围
    vmin, vmax = 0.2, 1
    print('min:{},max:{}'.format(mat_sim.min(), mat_sim.max()))
    # 创建一个与邻接矩阵相同大小的网格
    plt.figure(figsize=(8, 6))
    plt.imshow(mat_sim, cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Edge Weight')
    # 显示图形
    plt.title("adjacency matrix of epoch{}".format(epoch))
    plt.savefig(os.path.join(save_folder, 'epoch{}.png'.format(epoch)))
    plt.show()


def write_acc(round, acc, args, loss):
    # target_file_path = 'D:/python object/federate learning/FedVC_update/result_FedAvg/CIFAR10/K{}/acc.txt'.format(args.choose_classes)
    target_file_path = 'D:/python object/federate learning/FedVC_eeg/result_FLKD/SEEDIV/f{}'.format(
        args.alpha_d)
    if not os.path.exists(target_file_path):
        os.makedirs(target_file_path, exist_ok=True)

    file_path_acc = os.path.join(target_file_path, 'output{}.csv'.format(args.alpha_d))
    file_path_loss = os.path.join(target_file_path, 'loss{}.csv'.format(args.alpha_d))

    temp_df_acc = pd.DataFrame([acc])
    temp_df_loss = pd.DataFrame([loss])

    if not os.path.exists(file_path_acc):
        # 如果文件不存在，则写入数据并保存列名（表头）
        temp_df_acc.to_csv(file_path_acc, index=False, header=False)
    else:
        # 如果文件已存在，则追加数据并且不保存列名
        temp_df_acc.to_csv(file_path_acc, mode='a', index=False, header=False)

    if not os.path.exists(file_path_loss):
        # 如果文件不存在，则写入数据并保存列名（表头）
        temp_df_loss.to_csv(file_path_loss, index=False, header=False)
    else:
        # 如果文件已存在，则追加数据并且不保存列名
        temp_df_loss.to_csv(file_path_loss, mode='a', index=False, header=False)
    # with open(os.path.join(target_file_path, 'acc.txt'), 'a') as target_file:
    #     target_file.write("|---- Current Epoch: {}   Test Accuracy: {} \n".format((round + 1), acc))


def write_conf_matrix(idx, args, matrix):
    # target_file_path = 'D:/python object/federate learning/FedVC_update/result_FedAvg/CIFAR10/K{}/acc.txt'.format(args.choose_classes)
    target_file_path = 'D:/python object/federate learning/FedVC_eeg/result_FedMD/SEED/f{}'.format(
        args.alpha_d)
    if not os.path.exists(target_file_path):
        os.makedirs(target_file_path, exist_ok=True)

    file_path = os.path.join(target_file_path, 'matrix{}.csv'.format(idx))

    df_conf_matrix = pd.DataFrame(matrix)

    df_conf_matrix.to_csv(file_path, mode='a', index=False, header=False)
    # with open(os.path.join(target_file_path, 'acc.txt'), 'a') as target_file:
    #     target_file.write("|---- Current Epoch: {}   Test Accuracy: {} \n".format((round + 1), acc))

''' 这里是使用FedMD做对比试验。下面是一些设置：
----1.client设置：一共设置7个client，一个sever。
----2.数据集划分：每个client分2个sub的数据，最后一个sub作为test
----3.模型分配：和本文模型一样，dnn，cnn1，cnn2
'''
def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    logger = SummaryWriter('../logs')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.iid = 1
    args.alpha_d = 10

    # --------data preparation--------
    train_dataset, test_dataset, user_groups = get_dataset(args)  # user_groups的最后一类作为proxy_data
    # print(user_groups[1])

    # --------model preparation--------
    FL_local_model_list = {}
    FL_local_global_list = {}

    network_idx = [1, 2, 3]
    network_idx = network_idx * (args.num_clint // len(network_idx) + 1)
    for i in range(args.num_clint):
        FL_local_global_list[i] = get_network(args.model11, args).to(device)

        if network_idx[i] == 1:
            FL_local_model_list[i] = get_network(args.model8, args).to(device)
        elif network_idx[i] == 2:
            FL_local_model_list[i] = get_network(args.model9, args).to(device)
        elif network_idx[i] == 3:
            FL_local_model_list[i] = get_network(args.model11, args).to(device)
        # elif network_idx[i] == 4:
        #     FL_local_model_list[i] = get_network(args.model8, args).to(device)
        else:
            print('get model failed')
            return 0
    
    FL_global_model = get_network(args.model9, args).to(device)
        

    data_dom = ['3d', '3d', '2d', '3d', '3d', '2d', '3d', '3d']

    

    # --------criterion preparation--------
    criterion = _make_criterion(alpha=args.alpha, T=4.0, mode=args.kd_mode)
    # criterion = nn.CrossEntropyLoss().to(device)

    # -------------------------------------------------strat train-------------------------------------------------
    all_clints_idx = [i for i in range(args.num_clint)]
    all_clints_idx_idx = [i for i in range(args.num_clint)]
    train_cnt = [0]*10
    temp_epoch = 0

    for epoch in tqdm(range(args.epochs)):
        # local train
        print('|---------start sever aggregate epoch:{}---------|'.format(epoch))
        local_test_acc = []
        local_weights, local_losses = [], []
        # args.loc_epochs要变化
        if epoch <= 3:  # 20
            args.loc_epochs = 2  # 10
        elif epoch <= 10:
            args.loc_epochs = 2
        elif epoch <= 20:
            args.loc_epochs = 2
        elif epoch <= 30:
            args.loc_epochs = 2
        else:
            args.loc_epochs = 2
        # args.alpha变化，alpha越大，loss_kd的占比越大，loss_hard的占比越小
        if epoch <= 10:  # 15
            args.alpha = 0.5
        elif epoch <= 20:
            args.alpha = 0.5  # 0.2
        elif epoch <= 30:
            args.alpha = 0.5
        else:
            args.alpha = 0.5
        # KD_epochs变化， 一开始小一点，后面变大
        if epoch <= 10:  # 15
            args.KD_epochs = 2  # 2
        elif epoch <= 20:
            args.KD_epochs = 2  # 5
        elif epoch <= 30:
            args.KD_epochs = 2
        else:
            args.KD_epochs = 2
        # # lr变化
        # if epoch <= 30:
        #     args.lr = 0.0001
        # elif epoch <= 60:
        #     args.lr = 0.0001
        # elif epoch <= 90:
        #     args.lr = 0.0001
        # else:
        #     args.lr = 0.00005

        # ==== sever aggregate =====
        global_weight_list = []
        for idx in range(args.num_clint):
            global_weight_list.append(copy.deepcopy(FL_local_global_list[idx]).state_dict())

        # update global weights
        global_weights = average_weights(global_weight_list)

        # update global weights
        for idx in range(args.num_clint):
            FL_local_global_list[idx].load_state_dict(global_weights)

        




        # ----local train----
        # 1. 若是传递logit的知识蒸馏，使用proxy_data
        proxy_dataset = DataLoader(train_dataset[args.num_clint],
                                   batch_size=args.batch_size, shuffle=True)
        
        if args.loc_epochs != 0:
            for idx in range(args.num_clint):
                trainloader = DataLoader(train_dataset[idx], batch_size=args.batch_size, shuffle=True)
                # --- update local model ---
                model_local = FL_local_model_list[idx]
                model_global = FL_local_global_list[idx]
                model_local.train()
                model_global.eval()

                # Set optimizer for the leader_model    优化器应该放在循环外面！！
                if args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(model_local.parameters(), lr=args.lr[idx],
                                                momentum=0.5, weight_decay=args.weight_decay)
                elif args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(model_local.parameters(), lr=args.lr[idx],
                                                weight_decay=args.weight_decay)
                
                # 跟新第idx个客户端的local_model
                for kd_local_epoch in range(args.KD_epochs):
                    kd_epoch_loss = 0
                    for batch_idx, (images, labels) in enumerate(trainloader):
                        images, labels = images.to(device), labels.to(device)
                        model_local.zero_grad()
                        optimizer.zero_grad()

                        # 计算model_global在proxydata上的输出
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                            images = images.reshape(images.size(0), -1)
                            _, batch_soft_pre, batch_logit = model_global(images, t=4.0)
                         
                        # 跟新local_model
                        if data_dom[idx] == '2d':
                            images = images.reshape(images.size(0), -1)
                        else:
                            images = images.reshape(images.size(0), 1, -1)

                        model_local.train()
                        _, batch_client_soft_pre, batch_client_output = model_local(images, t=4.0)
                        # criterion
                        loss = criterion(batch_client_output, batch_logit, labels)
                        loss.backward()
                        optimizer.step()
                        kd_epoch_loss += loss.item()
                    print('| Client Num : {} | Client Epoch : {} | client_update|\tLoss: {:.6f}'.format(
                    idx, kd_local_epoch, kd_epoch_loss / len(trainloader)))
                
                # 跟新第idx个客户端的global_model
                # Set optimizer for the leader_model    优化器应该放在循环外面！！
                model_local.eval()
                model_global.train()

                if args.optimizer == 'sgd':
                    optimizer_g = torch.optim.SGD(model_global.parameters(), lr=args.lr[idx],
                                                momentum=0.5, weight_decay=args.weight_decay)
                elif args.optimizer == 'adam':
                    optimizer_g = torch.optim.Adam(model_global.parameters(), lr=args.lr[idx],
                                                weight_decay=args.weight_decay)
                for kd_local_epoch in range(args.KD_epochs):
                    kd_epoch_loss = 0
                    for batch_idx, (images, labels) in enumerate(trainloader):
                        images, labels = images.to(device), labels.to(device)
                        model_global.zero_grad()
                        optimizer_g.zero_grad()

                        # 计算model_global在proxydata上的输出
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                            if data_dom[idx] == '2d':
                                images = images.reshape(images.size(0), -1)
                            else:
                                images = images.reshape(images.size(0), 1, -1)
                            _, batch_soft_pre, batch_logit = model_local(images, t=4.0)
                         
                        # 跟新local_model
                        images = images.reshape(images.size(0), -1)
                        model_global.train()
                        _, batch_client_soft_pre, batch_client_output = model_global(images, t=4.0)
                        # criterion
                        loss = criterion(batch_client_output, batch_logit, labels)
                        loss.backward()
                        optimizer_g.step()
                        kd_epoch_loss += loss.item()
                    print('| Client Num : {} | Client Epoch : {} | client_global_update|\tLoss: {:.6f}'.format(
                    idx, kd_local_epoch, kd_epoch_loss / len(trainloader)))
                

        #  ------------------ test--------------------
        acc = []
        outputs = []
        loss = []
        label = torch.tensor([i // 10000 for i in range(100000)]).reshape(100000, 1)

        for idx in range(args.num_clint):
            test_acc, outputs_clint, loss_client, conf_matrix = test_inference(args, FL_local_model_list[idx],
                                                                               test_dataset, device, data_dom[idx])
            acc.append(test_acc)
            loss.append(loss_client)
            outputs.append(outputs_clint)
            # 最后的epooch输出混淆矩阵
            if epoch == args.epochs - 1:
                write_conf_matrix(idx, args, conf_matrix)
                print(conf_matrix)
        print("|---- Current Epoch: {}   Test Accuracy: ".format(epoch), acc)
        print("|---- Current Epoch: {}   Test loss: ".format(epoch + 1), loss)
        write_acc(epoch, acc, args, loss)

        # # 定义要保存的文件名
        # folder_name = '../Semi_decentralized_FD/result_FedMD/exp2'
        # filename = "acc.txt"
        # full_path = os.path.join(folder_name, filename)
        #
        # # 检查文件夹是否存在，如果不存在，则创建
        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name)
        #
        # # 在循环外打开文件，以'w'模式（写模式），如果文件已存在，会被清空
        # with open(full_path, "a") as file:
        #     # 创建要打印和保存的字符串
        #     output_string = "|---- Current Epoch: {}   Test Accuracy: {}\n".format(epoch + 1, acc)
        #
        #     # 打印到控制台
        #     print(output_string, end="")
        #
        #     # 写入到文件，注意加入换行符 '\n'
        #     file.write(output_string)
        #
        # # 生成模型分布图
        # # if epoch == 0 or epoch == 1 or epoch == 2 or epoch == 3 or (epoch + 1) % 10 == 0:
        # #     print_dis_image(outputs, label, epoch)
        #
        # # -------绘制相似度邻接矩阵------
        # print_matrix(args, train_dataset, user_groups, FL_local_model_list, device, epoch, test_dataset, data_dom)

    a = 1


if __name__ == '__main__':
    main()


# 6788*3*14=285,096
# 5010*4*14=280,560