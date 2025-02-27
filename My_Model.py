import os
import copy
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
from utils import get_dataset, get_network
from local_train import LocalUpdate, DatasetSplit, test_inference, prox_inference
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import math


def choose_E(args, candidate, Pre_local, total_err):
    total_sim = []
    for i in range(args.num_leader):
        cons_val = []
        sim_val = []
        for idx in range(args.num_clint):
            # -----sim-----
            A1_norm = torch.norm(candidate[i].type(torch.cuda.FloatTensor), 'fro')
            A2_norm = torch.norm(Pre_local[idx].type(torch.cuda.FloatTensor), 'fro')
            A1_A2 = A1_norm * A2_norm
            sum_sim = (candidate[i] * Pre_local[idx]).sum()
            # intersection_tensor = torch.logical_and(A[idx1].unsqueeze(1) == A[idx2], torch.ones_like(A[idx2], dtype=torch.bool))
            # num_sim = torch.sum(intersection_tensor).item()
            sim = (sum_sim / A1_A2).item()
            sim_val.append(sim)
            # -----err-----
            err = torch.abs(candidate[i] - Pre_local[idx])
            err_sum = torch.sum(err)
            cons_val.append(err_sum.item())
            # cons_val.append(float(torch.sum(torch.abs(candidate[i] - Pre_local[idx])).item))
        sorted_val = sorted(cons_val)
        sorted_sim = sorted(sim_val)
        index = len(sim_val) // 2  # 3
        total_err.append(sorted_val[index])  # 前1/3位置的误差
        total_sim.append(sorted_sim[index])  # 前1/3位置的误差
    E = np.mean(total_err)  # 误差阈值
    return E


def judge_qual(args, candidate, Pre_local, E):
    support_num = 0
    limit_num = args.num_clint // args.num_leader
    for idx in range(args.num_clint):
        # -----sim-----
        A1_norm = torch.norm(candidate.type(torch.cuda.FloatTensor), 'fro')
        A2_norm = torch.norm(Pre_local[idx].type(torch.cuda.FloatTensor), 'fro')
        A1_A2 = A1_norm * A2_norm
        sum_sim = (candidate * Pre_local[idx]).sum()
        # intersection_tensor = torch.logical_and(A[idx1].unsqueeze(1) == A[idx2], torch.ones_like(A[idx2], dtype=torch.bool))
        # num_sim = torch.sum(intersection_tensor).item()
        sim = (sum_sim / A1_A2).item()

        # -----err-----
        err = torch.abs(candidate - Pre_local[idx])
        err_sum = torch.sum(err).item()
        if err_sum <= E:
            support_num += 1
    if support_num >= limit_num:
        return 1
    else:
        return 0


def follower_chooser(args, leader_idx, pre_local):
    follower_idx = []
    for i in range(args.num_leader):
        # sim_list = []
        err_list = []
        # sorted_sim_list = []
        sorted_err_list = []
        sorted_idx = []
        num_follower = math.ceil(float(args.num_clint)/args.num_leader)
        clint_exclude_leader_idx = [x for x in range(args.num_clint) if x not in leader_idx]  # 将leader的idx去除
        for idx in clint_exclude_leader_idx:
            # err1:
            err = torch.abs(pre_local[leader_idx[i]] - pre_local[idx])
            err_sum = torch.sum(err)
            # err2:
            # A1_norm = torch.norm(pre_local[leader_idx[i]].type(torch.cuda.FloatTensor), 'fro')
            # A2_norm = torch.norm(pre_local[idx].type(torch.cuda.FloatTensor), 'fro')
            # A1_A2 = A1_norm * A2_norm
            # sum_sim = (pre_local[leader_idx[i]] * pre_local[idx]).sum()
            # sim = sum_sim / A1_A2    # 相似度
            err_list.append(err_sum.item())
            # sim_list.append(sim.item())
        # sorted_sim_list = sorted(sim_list)
        sorted_err_list = sorted(err_list)
        for item in sorted_err_list:
            idx = err_list.index(item)
            sorted_idx.append(clint_exclude_leader_idx[idx])
        # follower_idx.append(sorted_idx[len(clint_exclude_leader_idx)-num_follower:len(clint_exclude_leader_idx)]) # 相似度高的
        follower_idx.append(sorted_idx[0:num_follower])  # 误差低的
    return follower_idx, clint_exclude_leader_idx


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


def average_entropy_from_logits(logits):
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits/2.0, dim=1)
    # Ensure the values are within a valid range for log
    eps = 1e-9
    probabilities = torch.clamp(probabilities, eps, 1 - eps)
    # Calculate the entropy for each sample
    entropy_per_sample = -torch.sum(probabilities * torch.log(probabilities), dim=1)
    # Calculate the average entropy
    average_entropy = torch.mean(entropy_per_sample)

    return average_entropy.item()

def single_leader_train(args, device, num, dataset, leader_model,
                        follower_models, follower_idx, optimizer, criterion, leader_idx, data_dom):
    leader_model.train()
    for kd_epoch in range(args.KD_epochs):
        kd_epoch_loss = 0
        average_entropy_list = []

        for batch_idx, (images, labels) in enumerate(dataset):
            batch_follower_logit = []  # 所有follower在一个bs上的logit
            batch_follower_output = []
            images, labels = images.to(device), labels.to(device)
            if data_dom[leader_idx[num]] == '2d':
                images = images.reshape(images.size(0), -1)
            leader_model.zero_grad()
            optimizer.zero_grad()

            # get batch_leader_logit
            _, batch_leader_soft_pre, batch_leader_output = leader_model(images, t=args.temp)

            # get batch_follower_pre_mean
            with torch.no_grad():

                for fol_num in range(len(follower_idx[num])):
                    # batches = list(dataset[data_dom[follower_idx[num][fol_num]]])
                    # images_fol, label_fol = batches[batch_idx]
                    # images_fol = images_fol.to(device)
                    if data_dom[follower_idx[num][fol_num]] == '2d':
                        images = images.reshape(images.size(0), -1)
                    else:
                        images = images.reshape(images.size(0), 1, -1)
                    follower_model = follower_models[fol_num].eval()
                    _, batch_follower_soft_pre, batch_follower_output = follower_model(images, t=args.temp)
                    batch_follower_logit.append(batch_follower_output)  # 聚合logit

            stacked_tensor = torch.stack(batch_follower_logit)
            batch_follower_logit_mean = torch.mean(stacked_tensor, dim=0) # 这里使用均值法聚合logit，后续应该要改进
            # batch_follower_logit_mean = F.softmax(batch_follower_logit_mean / 0.5, dim=1)

            # 计算每个batch的平均熵，放在average_entropy里面
            batch_entropy = average_entropy_from_logits(batch_follower_logit_mean)
            average_entropy_list.append(batch_entropy)

            # print(batch_follower_logit_mean)

            # criterion
            loss = criterion(batch_leader_output, batch_follower_logit_mean, labels)
            # loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            kd_epoch_loss += loss.item()

        average_entropy = np.mean(np.array(average_entropy_list))
        print('| Leader Num : {} | Leader Epoch : {} | followers_average_entropy:{} | \tLoss: {:.6f}'.format(
            leader_idx[num], kd_epoch, average_entropy, kd_epoch_loss / len(dataset)))
    return average_entropy
    # return leader_model.state_dict()


def single_follower_train(args, device, fol_idx, dataset, leader_models,  belonged_leader_idx,
                          follower_model, optimizer, criterion, data_dom):
    follower_model.train()
    for kd_epoch in range(args.KD_epochs):
        kd_epoch_loss = 0
        average_entropy_list = []

        for batch_idx, (images, labels) in enumerate(dataset):
            batch_leader_logit = []  # 所有leader在一个bs上的logit

            images, labels = images.to(device), labels.to(device)
            if data_dom[fol_idx] == '2d':
                images = images.reshape(images.size(0), -1)
            else:
                images = images.reshape(images.size(0), 1, -1)
            follower_model.zero_grad()
            optimizer.zero_grad()

            # get batch_follower_logit
            _, batch_follower_soft_pre, batch_follower_output = follower_model(images, t=args.temp)
            # print(images.shape, batch_follower_output.shape)

            # get batch_leader_logit_mean
            with torch.no_grad():
                for lea_num in range(len(belonged_leader_idx)):
                    # batches = list(dataset[data_dom[belonged_leader_idx[lea_num]]])
                    # images_lea, label_lea = batches[batch_idx]
                    # images_lea = images_lea.to(device)
                    if data_dom[belonged_leader_idx[lea_num]] == '2d':
                        images = images.reshape(images.size(0), -1)
                    else:
                        images = images.reshape(images.size(0), 1, -1)
                    leader_model = leader_models[lea_num].eval()
                    _, batch_leader_soft_pre, batch_leader_output = leader_model(images, t=args.temp)
                    batch_leader_logit.append(batch_leader_output)
            stacked_tensor = torch.stack(batch_leader_logit)
            batch_leader_logit_mean = torch.mean(stacked_tensor, dim=0)

            # 计算每个batch的平均熵，放在average_entropy里面
            batch_entropy = average_entropy_from_logits(batch_leader_logit_mean)
            average_entropy_list.append(batch_entropy)

            # criterion
            loss = criterion(batch_follower_output, batch_leader_logit_mean, labels)
            # loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            kd_epoch_loss += loss.item()

        average_entropy = np.mean(np.array(average_entropy_list))
        print('| Follower Num : {} | follower Epoch : {} | leader_average_entropy:{} | belong_leader_num:{} | \tLoss: {:.6f}'.format(
            fol_idx, kd_epoch, average_entropy, len(belonged_leader_idx), kd_epoch_loss / len(dataset)))
    return average_entropy


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
            test_acc, outputs_clint, clients_pred = prox_inference(args, model, proxy_dataset, device, idx, epoch, data_dom)
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
    save_folder = 'D:/python object/federate learning/FedVC_eeg/result_FedVC/SEEDIV/leader4/f{}'.format(args.alpha_d)
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
    # plt.show()


def save_entropy(total_entropy, filename):
    # 均值
    average_entropy = sum(total_entropy) / len(total_entropy)
    # save
    target_file_path = 'D:/python object/federate learning/FedVC_eeg/result'
    if not os.path.exists(target_file_path):
        os.makedirs(target_file_path, exist_ok=True)
    file_path = os.path.join(target_file_path, filename)
    with open(file_path, 'a') as file:
        file.write(f"{average_entropy:.3f}\n")

def write_acc(round, acc, args, loss):
    # target_file_path = 'D:/python object/federate learning/FedVC_update/result_FedAvg/CIFAR10/K{}/acc.txt'.format(args.choose_classes)
    target_file_path = 'D:/python object/federate learning/FedVC_eeg/result_FedVC/SEED/leader3/f{}'.format(
        args.alpha_d)
    if not os.path.exists(target_file_path):
        os.makedirs(target_file_path, exist_ok=True)

    file_path_acc = os.path.join(target_file_path, 'output{}.csv'.format(args.alpha_d))
    file_path_loss = os.path.join(target_file_path, 'loss{}.csv'.format(args.alpha_d))

    temp_df_acc = pd.DataFrame([acc])
    temp_df_loss = pd.DataFrame([loss])

    # if not os.path.exists(file_path_acc):
    #     # 如果文件不存在，则写入数据并保存列名（表头）
    #     temp_df_acc.to_csv(file_path_acc, index=False, header=False)
    # else:
    #     # 如果文件已存在，则追加数据并且不保存列名
    #     temp_df_acc.to_csv(file_path_acc, mode='a', index=False, header=False)

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
    target_file_path = 'D:/python object/federate learning/FedVC_eeg/result_FedVC/SEED/leader3/f{}'.format(
        args.alpha_d)
    if not os.path.exists(target_file_path):
        os.makedirs(target_file_path, exist_ok=True)

    file_path = os.path.join(target_file_path, 'matrix{}.csv'.format(idx))

    df_conf_matrix = pd.DataFrame(matrix)

    df_conf_matrix.to_csv(file_path, mode='a', index=False, header=False)
    # with open(os.path.join(target_file_path, 'acc.txt'), 'a') as target_file:
    #     target_file.write("|---- Current Epoch: {}   Test Accuracy: {} \n".format((round + 1), acc))



def main():
    args = args_parser()

    torch.manual_seed(args.seed)
    logger = SummaryWriter('../logs')

    # save entropy
    save_entropy_follower = 'entropy_follower.txt'
    save_entropy_leader = 'entropy_leader.txt'
    # 确保保存文件夹存在
    # if not os.path.exists(save_entropy_follower):
    #     os.makedirs(save_entropy_follower)
    # if not os.path.exists(save_entropy_leader):
    #     os.makedirs(save_entropy_leader)

    accuracy_record = []
    max_acc = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    args.iid = 1
    args.alpha_d = 1.0

    # --------data preparation--------
    train_dataset, test_dataset, user_groups = get_dataset(args)  # user_groups的最后一类作为proxy_data
    # print(user_groups[1])

    # --------model preparation--------
    FL_local_model_list = {}
    network_idx = [1, 2, 3]
    network_idx = network_idx * (args.num_clint // len(network_idx)+1)
    for i in range(args.num_clint):
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
    data_dom = ['3d', '3d', '2d', '3d', '3d', '2d', '3d', '3d']

    # criterion = nn.CrossEntropyLoss().to(device)

    # -------------------------------------------------strat train-------------------------------------------------
    all_clints_idx = [i for i in range(args.num_clint)]
    all_clints_idx_idx = [i for i in range(args.num_clint)]
    train_cnt = [0]*args.num_clint
    temp_epoch = 0

    for epoch in range(args.epochs):
        # local train
        print('|---------start local train epoch:{}---------|'.format(epoch))
        local_test_acc = []
        local_weights, local_losses = [], []
        # args.loc_epochs要变化
        if epoch <= 3:   #20
            args.loc_epochs = 1 # 10
        elif epoch <= 10:
            args.loc_epochs = 2
        elif epoch <= 20:
            args.loc_epochs = 2
        elif epoch <= 30:
            args.loc_epochs = 2
        else:
            args.loc_epochs = 1
        # args.alpha变化，alpha越大，loss_kd的占比越大，loss_hard的占比越小
        if epoch <= 10:   # 15
            args.alpha = 0.5
        elif epoch <= 20:
            args.alpha = 0.5   #0.2
        elif epoch <= 30:
            args.alpha = 0.5
        else:
            args.alpha = 0.5
        # KD_epochs变化， 一开始小一点，后面变大
        if epoch <= 10:  #15
            args.KD_epochs = 1  #2
        elif epoch <= 20:
            args.KD_epochs = 2   #5
        elif epoch <= 30:
            args.KD_epochs = 2
        else:
            args.KD_epochs = 2
        # # lr变化
        # if epoch <= 30:
        #     args.lr = 0.0000005
        # elif epoch <= 60:
        #     args.lr = 0.0000005
        # elif epoch <= 90:
        #     args.lr = 0.0000005
        # else:
        #     args.lr = 0.0000005

        # --------criterion preparation--------
        criterion = _make_criterion(alpha=args.alpha, T=0.5, mode=args.kd_mode)

        # ----local train----
        if args.loc_epochs != 0:
            for idx in range(args.num_clint):
                FL_local_model_list[idx].train()
                local_model = LocalUpdate(args=args, dataset=train_dataset, idx=idx, logger=logger, data_dom=data_dom[idx])
                w, loss = local_model.update_weights(model=FL_local_model_list[idx], num_clint=idx, global_round=epoch)  # w = model.state_dict()

                local_losses.append(copy.deepcopy(loss))
                # --update global weights
                FL_local_model_list[idx].load_state_dict(copy.deepcopy(w))
                # local test  这里的model都变成eval了
                test_acc, outputs_clint, _, _ = test_inference(args, FL_local_model_list[idx], test_dataset, device, data_dom[idx])
                local_test_acc.append(test_acc)
            print('|---------End of local train epoch:{}---------|'.format(epoch))
            print('|---------local test acc:|', local_test_acc)
        else:
            print('|---------End of local train epoch:{}---------|'.format(epoch))
            # print('|---------local test acc:|', local_test_acc)

        # ------------------------------选择leader与followers-------------------------
        # 1.对所有clint用proxy_data预测，得到Pre
        Pre_local = []
        with torch.no_grad():
            for idx in range(args.num_clint):
                FL_local_model_list[idx].eval()
                torch.cuda.empty_cache()

                local_model = LocalUpdate(args=args, dataset=train_dataset, idx=args.num_clint, logger=logger, data_dom=data_dom[idx])
                soft_pre = local_model.prediction(model=FL_local_model_list[idx])
                # local_model.prediction(model=FL_local_model_list[idx])
                Pre_local.append(soft_pre)

        # 每过P_A_E个epoch，将train_cnt中训练次数最少的3个clint选为leader
        if temp_epoch == args.P_A_E:
            temp_epoch = 0
            sorted_indices = sorted(range(len(train_cnt)), key=lambda i: train_cnt[i])

            leader_idx = sorted_indices[:args.num_leader]
        else:
            # 2.随机选择num_leader个candidate
            candidate_idx_idx = np.random.choice(all_clints_idx_idx, args.num_leader, replace=False).tolist()

            # 3.判断每个candidate的资格
            candidate = [x for x in range(args.num_leader)]
            leader_idx = [x for x in range(args.num_leader)]
            leader_ele_success = 0  # 0--失败  1--成功
            for i in range(args.num_leader):
                candidate[i] = Pre_local[all_clints_idx[candidate_idx_idx[i]]]
            # candidate[0] = Pre_local[all_clints_idx[candidate_idx_idx[0]]]
            # candidate[1] = Pre_local[all_clints_idx[candidate_idx_idx[1]]]
            # candidate[2] = Pre_local[all_clints_idx[candidate_idx_idx[2]]]
            # 找合适的E
            total_err = []

            E = choose_E(args, candidate, Pre_local, total_err)
            print(E)
            # E = 3
            # 判断能否成为leader
            for i in range(args.num_leader):
                result = judge_qual(args, candidate[i], Pre_local, E)
                if result == 1:
                    leader_idx[i] = all_clints_idx[candidate_idx_idx[i]]
                else:
                    filtered_clint_idx = [all_clints_idx[i] for i in range(len(all_clints_idx)) if i not in candidate_idx_idx]
                    sec_candidate_idx = np.random.choice(filtered_clint_idx, 1, replace=False).tolist()
                    candidate_idx_idx.append(sec_candidate_idx[0])
                    candidate[i] = Pre_local[all_clints_idx[sec_candidate_idx[0]]]
                    result = judge_qual(args, candidate[i], Pre_local, E)
                    if result == 1:
                        leader_idx[i] = all_clints_idx[sec_candidate_idx[0]]  # leader中是leader的索引

                    else:
                        print('|------failed to elect leaders------|')
                        break
                print('|------finished leader{} election------|'.format(i))
                leader_ele_success += 1
            # 如果没有选出3个leader(if leader_ele_success ！= 3), 则跳过此次大循环，回到local train
            if leader_ele_success != args.num_leader:
                # 绘制邻接矩阵
                # print_matrix(args, train_dataset, user_groups, FL_local_model_list, device, epoch)

                # 如果是第10的倍数个epoch被跳过了，就先输出分布图然后再回到local train
                # if (epoch+1) % 10 == 0:
                #     outputs = []
                #     label = torch.tensor([i // 10000 for i in range(100000)]).reshape(100000, 1)
                #
                #     for idx in range(args.num_clint):
                #         _, outputs_clint = test_inference(args, FL_local_model_list[idx], test_dataset, device, idx, epoch)
                #         outputs.append(outputs_clint)
                #
                #     print_dis_image(outputs, label, epoch)

                continue
            # return 0

            temp_epoch += 1

        # 4.为每个leader选择followers
        follower_idx, clint_exclude_leader_idx = follower_chooser(args, leader_idx, Pre_local)  # follower_idx:[[follower1_idx],[follower2_idx],[follower3_idx]]
        print(follower_idx)
        print(leader_idx)

        # 计算每个clint的参与联邦蒸馏的次数
        for i in leader_idx:
            train_cnt[i] += 1
        temp = []
        for i in range(args.num_leader ):
            for j in follower_idx[i]:
                temp.append(j)
        for i in set(temp):
            train_cnt[i] += 1
        print("|clint_KD_train_count|", train_cnt)


        # test
        # acc = []
        # outputs = []
        # for idx in range(args.num_clint):
        #     test_acc, outputs_clint = test_inference(args, FL_local_model_list[idx], test_dataset, device,
        #                                              idx, epoch)
        #     acc.append(test_acc)
        #     outputs.append(outputs_clint)
        # print("|---- Current Epoch: {}   Test Accuracy: ".format(epoch + 1), acc)

        # -----------------------------knowledge distillation--------------------------
        if args.Fed_medium == 'model':
            # 1.更新leader的model
            for num in range(args.num_leader):
                # get model
                leader_model = FL_local_model_list[leader_idx[num]].to(device)
                follower_models = [FL_local_model_list[i].to(device) for i in follower_idx[num]]  # [model1, model2,...]
                leader_model.train()

                # get leader dataset
                leader_dataset = DataLoader(DatasetSplit(train_dataset, user_groups[leader_idx[num]]),
                                            batch_size=args.batch_size, shuffle=True)

                # Set optimizer for the leader_model
                if args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(leader_model.parameters(), lr=args.lr,
                                                momentum=0.5)
                elif args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(leader_model.parameters(), lr=args.lr,
                                                 weight_decay=1e-4)
                # kd
                single_leader_train(args, device, num, leader_dataset, leader_model,
                                    follower_models, follower_idx, optimizer, criterion)

            # 2. 更新followers的model
            for fol_idx in clint_exclude_leader_idx:
                belonged_leader_idx = []  # 存放所属leader的idx
                # 判断fol_idx是属于哪几个leader的,如果不属于任何leader，belonged_leader_idx为空集合
                for cluster_idx in range(args.num_leader):  # 0,1,2
                    if fol_idx in follower_idx[cluster_idx]:
                        belonged_leader_idx.append(leader_idx[cluster_idx])
                if len(belonged_leader_idx) == 0:
                    continue
                else:
                    # 此clint是属于belonged_leader_idx的follower，下面开始训练
                    # get model
                    follower_model = FL_local_model_list[fol_idx].to(device)
                    leader_models = [FL_local_model_list[i].to(device) for i in belonged_leader_idx]  # [model1, model2,...]
                    follower_model.train()

                    # get follower dataset
                    follower_dataset = DataLoader(DatasetSplit(train_dataset, user_groups[fol_idx]),
                                                batch_size=args.batch_size, shuffle=True)

                    # Set optimizer for the leader_model
                    if args.optimizer == 'sgd':
                        optimizer = torch.optim.SGD(leader_model.parameters(), lr=args.lr,
                                                    momentum=0.5)
                    elif args.optimizer == 'adam':
                        optimizer = torch.optim.Adam(leader_model.parameters(), lr=args.lr,
                                                     weight_decay=1e-4)

                    # kd train follower
                    single_follower_train(args, device, fol_idx, follower_dataset, leader_models, belonged_leader_idx,
                                          follower_model, optimizer, criterion)

            #  ----- test-----
            # acc = []
            # outputs = []
            # label = torch.tensor([i // 10000 for i in range(100000)]).reshape(100000, 1)
            #
            # for idx in range(args.num_clint):
            #     test_acc, outputs_clint = test_inference(args, FL_local_model_list[idx], test_dataset, device, idx, epoch)
            #     acc.append(test_acc)
            #     outputs.append(outputs_clint)
            # print("|---- Current Epoch: {}   Test Accuracy: ".format(epoch + 1), acc)

            # if epoch == 0 or epoch == 1 or epoch == 2 or epoch == 3 or (epoch+1) % 10 == 0:
            #     print_dis_image(outputs, label, epoch)


        else:
            # 若是传递logit的知识蒸馏，使用proxy_data
            proxy_dataset = DataLoader(train_dataset[args.num_clint],
                                       batch_size=args.batch_size, shuffle=False)
            # proxy_dataset['3d'] = DataLoader(train_dataset[args.num_clint]['3d'],
            #                                  batch_size=args.batch_size, shuffle=False)
            # entropy
            total_entropy_followers = []
            total_entropy_leaders = []

            # 1.更新leader的model
            for num in range(args.num_leader):

                # get model
                leader_model = FL_local_model_list[leader_idx[num]].to(device)
                follower_models = [FL_local_model_list[i].to(device) for i in follower_idx[num]]  # [model1, model2,...]

                # 之前的model都为eval所以leader要变成train
                leader_model.train()

                # Set optimizer for the leader_model
                if args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(leader_model.parameters(), lr=args.lr[leader_idx[num]],
                                                momentum=0.5, weight_decay=args.weight_decay)
                elif args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(leader_model.parameters(), lr=args.lr[leader_idx[num]],
                                                 weight_decay=args.weight_decay)

                # kd
                entropy = single_leader_train(args, device, num, proxy_dataset, leader_model,
                                    follower_models, follower_idx, optimizer, criterion, leader_idx, data_dom)
                total_entropy_followers.append(entropy.item())
                # --update leader weights
                # FL_local_model_list[leader_idx[num]].load_state_dict(copy.deepcopy(leader_model_w))
            acc = []
            outputs = []
            for idx in range(args.num_clint):
                test_acc, outputs_clint, _, _ = test_inference(args, FL_local_model_list[idx], test_dataset, device, data_dom[idx])
                acc.append(test_acc)
                outputs.append(outputs_clint)
            print("|----after leader KD: Current Epoch: {}   Test Accuracy: ".format(epoch + 1), acc)

            # 保存熵
            save_entropy(total_entropy_followers, save_entropy_follower)

            # 2. 更新followers的model
            for fol_idx in clint_exclude_leader_idx:
                belonged_leader_idx = []  # 存放所属leader的idx
                # 判断fol_idx是属于哪几个leader的,如果不属于任何leader，belonged_leader_idx为空集合
                for cluster_idx in range(args.num_leader):  # 0,1,2
                    if fol_idx in follower_idx[cluster_idx]:
                        belonged_leader_idx.append(leader_idx[cluster_idx])
                if len(belonged_leader_idx) == 0:
                    continue
                else:
                    # 此clint是属于belonged_leader_idx的follower，下面开始训练
                    # get model
                    follower_model = FL_local_model_list[fol_idx].to(device)
                    leader_models = [FL_local_model_list[i].to(device) for i in belonged_leader_idx]  # [model1, model2,...]
                    follower_model.train()

                    # Set optimizer for the leader_model
                    if args.optimizer == 'sgd':
                        optimizer = torch.optim.SGD(follower_model.parameters(), lr=args.lr[fol_idx],
                                                    momentum=0.5, weight_decay=args.weight_decay)
                    elif args.optimizer == 'adam':
                        optimizer = torch.optim.Adam(follower_model.parameters(), lr=args.lr[fol_idx],
                                                     weight_decay=args.weight_decay)

                    # kd train follower
                    entropy = single_follower_train(args, device, fol_idx, proxy_dataset, leader_models, belonged_leader_idx,
                                          follower_model, optimizer, criterion, data_dom)
                    total_entropy_leaders.append(entropy.item())

            save_entropy(total_entropy_leaders, save_entropy_leader)



        #  ----- test-----
        acc = []
        outputs = []
        loss = []
        label = torch.tensor([i // 10000 for i in range(100000)]).reshape(100000, 1)

        for idx in range(args.num_clint):
            test_acc, outputs_clint, loss_client, conf_matrix = test_inference(args, FL_local_model_list[idx], test_dataset, device, data_dom[idx])
            acc.append(test_acc)
            outputs.append(outputs_clint)
            loss.append(loss_client)
            # 最后的epooch输出混淆矩阵
            # if epoch == args.epochs-1:
            #     write_conf_matrix(idx, args, conf_matrix)
            #     print(conf_matrix)
        print("|---- Current Epoch: {}   Test Accuracy: ".format(epoch + 1), acc)
        print("|---- Current Epoch: {}   Test loss: ".format(epoch + 1), loss)

        # write_acc(epoch, acc, args, loss)
        #
        # 生成模型分布图

    #   print_dis_image(outputs, label, epoch)

        # -------绘制相似度邻接矩阵------
        # print_matrix(args, train_dataset, user_groups, FL_local_model_list, device, epoch, test_dataset, data_dom)

    a = 1


if __name__ == '__main__':
    main()



# seed: 6788*3*18+6788*3*7=509,100    6788*3*16+6788*3*7=468,372

# 1,466,208   366,552    325,824
# 5010*4*18=360720    5010*4*16= 320,640