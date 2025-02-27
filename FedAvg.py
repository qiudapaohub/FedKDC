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
    save_folder = 'D:/python object/federate learning/FedVC_eeg/result_FedAvg/SEED/f{}'.format(args.alpha_d)
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
    target_file_path = 'D:/python object/federate learning/FedVC_eeg/result_FedAvg/SEED/f{}'.format(
        args.alpha_d)
    if not os.path.exists(target_file_path):
        os.makedirs(target_file_path, exist_ok=True)

    file_path_acc = os.path.join(target_file_path, 'output{}.csv'.format(args.alpha_d))
    file_path_loss = os.path.join(target_file_path, 'loss{}.csv'.format(args.alpha_d))

    temp_df_acc = pd.DataFrame([acc])
    temp_df_loss = pd.DataFrame([loss])
    #
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

''' 这里是使用FedAvg做对比试验。下面是一些设置：
----1.client设置：一共设置10个client，一个sever。
----2.数据集划分：将训练数据集划分为11份，前十份分别为每个client的local_data，最后一份为proxy_data,但我们只使用前十份分配给client
      测试集不用划分。
----3.模型分配：使用VGG11给global model
'''
def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    logger = SummaryWriter('../logs')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.iid = 1
    args.alpha_d = 11

    # --------data preparation--------
    train_dataset, test_dataset, user_groups = get_dataset(args)  # user_groups的最后一类作为proxy_data
    # print(user_groups[1])

    # --------model preparation--------
    global_model = get_network(args.model8, args).to(device)  #VGG11
    total_params = sum(p.numel() for p in global_model.parameters())
    print(total_params*7*2)

    local_model_test = get_network(args.model11, args).to(device)
    # Set the model to train and send it to device.
    global_model.to(device)
    local_model_test.to(device)
    global_model.train()
    local_model_test.eval()

    data_dom = ['2d', '2d', '2d', '2d', '2d', '2d', '2d', '2d']

    # copy weights
    global_weights = global_model.state_dict()  # state_dict用于返回模型的参数字典。这个字典包含了模型中所有参数的名称和对应的张量值。

    # -------------------------------------------------strat train-------------------------------------------------
    for epoch in tqdm(range(args.epochs)):
        # local train
        print('|---------start local train epoch:{}---------|'.format(epoch))
        local_test_acc = []
        local_weights, local_losses = [], []

        # # lr变化
        # if epoch <= 30:
        #     args.lr = 0.0003
        # elif epoch <= 60:
        #     args.lr = 0.00015
        # elif epoch <= 90:
        #     args.lr = 0.0001
        # else:
        #     args.lr = 0.00005

        # ----local train----
        global_model.train()

        # 每轮communication选择训练的client
        idxs_users = np.random.choice(range(args.num_clint), args.m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idx=idx, logger=logger, data_dom=data_dom[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                 num_clint=idx, global_round=epoch)  # w = model.state_dict()
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            local_model_test.load_state_dict(w)

            test_acc, outputs_clint,_,_ = test_inference(args, local_model_test, test_dataset, device, data_dom[idx])
            local_test_acc.append(test_acc)

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        print('|---------End of local train epoch:{}---------|'.format(epoch))
        print('|---------local test acc:|', local_test_acc)

        #  ------------------ test--------------------
        acc = []
        outputs = []
        label = torch.tensor([i // 10000 for i in range(100000)]).reshape(100000, 1)

        test_acc, outputs_clint, loss_client, conf_matrix = test_inference(args, global_model, test_dataset, device, data_dom[0])
        print("|---- Current Epoch: {}   Test Accuracy: ".format(epoch + 1), test_acc)
        print("|---- Current Epoch: {}   Test loss: ".format(epoch + 1), loss_client)
        write_acc(epoch, test_acc, args, loss_client)


if __name__ == '__main__':
    main()


# 8770986