import scipy.io
import numpy as np
import os
import pickle
import h5py

file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_FEATURE/TESTIID'
file_path_seediv = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV/SEEDIV_FEATURE/TESTIID_3Session'
output_file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_FEATURE/TEST3.0'
output_file_path_SEEDIV = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV/SEEDIV_FEATURE/TEST3.0'
file_name = [f'client{i}.h5' for i in range(1, 8)]

# 确保保存文件夹存在
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

data_dic = {}
label_dic = {}
for i in range(len(file_name)):
    file_path_full = os.path.join(file_path_seediv, file_name[i])
    file_path_full = file_path_full.replace(os.sep, '/')

    with h5py.File(file_path_full, 'r') as f:
        eeg_data = f['eeg_data'][:]
        labels = f['labels'][:]

    data_dic[i] = eeg_data
    label_dic[i] = labels

    class_num = {}
    for i in range(4):
        class_num[i] = len([1 for x in labels if x == i])
    print(class_num)

np.random.seed(42)
alpha = 3.0
samples_per_client = 8000
dirichlet_all = []
for i in range(7):
    dirichlet_all.append(np.random.dirichlet([alpha] * 4, size=1)[0])
    # num_sample_per_class = (dirichlet_dist[i] * samples_per_client).astype(int)
print(dirichlet_all)

data_sampled = {}
label_sampled = {}

for i in range(7):
    # 假设 data 和 label 是你的数据集，dirichlet_dist 是给定的采样比例
    data = data_dic[i]  # 示例数据
    label = label_dic[i]  # 示例标签，标签为 0, 1, 2

    dirichlet_dist = dirichlet_all[i].astype(float)  # 目标比例

    # 计算每个类别的数量
    unique_labels, label_counts = np.unique(label, return_counts=True)
    label_counts_dict = dict(zip(unique_labels, label_counts))  # {0: count, 1: count, 2: count}

    # 1. 找到比例最大的类别，并确保这个类别的所有数据都被采样
    max_category_idx = np.argmax(dirichlet_dist)
    max_category_samples = label_counts_dict[max_category_idx]

    # 初始化存储采样结果
    sampled_data = []
    sampled_labels = []

    # 采样比例最大的类别，确保它的所有样本都被采样
    max_category_indices = np.where(label == max_category_idx)[0]
    sampled_data.append(data[max_category_indices])
    sampled_labels.append(label[max_category_indices])

    # 2. 计算其他类别的采样数量
    remaining_samples = max_category_samples  # 以最大类别样本数为基准

    for j in range(len(dirichlet_dist)):
        if j != max_category_idx and dirichlet_dist[j] > 0:
            # 计算当前类别应采样的样本数量
            num_to_sample = int(remaining_samples * (dirichlet_dist[j] / dirichlet_dist[max_category_idx]))

            # 确保不会采样超过该类别的样本数
            num_to_sample = min(num_to_sample, label_counts_dict[j])

            # 获取该类别的索引
            category_indices = np.where(label == j)[0]

            # 进行随机采样，确保不重复
            sampled_indices = np.random.choice(category_indices, num_to_sample, replace=False)
            sampled_data.append(data[sampled_indices])
            sampled_labels.append(label[sampled_indices])

    # 3. 合并采样结果
    sampled_data = np.vstack(sampled_data)
    sampled_labels = np.hstack(sampled_labels)

    # 生成一个随机排列的索引数组
    indices = np.random.permutation(sampled_data.shape[0])

    # 打乱数据和标签
    shuffled_data = sampled_data[indices]
    shuffled_labels = sampled_labels[indices]

    # 截取前500个样本和对应的标签
    train_data = shuffled_data[:2707]
    train_labels = shuffled_labels[:2707]

    # 使用h5py将数据保存到HDF5文件中
    OUTPUT_file_name = f'client{i+1}_3.h5'
    path = os.path.join(output_file_path_SEEDIV, OUTPUT_file_name)
    with h5py.File(path, 'w') as f:
        f.create_dataset('eeg_data', data=train_data, compression="gzip")
        f.create_dataset('labels', data=train_labels, compression="gzip")

    # 打印采样结果
    print(f"采样后的样本数量: {train_data.shape[0]}")
    print(f"采样后的类别分布: {np.unique(train_labels, return_counts=True)}")




