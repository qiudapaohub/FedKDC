import os
import scipy.io as sio
import numpy as np
import pickle

# 假设.mat文件存放在当前目录下的DEAP_data文件夹中
data_directory = 'D:/python_projects/federated_learning/FedVC/data/DEAP_MAT'

# 客户端分配列表
client_assignments = [
    [7, 8, 12], [4, 5, 6], [11, 2, 9], [10, 1, 3],
    [27, 14, 15], [32, 26, 31], [19, 20, 13], [22, 23, 24],
    [25, 17, 21], [28, 29, 30]
]

# 设置随机种子以确保每次抽样相同
random_seed = 42
np.random.seed(random_seed)


# 映射标签函数
def map_labels(labels):
    mapped_labels = []
    for l in labels:
        if 1 <= l < 4:
            mapped_labels.append(0)
        elif 4 <= l < 7:
            mapped_labels.append(1)
        elif 7 <= l <= 9:
            mapped_labels.append(2)
    return np.array(mapped_labels)


# 处理每个客户端的数据
clients_data = {}
proxy_dataset = {}
processed_proxy_data = []
processed_proxy_labels = []
remaining_data = []
remaining_labels = []
for client_id, subject_ids in enumerate(client_assignments):
    all_data = []
    all_labels = []
    all_mapped_labels = []

    # 读取并处理每个被试的数据
    for subject_id in subject_ids:
        file_name = f's{subject_id:02d}.mat'
        file_path = os.path.join(data_directory, file_name)

        if os.path.exists(file_path):
            data = sio.loadmat(file_path)
            labels = data['labels'][:, 0]
            mapped_labels = map_labels(labels)

            # 提取并组合所有数据和标签
            all_data.append(data['data'][:, :32, 384:])  # 去除前3秒（128*3=384个数据点）
            all_labels.append(labels)
            all_mapped_labels.append(mapped_labels)
        else:
            print(f"文件 {file_path} 不存在")

    # 将组合的数据和标签拼接
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_mapped_labels = np.concatenate(all_mapped_labels, axis=0)

    # 随机抽样每个标签30个样本
    indices_0 = np.random.choice(np.where(all_mapped_labels == 0)[0], 30, replace=False)
    indices_1 = np.random.choice(np.where(all_mapped_labels == 1)[0], 30, replace=False)
    indices_2 = np.random.choice(np.where(all_mapped_labels == 2)[0], 30, replace=False)
    proxy_0 = np.random.choice(np.where(all_mapped_labels == 0)[0], 3, replace=False)
    proxy_1 = np.random.choice(np.where(all_mapped_labels == 1)[0], 3, replace=False)
    proxy_2 = np.random.choice(np.where(all_mapped_labels == 2)[0], 3, replace=False)

    selected_indices = np.concatenate((indices_0, indices_1, indices_2))
    proxy_indices = np.concatenate((proxy_0, proxy_1, proxy_2))

    # 提取对应的数据样本
    selected_data = all_data[selected_indices]
    selected_labels = all_mapped_labels[selected_indices]
    proxy_data = all_data[proxy_indices]
    proxy_labels = all_mapped_labels[proxy_indices]

    # 切割数据
    processed_data = []
    processed_labels = []

    for sample, label in zip(selected_data, selected_labels):
        for start in range(0, sample.shape[1], 256):
            segment = sample[:, start:start + 256]
            processed_data.append(segment)
            processed_labels.append(label)

    processed_data = np.array(processed_data)
    processed_labels = np.array(processed_labels)

    for sample, label in zip(proxy_data, proxy_labels):
        for start in range(0, sample.shape[1]-255, 256):
            segment = sample[:, start:start + 256]
            processed_proxy_data.append(segment)
            processed_proxy_labels.append(label)
    # 存储到字典中
    clients_data[client_id] = {'data': processed_data, 'labels': processed_labels}

    # 收集未被选中的样本以构建测试集
    remaining_indices = np.setdiff1d(np.arange(len(all_mapped_labels)), selected_indices)
    remaining_data.append(all_data[remaining_indices])
    remaining_labels.append(all_mapped_labels[remaining_indices])
# 处理proxydata
proxy_data = np.array(processed_proxy_data)
proxy_labels = np.array(processed_proxy_labels)
proxy_dataset['data'] = proxy_data
proxy_dataset['labels'] = proxy_labels

# 将所有未被选中的数据和标签拼接
remaining_data = np.concatenate(remaining_data, axis=0)
remaining_labels = np.concatenate(remaining_labels, axis=0)

# 构建标签均匀分布的测试集，每个标签抽取相同数量的样本
test_indices_0 = np.random.choice(np.where(remaining_labels == 0)[0], 30, replace=False)
test_indices_1 = np.random.choice(np.where(remaining_labels == 1)[0], 30, replace=False)
test_indices_2 = np.random.choice(np.where(remaining_labels == 2)[0], 30, replace=False)

test_indices = np.concatenate((test_indices_0, test_indices_1, test_indices_2))

# 提取测试集数据和标签
test_data = remaining_data[test_indices]
test_labels = remaining_labels[test_indices]

# 切割测试集数据
test_processed_data = []
test_processed_labels = []
for sample, label in zip(test_data, test_labels):
    for start in range(0, sample.shape[1], 256):
        segment = sample[:, start:start+256]
        test_processed_data.append(segment)
        test_processed_labels.append(label)

test_processed_data = np.array(test_processed_data)
test_processed_labels = np.array(test_processed_labels)

# 存储测试集数据
# with open('DEAP_test_data.pkl', 'wb') as f:
#     pickle.dump({'data': test_processed_data, 'labels': test_processed_labels}, f)
#
# # 将处理后的数据存储到一个文件中
# with open('DEAP_clients_data.pkl', 'wb') as f:
#     pickle.dump(clients_data, f)
with open('DEAP_proxy_dataset_small.pkl', 'wb') as f:
    pickle.dump(proxy_dataset, f)

# 示例: 显示客户端 0 的数据形状
print(f"Client 0 data shape: {clients_data[1]['data'].shape}")
print(f"Client 0 labels shape: {clients_data[1]['labels'].shape}")
print(f"DEAP_proxy_dataset shape: {proxy_dataset['data'].shape}")
print(f"DEAP_proxy_dataset labels shape: {proxy_dataset['labels'].shape}")
