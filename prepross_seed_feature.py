import scipy.io
import numpy as np
import os
import pickle
import h5py
import random

# 文件路径
mat_files_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_ORIGINAL/ExtractedFeatures'
label_file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_ORIGINAL/ExtractedFeatures/label.mat'
output_file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV'

# 读取标签文件
label_data = scipy.io.loadmat(label_file_path)
labels = label_data['label'].flatten()  # 假设标签是二维的，需要flatten

# 标签映射函数
def map_labels(label):

    return label+1

# # 映射标签
# mapped_labels = {}
# mapped_labels[1] = np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3])
# mapped_labels[2] = np.array([2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1])
# mapped_labels[3] = np.array([1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0])
np.random.seed(42)

mapped_labels = map_labels(labels)
# -------------------------------------------------------------------------------------------------------------------
# # test_feature
# for i in range(1, 4):
#     mat_files_path = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV/eeg_feature/{}'.format(i)
#     file_name = f'{15}_{i}.mat'
#     mat_data = scipy.io.loadmat(os.path.join(mat_files_path, file_name))
#
#     # 提取以 _eeg1 到 _eeg15 结尾的键
#     eeg_keys = [key for key in mat_data.keys() if any(key.startswith(f'de_LDS{j}') for j in range(2, 25))]
#
#     eeg_data_array = mat_data['de_LDS1']
#     _, num, _ = eeg_data_array.shape
#     label_array = np.zeros(num, ) + mapped_labels[i][0]
#     eeg_data_array = np.swapaxes(eeg_data_array, 0, 1)
#     eeg_data_array = np.reshape(eeg_data_array, (num, -1))
#
#     for j, key in enumerate(eeg_keys):  # 第4到第18的值（即下标3到17）
#
#         de_feature = mat_data[key]
#         _, num, _ = de_feature.shape
#         used_label = np.zeros(num,) + mapped_labels[i][j+1]
#         de_feature = np.swapaxes(de_feature, 0, 1)
#         de_feature = np.reshape(de_feature, (num,-1))
#         eeg_data_array = np.vstack((eeg_data_array, de_feature))
#         label_array = np.hstack((label_array, used_label))
#
#     if i == 1:
#         eeg_data_array_all = eeg_data_array
#         label_array_all = label_array
#     else:
#         eeg_data_array_all = np.vstack((eeg_data_array_all, eeg_data_array))
#         label_array_all = np.hstack((label_array_all, label_array))
#
#     # 使用h5py将数据保存到HDF5文件中
#     OUTPUT_file_name = f'FEATUREtest15.h5'
#     # path = os.path.join(output_file_path, OUTPUT_file_name)
#     with h5py.File(OUTPUT_file_name, 'w') as f:
#         f.create_dataset('eeg_data', data=eeg_data_array_all, compression="gzip")
#         f.create_dataset('labels', data=label_array_all, compression="gzip")



# # train_feature
# number = list(range(1, 16))
# random.shuffle(number)
# client1 = number[0:2]
# client2 = number[2:4]
# client3 = number[4:6]
# client4 = number[6:8]
# clients = [client1, client2, client3, client4]
# print(clients)
# for z in range(4):
#
#     for i in clients[z]:
#         # for g in range(1,4):
#         # mat_files_path = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV/eeg_feature/{}'.format(g)
#         file_name = f'{i}_{1}.mat'
#         mat_data = scipy.io.loadmat(os.path.join(mat_files_path, file_name))
#
#         # 提取以 _eeg1 到 _eeg15 结尾的键
#         eeg_keys = [key for key in mat_data.keys() if any(key.startswith(f'de_LDS{j}') for j in range(2, 16))]
#
#         eeg_data_array = mat_data['de_LDS1']
#         _, num, _ = eeg_data_array.shape
#         label_array = np.zeros(num, ) + mapped_labels[0]
#         eeg_data_array = np.swapaxes(eeg_data_array, 0, 1)
#         eeg_data_array = np.reshape(eeg_data_array, (num, -1))
#
#         for j, key in enumerate(eeg_keys):  # 第4到第18的值（即下标3到17）
#
#             de_feature = mat_data[key]
#             _, num, _ = de_feature.shape
#             used_label = np.zeros(num,) + mapped_labels[j+1]
#             de_feature = np.swapaxes(de_feature, 0, 1)
#             de_feature = np.reshape(de_feature, (num,-1))
#             eeg_data_array = np.vstack((eeg_data_array, de_feature))
#             label_array = np.hstack((label_array, used_label))
#         if i == clients[z][0]:
#             eeg_data_array_all = eeg_data_array
#             label_array_all = label_array
#         else:
#             eeg_data_array_all = np.vstack((eeg_data_array_all, eeg_data_array))
#             label_array_all = np.hstack((label_array_all, label_array))
#
#     idx = np.random.permutation(eeg_data_array_all.shape[0])
#     eeg_data_shuffled = eeg_data_array_all[idx]
#     label_shuffled = label_array_all[idx]
#
#     eeg_data_final = eeg_data_shuffled[:4456]
#     label_final = label_shuffled[:4456]
#
#     # 使用h5py将数据保存到HDF5文件中
#     OUTPUT_file_name = f'CHI{z}.h5'
#     # path = os.path.join(output_file_path, OUTPUT_file_name)
#     with h5py.File(OUTPUT_file_name, 'w') as f:
#         f.create_dataset('eeg_data', data=eeg_data_final, compression="gzip")
#         f.create_dataset('labels', data=label_final, compression="gzip")



# proxydata
for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
    file_name = f'{i}_{1}.mat'
    mat_data = scipy.io.loadmat(os.path.join(mat_files_path, file_name))

    # 提取以 _eeg1 到 _eeg15 结尾的键
    eeg_keys = [key for key in mat_data.keys() if any(key.startswith(f'de_LDS{j}') for j in range(2, 16))]

    eeg_data_array = mat_data['de_LDS1']
    _, num, _ = eeg_data_array.shape
    label_array = np.zeros(num, ) + mapped_labels[0]
    eeg_data_array = np.swapaxes(eeg_data_array, 0, 1)
    eeg_data_array = np.reshape(eeg_data_array, (num, -1))

    for j, key in enumerate(eeg_keys):  # 第4到第18的值（即下标3到17）

        de_feature = mat_data[key]
        _, num, _ = de_feature.shape
        used_label = np.zeros(num, ) + mapped_labels[j + 1]
        de_feature = np.swapaxes(de_feature, 0, 1)
        de_feature = np.reshape(de_feature, (num, -1))
        eeg_data_array = np.vstack((eeg_data_array, de_feature))
        label_array = np.hstack((label_array, used_label))
    if i == 1:
        eeg_data_array_all = eeg_data_array
        label_array_all = label_array
    else:
        eeg_data_array_all = np.vstack((eeg_data_array_all, eeg_data_array))
        label_array_all = np.hstack((label_array_all, label_array))

# 生成一个随机排列的索引数组
indices = np.random.permutation(eeg_data_array_all.shape[0])

# 打乱数据和标签
shuffled_data = eeg_data_array_all[indices]
shuffled_labels = label_array_all[indices]

# 截取前500个样本和对应的标签
train_data = shuffled_data[:5000]
train_labels = shuffled_labels[:5000]

# 使用h5py将数据保存到HDF5文件中
OUTPUT_file_name = f'CHI_proxy_14.h5'
# path = os.path.join(output_file_path, OUTPUT_file_name)
with h5py.File(OUTPUT_file_name, 'w') as f:
    f.create_dataset('eeg_data', data=train_data, compression="gzip")
    f.create_dataset('labels', data=train_labels, compression="gzip")