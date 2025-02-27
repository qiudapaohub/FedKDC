import os
import numpy as np
import pickle
import scipy.io
import random
import h5py

mat_files_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED_FRA/French/02-EEG-DE-features/eeg_used_1s'


#
# # 生成 1 到 8 的列表
# numbers = list(range(1, 9))
# np.random.seed(42)
# # 随机打乱顺序
# random.shuffle(numbers)
# # 分成四组
# client1 = numbers[0:2]
# client2 = numbers[2:4]
# client3 = numbers[4:6]
# client4 = numbers[6:8]
# # 将每组放入单独的列表中
# clients = [client1, client2, client3, client4]
# print(clients[0], clients[1], clients[2], clients[3])
#
# for i in range(4):  # 第i个client
#     data_feature = []
#     label = []
#     for j in clients[i]:   # 第j个sub
#         file_name = f'{j}_1.npz'
#         npz_data = np.load(os.path.join(mat_files_path, file_name))
#
#         data = pickle.loads(npz_data['train_data'])
#         values = list(data.values())
#
#         data_feature.append(np.hstack(values))
#         label.append(npz_data['train_label'])
#
#     eeg_data_array = np.vstack(data_feature)
#     label_array = np.hstack(label)
#
#     # 切割  17824  19280
#     idx = np.random.permutation(eeg_data_array.shape[0])
#
#     # 打乱数据和标签
#     shuffled_data = eeg_data_array[idx]
#     shuffled_labels = label_array[idx]
#
#     data_final = shuffled_data[:4456]
#     label_final = shuffled_labels[:4456]
#
#     # 使用h5py将数据保存到HDF5文件中
#     OUTPUT_file_name = f'GER{i+1}.h5'
#     # path = os.path.join(output_file_path, OUTPUT_file_name)
#     with h5py.File(OUTPUT_file_name, 'w') as f:
#         f.create_dataset('eeg_data', data=data_final, compression="gzip")
#         f.create_dataset('labels', data=label_final, compression="gzip")



# PROXY_ALL
FRA = [2,5,6,7,4,3]
GER = [6,7,2,3,1,5]
data_feature = []
label = []
for j in FRA:   # 第j个sub
    file_name = f'{j}_1.npz'
    npz_data = np.load(os.path.join(mat_files_path, file_name))

    data = pickle.loads(npz_data['train_data'])
    values = list(data.values())

    data_feature.append(np.hstack(values))
    label.append(npz_data['train_label'])

eeg_data_array = np.vstack(data_feature)
label_array = np.hstack(label)

# 切割  17824  19280
idx = np.random.permutation(eeg_data_array.shape[0])

# 打乱数据和标签
shuffled_data = eeg_data_array[idx]
shuffled_labels = label_array[idx]

data_final = shuffled_data[:5000]
label_final = shuffled_labels[:5000]

# 使用h5py将数据保存到HDF5文件中
OUTPUT_file_name = f'FRA_proxy.h5'
# path = os.path.join(output_file_path, OUTPUT_file_name)
with h5py.File(OUTPUT_file_name, 'w') as f:
    f.create_dataset('eeg_data', data=data_final, compression="gzip")
    f.create_dataset('labels', data=label_final, compression="gzip")