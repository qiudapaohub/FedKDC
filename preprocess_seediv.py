import scipy.io
import numpy as np
import os
import pickle
import h5py

# 文件路径D:\python object\federate learning\FedVC_eeg\data\SEED_MAT
mat_files_path = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV'

labels = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]

np.random.seed(42)

# test
eeg_data_list = []
label_list = []
file_name = f'{15}.mat'
mat_data = scipy.io.loadmat(os.path.join(mat_files_path, file_name))

# 提取以 _eeg1 到 _eeg15 结尾的键
eeg_keys = [key for key in mat_data.keys() if any(key.endswith(f'_eeg{j}') for j in range(1, 25))]

for j, key in enumerate(eeg_keys):  # 第4到第18的值（即下标3到17）

    eeg_segment = mat_data[key]
    segments = [eeg_segment[:, k:k + 400] for k in range(0, eeg_segment.shape[1]-399, 150)]

    # 只保留长度为400的片段
    segments = [seg for seg in segments if seg.shape[1] == 400]

    eeg_data_list.extend(segments)
    label_list.extend([labels[j]] * len(segments))  # 标签对应

# 将列表转换为numpy数组
eeg_data_array = np.array(eeg_data_list)
label_array = np.array(label_list)

# 使用h5py将数据保存到HDF5文件中
OUTPUT_file_name = f'test.h5'
# path = os.path.join(output_file_path, OUTPUT_file_name)
with h5py.File(OUTPUT_file_name, 'w') as f:
    f.create_dataset('eeg_data', data=eeg_data_array, compression="gzip")
    f.create_dataset('labels', data=label_array, compression="gzip")



# proxydata
eeg_data_list = []
label_list = []

# 随机选取8000个作为proxydata
indices = np.arange(67965)
np.random.shuffle(indices,)
choose_idx = indices[0:8000]


for i in range(1, 16):
    file_name = f'{i}.mat'
    mat_data = scipy.io.loadmat(os.path.join(mat_files_path, file_name))

    # 提取以 _eeg1 到 _eeg15 结尾的键
    eeg_keys = [key for key in mat_data.keys() if any(key.endswith(f'_eeg{j}') for j in range(1, 25))]

    for j, key in enumerate(eeg_keys):  # 第4到第18的值（即下标3到17）

        eeg_segment = mat_data[key]
        segments = [eeg_segment[:, k:k + 400] for k in range(0, eeg_segment.shape[1]-399, 150)]

        # 只保留长度为400的片段
        segments = [seg for seg in segments if seg.shape[1] == 400]

        eeg_data_list.extend(segments)
        label_list.extend([labels[j]] * len(segments))  # 标签对应

# 将列表转换为numpy数组
eeg_data_array = np.array(eeg_data_list)[choose_idx]
label_array = np.array(label_list)[choose_idx]

class_num = {}
for i in range(4):
    class_num[i] = len([1 for x in label_array if x == i])
print(class_num)

# 使用h5py将数据保存到HDF5文件中
OUTPUT_file_name = f'proxy.h5'
# path = os.path.join(output_file_path, OUTPUT_file_name)
with h5py.File(OUTPUT_file_name, 'w') as f:
    f.create_dataset('eeg_data', data=eeg_data_array, compression="gzip")
    f.create_dataset('labels', data=label_array, compression="gzip")