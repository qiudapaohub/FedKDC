import scipy.io
import numpy as np
import os
import pickle
import h5py

# 文件路径
mat_files_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_ORIGINAL/Preprocessed_EEG'
label_file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_ORIGINAL/Preprocessed_EEG/label.mat'
output_file_path = 'D:/python object/federate learning/FedVC_eeg/data/SEED'

# 读取标签文件
label_data = scipy.io.loadmat(label_file_path)
labels = label_data['label'].flatten()  # 假设标签是二维的，需要flatten

# 标签映射函数
def map_labels(label):
    if label == -1:
        return 2
    else:
        return label

# 映射标签
mapped_labels = np.array([map_labels(label) for label in labels])

np.random.seed(42)
indices = np.arange(9006)
np.random.shuffle(indices,)
choose_idx = indices[0:6000]

# # 遍历所有的MAT文件
# for i in range(1, 8):
#     # 初始化存放数据和标签的列表
#     eeg_data_list = []
#     label_list = []
#
#     for x in range(2):
#         for z in range(1, 4):
#             file_name = f'{i*2-x}_{z}.mat'
#             mat_data = scipy.io.loadmat(os.path.join(mat_files_path, file_name))
#
#             # 提取以 _eeg1 到 _eeg15 结尾的键
#             eeg_keys = [key for key in mat_data.keys() if any(key.endswith(f'_eeg{j}') for j in range(1, 16))]
#
#             for j, key in enumerate(eeg_keys):  # 第4到第18的值（即下标3到17）
#
#                 eeg_segment = mat_data[key]
#                 segments = [eeg_segment[:, k:k + 400] for k in range(0, eeg_segment.shape[1]-399, 450)]
#
#                 # 只保留长度为400的片段
#                 segments = [seg for seg in segments if seg.shape[1] == 400]
#
#                 eeg_data_list.extend(segments)
#                 label_list.extend([mapped_labels[j]] * len(segments))  # 标签对应
#             a = 1
#     # 将列表转换为numpy数组
#     eeg_data_array = np.array(eeg_data_list)[choose_idx]
#     label_array = np.array(label_list)[choose_idx]
#
#     class_num = {}
#     for k in range(3):
#         class_num[k] = len([1 for x in label_array if x == k])
#     print(class_num)
#
#     # 使用h5py将数据保存到HDF5文件中
#     OUTPUT_file_name = f'client{i}.h5'
#     # path = os.path.join(output_file_path, OUTPUT_file_name)
#     # path = path.replace(os.sep, '/')
#     # np.savez_compressed(OUTPUT_file_name, eeg_data=eeg_data_array, labels=label_array)
#
#     with h5py.File(OUTPUT_file_name, 'w') as f:
#         f.create_dataset('eeg_data', data=eeg_data_array, compression="gzip")
#         f.create_dataset('labels', data=label_array, compression="gzip")



# test
eeg_data_list = []
label_list = []

file_name = f'{15}_1.mat'
mat_data = scipy.io.loadmat(os.path.join(mat_files_path, file_name))

# 提取以 _eeg1 到 _eeg15 结尾的键
eeg_keys = [key for key in mat_data.keys() if any(key.endswith(f'_eeg{j}') for j in range(1, 16))]

for j, key in enumerate(eeg_keys):  # 第4到第18的值（即下标3到17）

    eeg_segment = mat_data[key]
    segments = [eeg_segment[:, k:k + 400] for k in range(0, eeg_segment.shape[1]-399, 150)]

    # 只保留长度为400的片段
    segments = [seg for seg in segments if seg.shape[1] == 400]

    eeg_data_list.extend(segments)
    label_list.extend([mapped_labels[j]] * len(segments))  # 标签对应

# 将列表转换为numpy数组
eeg_data_array = np.array(eeg_data_list)
label_array = np.array(label_list)

# 使用h5py将数据保存到HDF5文件中
OUTPUT_file_name = f'test.h5'
# path = os.path.join(output_file_path, OUTPUT_file_name)
with h5py.File(OUTPUT_file_name, 'w') as f:
    f.create_dataset('eeg_data', data=eeg_data_array, compression="gzip")
    f.create_dataset('labels', data=label_array, compression="gzip")



# # proxydata
# eeg_data_list = []
# label_list = []
#
# # 随机选取8000个作为proxydata
# indices = np.arange(21014)    # 没有最后一个被试的数据
# np.random.shuffle(indices,)
# choose_idx = indices[0:18000]
#
#
# for i in range(1, 16):
#     # for z in range(1, 4):
#     file_name = f'{i}_{1}.mat'
#     mat_data = scipy.io.loadmat(os.path.join(mat_files_path, file_name))
#
#     # 提取以 _eeg1 到 _eeg15 结尾的键
#     eeg_keys = [key for key in mat_data.keys() if any(key.endswith(f'_eeg{j}') for j in range(1, 16))]
#
#     for j, key in enumerate(eeg_keys):  # 第4到第18的值（即下标3到17）
#
#         eeg_segment = mat_data[key]
#         segments = [eeg_segment[:, k:k + 400] for k in range(0, eeg_segment.shape[1]-399, 450)]
#
#         # 只保留长度为400的片段
#         segments = [seg for seg in segments if seg.shape[1] == 400]
#
#         eeg_data_list.extend(segments)
#         label_list.extend([mapped_labels[j]] * len(segments))  # 标签对应
#
# # 将列表转换为numpy数组
# eeg_data_array = np.array(eeg_data_list)[choose_idx]
# label_array = np.array(label_list)[choose_idx]
#
# class_num = {}
# for i in range(3):
#     class_num[i] = len([1 for x in label_array if x == i])
#
# # 使用h5py将数据保存到HDF5文件中
# OUTPUT_file_name = f'test1_15.h5'
# # path = os.path.join(output_file_path, OUTPUT_file_name)
# with h5py.File(OUTPUT_file_name, 'w') as f:
#     f.create_dataset('eeg_data', data=eeg_data_array, compression="gzip")
#     f.create_dataset('labels', data=label_array, compression="gzip")



