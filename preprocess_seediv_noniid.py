import scipy.io
import numpy as np
import os
import pickle
import h5py

# 文件路径
mat_files_path = 'D:/python object/federate learning/FedVC_eeg/data/SEEDIV'
# label_file_path = 'D:/python_projects/federated_learning/FedVC_depend/data/SEED_MAT/label.mat'

labels = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]

# 读取MAT文件中的EEG数据
def load_eeg_data_from_mat_files(mat_files, eeg_keys_prefix='_eeg', num_keys=15):
    eeg_data_list = []
    for mat_file in mat_files:
        mat_data = scipy.io.loadmat(mat_file)
        eeg_keys = [key for key in mat_data.keys() if any(key.endswith(f'{eeg_keys_prefix}{j}') for j in range(1, num_keys + 1))]
        for key in eeg_keys:
            eeg_data_list.append(mat_data[key])
    return eeg_data_list


# 使用滑动窗口分割数据
def window_slice(eeg_data, window_size=400, step=150):
    segments = [eeg_data[:, i:i + window_size] for i in range(0, eeg_data.shape[1] - window_size + 1, step)]
    segments = [seg for seg in segments if seg.shape[1] == window_size]
    return segments


# 根据Dirichlet分布分配样本数
def distribute_samples(total_samples, dirichlet_dist):
    samples_per_class = np.floor(total_samples * dirichlet_dist).astype(int)
    # 调整总样本数恰好为8000
    diff = total_samples - np.sum(samples_per_class)
    for i in range(diff):
        samples_per_class[i % len(samples_per_class)] += 1
    return samples_per_class


# 映射标签
mapped_labels_d = np.array(list(labels)*2)

np.random.seed(42)
alpha = 3
samples_per_client = 8000
mat_files = [os.path.join(mat_files_path, f'{i}.mat') for i in range(13, 15)]
dirichlet_dist = []
for i in range(7):
    dirichlet_dist.append(np.random.dirichlet([alpha] * 4, size=1)[0])
    num_sample_per_class = (dirichlet_dist[i] * samples_per_client).astype(int)
    print(num_sample_per_class)

total_samples = 8000

# 加载EEG数据
eeg_data_list = load_eeg_data_from_mat_files(mat_files)

# 分割数据
window_size = 400
eeg_segments = []
label_segments = []
for label, eeg_data in zip(mapped_labels_d, eeg_data_list):
    segments = window_slice(eeg_data, window_size=window_size, step=55)  # iid的分割用的80  A0.1用的25
    eeg_segments.extend(segments)
    label_segments.extend([label] * len(segments))

num_per_class = {}
for i in range(4):
    num_per_class[i] = len([x for x in label_segments if x == i])
# print(num_per_class)

# 确保样本数量符合要求
if len(eeg_segments) < total_samples:
    raise ValueError("分割后的样本数量不足")

# 打乱顺序
indices = np.arange(len(eeg_segments))
np.random.shuffle(indices)
eeg_segments = np.array(eeg_segments)[indices]
label_segments = np.array(label_segments)[indices]

# 根据Dirichlet分布分配样本
samples_per_class = distribute_samples(total_samples, dirichlet_dist[6])

# 按照分配的样本数选取数据
selected_eeg_segments = []
selected_labels = []

for class_idx, num_samples in enumerate(samples_per_class):
    class_indices = np.where(label_segments == class_idx)[0]
    selected_indices = class_indices[:num_samples]
    selected_eeg_segments.extend(eeg_segments[selected_indices])
    selected_labels.extend(label_segments[selected_indices])

# 打乱数据和标签
selected_indices = np.arange(total_samples)
np.random.shuffle(selected_indices)
selected_eeg_segments = np.array(selected_eeg_segments)[selected_indices]
selected_labels = np.array(selected_labels)[selected_indices]

print(samples_per_class, len(selected_labels))
# 存储到HDF5文件
output_file_name = 'client7.h5'

with h5py.File(output_file_name, 'w') as f:
    f.create_dataset('eeg_data', data=selected_eeg_segments, compression="gzip")
    f.create_dataset('labels', data=selected_labels, compression="gzip")