import numpy as np
from torchvision import datasets, transforms
import h5py

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]  # dict_users是字典
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,  # dict_users[i]是集合
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, num_classes):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    训练集60000个， 分成11份， 每份5454个
    :return:
    """
    num_items = int(len(dataset) / (num_users+1))
    idx_class_total = [i for i in range(10)]
    dict_users = {i: np.array([]) for i in range(num_users+1)}
    idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()

    # sort labels   把idx和label堆叠后根据label排序，再得到排序后的idx
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # prepare proxy_data   5450
    proxy_data = np.array([])
    for MNIST_class_num in range(10):
        rand_data = np.random.choice(idxs[MNIST_class_num*6000:(MNIST_class_num+1)*6000], 545, replace=False)
        proxy_data = np.concatenate((proxy_data, rand_data))
    # idxs = np.array(list(set(idxs) - set(proxy_data)))  # proxy_data与clint_data不重复
    idxs = idxs[~np.isin(idxs, proxy_data)]

    # 确定每个clint中每个class有多少样本
    sampel_per_class = int(num_items / num_classes)

    # divide and assign 2 shards/client  irichlet_params
    for i in range(num_users):
        # 随机选取num_classes个类别
        rand_class_set = set(np.random.choice(idx_class_total, num_classes, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_class_set:
            class_sample = np.random.choice(idxs[rand * 5455 : (rand + 1) * 5455], sampel_per_class, replace=False)
            dict_users[i] = np.concatenate(
                (dict_users[i], class_sample), axis=0)
    dict_users[num_users] = proxy_data
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar10_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    class_num = {}
    labels = np.array(dataset.targets)
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        chosen_labels = [labels[x] for x in dict_users[i]]
        for t in range(10):
            class_num[t] = len([x for x in chosen_labels if x == t])

        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar10_noniid(dataset, num_users, num_classes):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    训练集50000个， 分成11份， 每份4545个
    """
    num_items = int(len(dataset) / (num_users + 1))
    idx_class_total = [i for i in range(10)]
    dict_users = {i: np.array([]) for i in range(num_users + 1)}
    chosen_labels = {i: np.array([]) for i in range(num_classes)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    # sort labels   把idx和label堆叠后根据label排序，再得到排序后的idx
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]


    # prepare proxy_data   4540
    proxy_data = np.array([])
    for MNIST_class_num in range(10):
        rand_data = np.random.choice(idxs[MNIST_class_num * 5000:(MNIST_class_num + 1) * 5000], 454, replace=False)
        proxy_data = np.concatenate((proxy_data, rand_data))
    idxs = idxs[~np.isin(idxs, proxy_data)]
    # idxs = np.array(list(set(idxs) - set(proxy_data)))  # proxy_data与clint_data不重复
    # label = [labels[i] for i in idxs]
    # print(label[4545], label[4546])

    # 确定每个clint中每个class有多少样本
    sampel_per_class = int(num_items / num_classes)

    # divide and assign 2 shards/client  irichlet_params
    for i in range(num_users):
        # 随机选取num_classes个类别
        rand_class_set = set(np.random.choice(idx_class_total, num_classes, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_class_set:
            class_sample = np.random.choice(idxs[rand * 4546: (rand + 1) * 4546], sampel_per_class, replace=False)
            dict_users[i] = np.concatenate(
                (dict_users[i], class_sample), axis=0)
            # chosen_label = [labels[i] for i in class_sample]
            # chosen_labels[rand] = chosen_label

    dict_users[num_users] = proxy_data
    # num_same_elements = np.sum(dict_users[0] == dict_users[6])  # 不同clint之间基本上没用重复的


    return dict_users


def seed_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)  #2214
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    class_num = {}
    labels = np.array(dataset.tensors[1])
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        chosen_labels = [labels[x] for x in dict_users[i]]
        for t in range(3):
            class_num[t] = len([x for x in chosen_labels if x == t])

        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def SEED_noniid(dataset, num_users, alpha):
    # 设置随机种子
    seed = 42
    # np.random.seed(seed)

    # 假设 labels 是一个包含标签的 NumPy 数组，总共有 24364 个样本
    labels = np.array(dataset.tensors[1])
    # 定义每个client应该分到的样本数
    samples_per_client = int(len(labels) / (num_users + 1))
    # 初始化用于存储分配结果的字典
    client_indices = {i: np.array([], dtype=int) for i in range(num_users + 1)}

    # 先把proxydata确定
    idxs = np.arange(len(dataset))
    client_indices[num_users] = np.random.choice(idxs, samples_per_client, replace=False)
    idxs = idxs[~np.isin(idxs, client_indices[num_users])]
    clients_labels = np.array([labels[x] for x in idxs])  # 去除proxy后的labels

    # 获取每个类别的样本索引
    class_indices = [idxs[list(np.where(clients_labels == i)[0])] for i in range(3)]

    # 为每个类别生成Dirichlet分布的样本分配，并分配给客户端
    for client_idx in range(num_users):
        # 生成Dirichlet分布
        dirichlet_dist = np.random.dirichlet([alpha] * 3, size=1)[0]
        num_sample_per_class = (dirichlet_dist * samples_per_client).astype(int)

        for class_idx in range(3):
            chosen_sample = np.random.choice(class_indices[class_idx], num_sample_per_class[class_idx], replace=False)
            client_indices[client_idx] = np.concatenate((client_indices[client_idx], chosen_sample))

    # 确保每个客户端分到的样本总数是2214
    for i in range(num_users):
        if len(client_indices[i]) > samples_per_client:
            client_indices[i] = client_indices[i][:samples_per_client]
        elif len(client_indices[i]) < samples_per_client:
            needed_samples = samples_per_client - len(client_indices[i])
            extra_samples = np.random.choice(np.setdiff1d(np.arange(len(labels)), client_indices[i]), needed_samples,
                                             replace=False)
            client_indices[i] = np.concatenate((client_indices[i], extra_samples))

    # 确保每个客户端的样本数量一致
    for i in range(num_users):
        assert len(client_indices[i]) == samples_per_client, f"Client {i} does not have {samples_per_client} samples."

    # 计算每个客户端中每个类别的样本数量
    client_class_counts = np.zeros((num_users+1, 3), dtype=int)

    for i in range(num_users+1):
        client_labels = labels[client_indices[i]]
        for c in range(3):
            client_class_counts[i, c] = (client_labels == c).sum().item()
    # 输出结果
    return client_indices


if __name__ == '__main__':
    # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,),
    #                                                         (0.3081,))
    #                                ]))
    dataset_train = datasets.CIFAR10('./data/cifar10/', train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    num = 10
    # d = mnist_noniid(dataset_train, num, 6)
    f = cifar10_noniid(dataset_train, num, 5)

