import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
# 假设你有多个实验的结果
# 生成随机数据模拟不同轮次的多次实验
np.random.seed(0)
communication_rounds = 100
num_experiments = 10

# 随机生成模拟数据，每个实验有多个轮次的准确率
data = np.random.normal(loc=np.linspace(50, 80, communication_rounds),
                        scale=5,
                        size=(num_experiments, communication_rounds))

# 计算每轮的平均值和标准差
mean_accuracy = np.mean(data, axis=0)
std_accuracy = np.std(data, axis=0)

# 绘图
plt.figure(figsize=(10, 5))

# 绘制平均值曲线
plt.plot(mean_accuracy, color='blue', label='Mean Accuracy')

# 绘制标准差范围区域
plt.fill_between(np.arange(communication_rounds),
                 mean_accuracy - std_accuracy,
                 mean_accuracy + std_accuracy,
                 color='blue', alpha=0.2)

# 添加图例和标签
plt.xlabel('# of communication rounds')
plt.ylabel('AMP (%)')
plt.title('Accuracy vs Communication Rounds')
plt.legend()
save_folder = 'D:/python object/federate learning/FedVC_eeg'
plt.savefig(os.path.join(save_folder, 'distribute.png'), format='png')
# 显示图形
plt.show()
