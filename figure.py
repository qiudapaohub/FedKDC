import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

# 示例数据：表示7个客户端，每个客户端有3个类别的数据样本数
sample_counts = np.array([
    [380, 1830, 63,  266, 1752, 733,  1531, 1083, 973],
    [20, 2251, 2,    2002, 372, 377,  645, 1756, 1186],
    [2043, 62, 168,  49, 1862, 840,   822, 1501, 1264],
    [650, 1332, 291, 708, 13, 2030,   420, 927, 2240],
    [710, 152, 1411, 2208, 295, 248,  2092, 834, 661],
    [1, 2213, 59,    444, 744, 1563,  1497, 943, 1147],
    [2208, 61, 4,    830, 497, 1424,  1156, 565, 1866]
])

# 定义横坐标（样本类别为循环的0, 1, 2）
class_labels = np.tile([0, 1, 2], 3)  # 类别：0, 1, 2, 0, 1, 2, 0, 1, 2

# 定义纵坐标（7个客户端ID）
client_ids = np.arange(7)

# 创建横纵坐标
x = np.tile(np.arange(9), len(client_ids))  # 横坐标重复9次（9个类别）
y = np.repeat(client_ids, len(class_labels))  # 纵坐标重复7次（7个客户端）

# 将样本数量展开为1D数组，作为气泡的大小
sizes = sample_counts.flatten() * 0.5

# 如果气泡过大，使用np.clip对气泡大小进行裁剪
sizes = np.clip(sizes, 0, 1200)  # 设置气泡大小的范围，防止过大或过小

# 定义三组颜色
colors_map = ['#92AEDE', '#92AEDE', '#92AEDE',  # 第一组颜色
              '#86CB66', '#86CB66', '#86CB66',  # 第二组颜色
              '#9A78B4', '#9A78B4', '#9A78B4']  # 第三组颜色

# 为每个客户端重复颜色
colors = np.tile(colors_map, len(client_ids))  # 为每个客户端重复颜色

# 创建自定义颜色条
cmap = mcolors.ListedColormap(['#92AEDE', '#86CB66', '#9A78B4'])
bounds = [0, 1, 2, 3]  # 颜色边界
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# 设置图形大小和比例
plt.figure(figsize=(10, 6))

# 绘制气泡图
sc = plt.scatter(x, y, s=sizes, color=colors, alpha=0.7, cmap=cmap, norm=norm)

# 设置坐标轴和标题
plt.xlabel('Class Labels', fontsize=14)
plt.ylabel('Client IDs', fontsize=14)
plt.title('Sample Distribution per Client', fontsize=16)

# 设置x轴和y轴的范围和刻度
plt.xticks(np.arange(9), np.tile([0, 1, 2], 3))  # 使得横坐标正确显示0,1,2重复
plt.yticks(client_ids)  # 显示7个客户端的标记

# 添加颜色条，并指定标签
cbar = plt.colorbar(sc, ticks=[0.5, 1.0, 3.0])  # 设置颜色条的刻度标签
cbar.set_label('Custom Values')  # 添加颜色条标签
cbar.ax.set_yticklabels(['0.5', '1.0', '3.0'])  # 设置颜色条标签显示

# 保存图形为PDF格式
save_folder = 'D:/python object/federate learning/FedVC_eeg'
plt.savefig(os.path.join(save_folder, 'distribute.png'), format='png')

# 显示图形
plt.show()
