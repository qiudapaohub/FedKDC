# ----------------------------------------------------------------
# 作用是将txt文件中的输出内容中'|---- Current Epoch: 21   Test Accuracy:  []'的行提取出来，保存到acc.txt
# ----------------------------------------------------------------

# 源文件路径
source_file_path = 'D:/python object/federate learning/Semi_decentralized_FD/result_DS-FL/exp1.txt'
# 目标文件路径
target_file_path = 'D:/python object/federate learning/Semi_decentralized_FD/result_DS-FL/acc.txt'

# 打开源文件和目标文件
with open(source_file_path, 'r') as source_file, open(target_file_path, 'w') as target_file:
    # 逐行读取源文件
    for line in source_file:
        # 检查行是否以指定的字符串开头
        if line.startswith('|---- Current Epoch:'):
            # 将匹配的行写入到目标文件中
            target_file.write(line)

# 输出完成消息
print("Finished filtering lines.")
