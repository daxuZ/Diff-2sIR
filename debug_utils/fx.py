import pandas as pd

# 读取 image_list.txt 文件
image_list_df = pd.read_csv('/media/daxu/diskd/Datasets/CelebA/Eval/image_list.txt', delim_whitespace=True)

# 确认读取是否正确，检查前几行
print(image_list_df.head())

# 提取第三列 (orig_file)
image_list = image_list_df['orig_file'].tolist()

# 读取 list_eval_partition.txt 文件
eval_partition_df = pd.read_csv('/media/daxu/diskd/Datasets/CelebA/Eval/list_eval_partition.txt', delim_whitespace=True, header=None,
                                names=['image_name', 'partition'])

# 确认读取是否正确，检查前几行
print(eval_partition_df.head())

# 创建训练集、验证集和测试集列表
train_set = []
val_set = []
test_set = []

# 处理 partition == 0 的图像文件名
for _, row in eval_partition_df[eval_partition_df['partition'] == 0].iterrows():
    image_name = row['image_name']
    if image_name in image_list:
        train_set.append(image_name)

# 处理 partition == 1 的图像文件名
for _, row in eval_partition_df[eval_partition_df['partition'] == 1].iterrows():
    image_name = row['image_name']
    if image_name in image_list:
        val_set.append(image_name)
        #--------------------------
        # test_set.append(image_name)

# 处理 partition == 2 的图像文件名
for _, row in eval_partition_df[eval_partition_df['partition'] == 2].iterrows():
    image_name = row['image_name']
    if image_name in image_list:
        test_set.append(image_name)

# 输出训练集、验证集和测试集
print("Train Set:")
print(train_set)
print("Validation Set:")
print(val_set)
print("Test Set:")
print(test_set)

# 保存结果到文件
with open('train_set.txt', 'w') as f:
    for item in train_set:
        f.write("%s\n" % item)

with open('val_set.txt', 'w') as f:
    for item in val_set:
        f.write("%s\n" % item)

with open('test_set.txt', 'w') as f:
    for item in test_set:
        f.write("%s\n" % item)

print("Files have been created successfully.")
