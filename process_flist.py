import os

# 读取文件
with open('/home/daxu/桌面/train.flist', 'r') as file:
    lines = file.readlines()

# 处理每一行
new_lines = []
for line in lines:
    # 去掉行尾的换行符
    line = line.strip()

    # 替换路径前缀
    new_line = line.replace('/media/hjf/E:/Dataset/Daxu', '/media/daxu/diskd/Datasets')

    # 获取文件名部分并处理成6位数
    dir_path, filename = os.path.split(new_line)
    name, ext = os.path.splitext(filename)
    new_name = name.zfill(6) + ext

    # 重新组合成新的路径
    new_line = os.path.join(dir_path, new_name)

    # 添加到新行列表
    new_lines.append(new_line)

# 写回文件
with open('train.flist', 'w') as file:
    file.write('\n'.join(new_lines))

print("处理完成!")
