import os
def pad_filenames(flist_path, new_flist_path):
    with open(flist_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        dir_name, file_name = os.path.split(line)
        base_name, ext = os.path.splitext(file_name)
        new_base_name = base_name.zfill(6)
        new_file_name = new_base_name + ext
        new_full_path = os.path.join(dir_name, new_file_name)
        new_lines.append(new_full_path)

    with open(new_flist_path, 'w') as file:
        for line in new_lines:
            file.write(line + '\n')

def remove_duplicate_suffix(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg.jpg'):
            new_filename = filename[:-4]  # 去掉最后一个'.jpg'
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f'Renamed: {filename} -> {new_filename}')
        else:
            print(f'No change: {filename}')

# 调用函数处理文件并保存为新文件，更改文件名为6位数
# pad_filenames('/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq/test.flist',
#               '/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq_mat/test.flist')

# 调用函数处理目录
remove_duplicate_suffix('/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/celeba_512')