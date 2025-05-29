import os

def generate_flist(directory, output_file):
    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    file.write(os.path.join(root, name) + '\n')


# 使用示例
directory = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_val_256_small_eval'
output_file = '/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq_mat/masks_val_256_small_eval.flist'
generate_flist(directory, output_file)
