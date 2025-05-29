import os
import torch
# import numpy as np
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.image.inception import InceptionScore
from models.metric import inception_score
from PIL import Image
from tqdm import tqdm
from core.base_dataset import BaseDataset
from cleanfid import fid

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理（注意保持数据类型）
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 输入大小
    transforms.ToTensor()
])


# 加载图像并转换为 uint8
def load_images(path):
    images = []
    for img_name in tqdm(os.listdir(path), desc=f"Loading images from {path}"):
        img = Image.open(os.path.join(path, img_name)).convert("RGB")
        img = transform(img)
        img = (img * 255).byte()  # 转换为 uint8
        images.append(img)
    return torch.stack(images)


# 计算 FID 分数
# def compute_fid(real_path, fake_path):
#     fid = FrechetInceptionDistance(feature=2048).to(device)
#
#     real_images = load_images(real_path).to(device)
#     fake_images = load_images(fake_path).to(device)
#
#     fid.update(real_images, real=True)
#     fid.update(fake_images, real=False)
#
#     return fid.compute().item()


# 计算 IS（Inception Score）
def compute_is(fake_path):
    # inception_score = inception_score(feature=2048).to(device)
    is_mean, is_std = inception_score(BaseDataset(fake_path), cuda=True, batch_size=8, resize=True, splits=10)

    # images = load_images(fake_path).to(device)
    # inception_score.update(images)

    # return inception_score.compute()
    return is_mean, is_std


if __name__ == "__main__":
    real_img_dir = "D:\Datasets\CelebA-HQ\celeba-hq\images_val_png"  # 修改为你的原始图像目录
    fake_img_dir = "E:\Pycharm\Palette_Windows\experiments\\test_1000_200_s10\\results\\test\\1"  # 修改为你的生成图像目录

    # fid_score = compute_fid(real_img_dir, fake_img_dir)
    is_mean, is_std = compute_is(fake_img_dir)
    fid_score = fid.compute_fid(real_img_dir, fake_img_dir, num_workers=2)

    # print(f"FID Score: {fid_score}")
    print('FID: {}'.format(fid_score))
    print('IS:{} {}'.format(is_mean, is_std))
