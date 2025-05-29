import os                   # 导入操作系统相关功能
import sys                  # 导入系统相关功能
import glob                 # 导入文件路径查找相关功能
import pickle               # 导入pickle序列化功能
import numpy as np          # 导入数值计算库numpy
import torch                # 导入深度学习框架PyTorch
import scipy.linalg         # 导入科学计算库中的线性代数模块
import sklearn.svm          # 导入机器学习库scikit-learn中的支持向量机模块
from PIL import Image       # 导入图像处理库Pillow
import pyspng              # 导入PNG文件处理库

sys.path.insert(0, '../')   # 将上级目录添加到系统路径，用于导入自定义模块

import dnnlib               # 导入dnnlib库
import dnnlib.util          # 导入dnnlib的工具模块

from scipy.linalg import sqrtm
from sklearn.svm import LinearSVC

_feature_detector_cache = dict()  # 定义一个空的特征检测器缓存字典

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    """
    获取特征检测器模型，支持分布式加载。

    Args:
        url (str): 模型文件的URL或路径。
        device (torch.device): 设备类型，默认为CPU。
        num_gpus (int): GPU数量，默认为1。
        rank (int): 当前GPU的排名，默认为0。
        verbose (bool): 是否显示详细信息，默认为False。

    Returns:
        torch.nn.Module: 加载并配置好的特征检测器模型。
    """
    assert 0 <= rank < num_gpus  # 断言排名在合理范围内

    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier()  # 分布式训练时，同步所有GPU
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)  # 加载并评估模型
        if is_leader and num_gpus > 1:
            torch.distributed.barrier()  # 分布式训练时，同步所有GPU
    return _feature_detector_cache[key]  # 返回特征检测器模型

def read_image(image_path):
    """
    读取图像并转换为PyTorch张量。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        torch.Tensor: 转换后的图像张量。
    """
    try:
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())  # 如果是PNG文件，使用pyspng加载
            else:
                image = np.array(Image.open(f))  # 否则使用PIL加载
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # 如果是灰度图，扩展为三通道
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)  # 如果是单通道，复制为三通道
        image = image.transpose(2, 0, 1)  # HWC => CHW
        image = torch.from_numpy(image).unsqueeze(0).to(torch.uint8)  # 转换为PyTorch张量
        return image
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

class FeatureStats:
    """
    特征统计类，用于统计特征数据的均值、协方差等信息。
    """
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        """
        初始化特征统计对象。

        Args:
            capture_all (bool): 是否捕获所有特征。
            capture_mean_cov (bool): 是否捕获均值和协方差。
            max_items (int): 最大捕获样本数。
        """
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        """
        设置特征数量。

        Args:
            num_features (int): 特征数量。
        """
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        """
        检查是否已捕获足够的样本。

        Returns:
            bool: 是否已满。
        """
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        """
        添加特征数据。

        Args:
            x (np.ndarray): 特征数据数组。
        """
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        """
        添加PyTorch张量类型的特征数据。

        Args:
            x (torch.Tensor): PyTorch张量类型的特征数据。
            num_gpus (int): GPU数量。
            rank (int): 当前GPU的排名。
        """
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1)  # 将多个GPU的数据交错排列
        self.append(x.cpu().numpy())

    def get_all(self):
        """
        获取所有捕获的特征数据。

        Returns:
            np.ndarray: 所有特征数据数组。
        """
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        """
        获取所有捕获的特征数据，并转换为PyTorch张量。

        Returns:
            torch.Tensor: 所有特征数据的PyTorch张量。
        """
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        """
        计算特征数据的均值和协方差。

        Returns:
            tuple: 均值向量和协方差矩阵。
        """
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        """
        将特征统计对象保存到pickle文件。

        Args:
            pkl_file (str): 保存的pickle文件路径。
        """
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        """
        从pickle文件加载特征统计对象。

        Args:
            pkl_file (str): 加载的pickle文件路径。

        Returns:
            FeatureStats: 加载后的特征统计对象。
        """
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)

        return obj

def calculate_metrics(folder1, folder2):
    """
    计算FID、PIDS和UIDS指标。

    Args:
        folder1 (str): 第一个文件夹路径，包含生成的图像。
        folder2 (str): 第二个文件夹路径，包含真实的图像。

    Returns:
        tuple: FID、PIDS和UIDS指标的值。
    """
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))  # 获取文件夹1中所有PNG和JPG图像路径并排序
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))  # 获取文件夹2中所有PNG和JPG图像路径并排序
    assert(len(l1) == len(l2))  # 断言文件夹1和文件夹2中图像数量相同
    print('length:', len(l1))   # 打印图像数量

    # 构建检测器
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True)  # 返回softmax层前的原始特征
    device = torch.device('cuda:0')  # 指定CUDA设备
    detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0, verbose=False)  # 获取特征检测器模型
    detector.eval()  # 设置为评估模式

    # 初始化特征统计对象
    stat1 = FeatureStats(capture_all=True, capture_mean_cov=True, max_items=len(l1))
    stat2 = FeatureStats(capture_all=True, capture_mean_cov=True, max_items=len(l1))

    with torch.no_grad():
        for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
            print(i)
            _, name1 = os.path.split(fpath1)  # 获取文件名1
            _, name2 = os.path.split(fpath2)  # 获取文件名2
            name1 = name1.split('.')[0]       # 去掉文件扩展名1
            name2 = name2.split('.')[0]       # 去掉文件扩展名2
            assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)  # 断言文件名匹配

            img1 = read_image(fpath1).to(device)  # 读取并转换图像1为PyTorch张量
            img2 = read_image(fpath2).to(device)  # 读取并转换图像2为PyTorch张量
            assert img1.shape == img2.shape, 'Illegal shape'  # 断言图像形状一致
            fea1 = detector(img1, **detector_kwargs)  # 使用特征检测器提取特征1
            stat1.append_torch(fea1, num_gpus=1, rank=0)  # 将特征1添加到统计对象1中
            fea2 = detector(img2, **detector_kwargs)  # 使用特征检测器提取特征2
            stat2.append_torch(fea2, num_gpus=1, rank=0)  # 将特征2添加到统计对象2中

    # 计算FID
    mu1, sigma1 = stat1.get_mean_cov()  # 获取统计对象1的均值和协方差
    mu2, sigma2 = stat2.get_mean_cov()  # 获取统计对象2的均值和协方差
    m = np.square(mu1 - mu2).sum()  # 计算均值之差的平方和
    # s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)  # 计算协方差之间的平方根
    s, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
    fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))  # 计算FID值

    # 计算PIDS和UIDS
    fake_activations = stat1.get_all()  # 获取所有捕获的特征数据1
    real_activations = stat2.get_all()  # 获取所有捕获的特征数据2
    # svm = sklearn.svm.LinearSVC(dual=False)  # 初始化线性支持向量机分类器
    svm = LinearSVC(dual=False)
    svm_inputs = np.concatenate([real_activations, fake_activations])  # 合并真实和生成特征数据作为SVM输入
    svm_targets = np.array([1] * real_activations.shape[0] + [0] * fake_activations.shape[0])  # SVM目标标签
    print('SVM fitting ...')  # 打印信息：拟合SVM模型
    svm.fit(svm_inputs, svm_targets)  # 训练SVM模型
    uids = 1 - svm.score(svm_inputs, svm_targets)  # 计算UIDS
    real_outputs = svm.decision_function(real_activations)  # 计算真实数据的决策函数值
    fake_outputs = svm.decision_function(fake_activations)  # 计算生成数据的决策函数值
    pids = np.mean(fake_outputs > real_outputs)  # 计算PIDS

    return fid, pids, uids  # 返回FID、PIDS和UIDS值

if __name__ == '__main__':
    # /home/daxu/下载/test_先验_unet_推理_mambaunet_500_200/results/test/out_img
    folder1 = r'E:\Pycharm\Palette_Windows\experiments\test_1000_400_s30\results\test\1'  # 第一个文件夹路径，包含生成的图像
    # /home/daxu/下载/test_mamba_seed0/test_mamba_seed_0/results/test/gt_img1
    # /media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/images_val_png
    folder2 = r'D:\Datasets\CelebA-HQ\celeba-hq\images_val_png'  # 第二个文件夹路径，包含真实的图像

    fid, pids, uids = calculate_metrics(folder1, folder2)  # 计算FID、PIDS和UIDS指标
    print('fid: %.4f, pids: %.4f, uids: %.4f' % (fid, pids, uids))  # 打印计算结果
    with open('fid_pids_uids.txt', 'w') as f:
        f.write('fid: %.4f, pids: %.4f, uids: %.4f' % (fid, pids, uids))  # 将结果写入文件
