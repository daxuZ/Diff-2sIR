import numpy as np
import sklearn.svm
import os
import sys
import torch
import uuid
import hashlib
import dnnlib.util
import metric_utils

# 将项目根目录添加到系统路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 将 dnnlib 目录添加到系统路径中
dnnlib_path = os.path.abspath(os.path.join(project_root, 'dnnlib'))
sys.path.append(dnnlib_path)

class Options:
    def __init__(self):
        self.dataset_path = '/home/daxu/Desktop/gt_img'
        self.generator_path = '/home/daxu/Desktop/out_img'
        self.rank = 0
        self.num_gpus = 1
        self.device = 'cuda'
        self.cache = True
        self.progress = None
        self.dataset_kwargs = {
            'class_name': 'data.dataset.Dataset',
            'data_dir': '/path/to/data'
        }

opts = Options()

def compute_ids(opts, max_real, num_gen):
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True)

    real_activations = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all()

    fake_activations = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all()

    if opts.rank != 0:
        return float('nan')

    svm = sklearn.svm.LinearSVC(dual=False)
    svm_inputs = np.concatenate([real_activations, fake_activations])
    svm_targets = np.array([1] * real_activations.shape[0] + [0] * fake_activations.shape[0])
    print('Fitting ...')
    svm.fit(svm_inputs, svm_targets)
    u_ids = 1 - svm.score(svm_inputs, svm_targets)
    real_outputs = svm.decision_function(real_activations)
    fake_outputs = svm.decision_function(fake_activations)
    p_ids = np.mean(fake_outputs > real_outputs)

    return float(u_ids), float(p_ids)

# 设置用于计算的图像数量
max_real = 1000
num_gen = 1000

# 计算IDS指标
u_ids, p_ids = compute_ids(opts, max_real, num_gen)
print(f"Unmatched IDS: {u_ids}, Probability IDS: {p_ids}")
