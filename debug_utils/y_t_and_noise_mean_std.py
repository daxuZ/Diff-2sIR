import numpy as np
from inspect import isfunction
import torch
import cv2
from torchvision import transforms
from PIL import Image

def q_sample(y_0, sample_gammas, noise=None):
    # y_t
    noise = default(noise, lambda: torch.randn_like(y_0))
    # noise = noise
    return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
    )


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(x):
    return x is not None

tfs = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

betas = torch.linspace(1e-6, 1e-2, 2000)  # 1e-6 1e-2
alphas = 1. - betas
gammas = torch.cumprod(alphas, axis=0)
sample_gammas = gammas[1999]
y_0 = Image.open("/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Models/experiments/test_inpainting_celebahq_240309_174622/results/test/0/GT_006622.jpg").convert('RGB')

y_0_tf = tfs(y_0)
noise = torch.randn_like(y_0_tf)
y_t = q_sample(y_0_tf, sample_gammas, noise=noise)
mean_n = torch.mean(noise)
std_n = torch.std(noise)
mean = torch.mean(y_t)  # tensor(0.1091) tensor(0.5091)/tensor(-5.5575e-05) tensor(1.0007)
std = torch.std(y_t)  # tensor(-0.0003) tensor(1.0009)
print(mean, std)
print(mean_n, std_n)
print(gammas[1]-gammas[0], gammas[2]-gammas[1], gammas[3]-gammas[2], gammas[4]-gammas[3])