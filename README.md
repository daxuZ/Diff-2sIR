# Diff-2sIR
论文《Diff-2sIR: Diffusion-Based Refinement Two-Stage Image Restoration Model》的实施。
![](imgs/img1.png)

## 1. 环境准备

  - Ubuntu 22.04 LTS
  - Python 3.10

1) 克隆仓库
    ```shell
    git clone https://github.com/daxuZ/Diff-2sIR.git
    ```
    
2) 创建环境
   ```shell
   conda create -n diff_2sir python=3.10
   ```

   ```shell
   pip install -r requirements.txt
   ```

## 2. 文件准备

下载  [权重文件和CelebA-HQ](https://pan.baidu.com/s/1LTjlCdN7Gc64nn6n87KRhw?pwd=daxu) 数据集、[掩码](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA?e=7OwJyE) 数据集(来源于 [MAT](https://github.com/fenglinglwb/MAT) )

将权重文件放置到 ./experiments/checkpoint 目录下，准备测试使用的 .flist 文件
