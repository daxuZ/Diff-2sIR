import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def calculate_psnr_ssim(original_dir, restored_dir, output_file):
    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.jpg')])
    restored_files = sorted([f for f in os.listdir(restored_dir) if f.endswith('.jpg')])

    psnr_list = []
    ssim_list = []

    with open(output_file, 'w') as f:
        for orig_file, rest_file in zip(original_files, restored_files):
            orig_path = os.path.join(original_dir, orig_file)
            rest_path = os.path.join(restored_dir, rest_file)

            original_image = cv2.imread(orig_path)
            restored_image = cv2.imread(rest_path)

            if original_image is None or restored_image is None:
                print(f"Could not read image pair: {orig_file}, {rest_file}")
                continue

            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            restored_image = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)

            psnr = compare_psnr(original_image, restored_image)
            ssim = compare_ssim(original_image, restored_image, channel_axis=2, multichannel=True)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            result = f"{orig_file} - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}\n"
            f.write(result)

    average_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    average_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0

    with open(output_file, 'a') as f:
        f.write(f"Average PSNR: {average_psnr:.4f}\n")
        f.write(f"Average SSIM: {average_ssim:.4f}\n")


if __name__ == "__main__":
    original_dir = "/home/daxu/下载/test_双mamba推理_500_200/results/test/gt_img1"  # 替换为原始图像的目录
    restored_dir = "/home/daxu/下载/test_双mamba推理_500_200/results/test/out_img"  # 替换为修复后图像的目录
    output_file = "psnr_ssim/test_双mamba推理_500_200_psnr_ssim_results.txt"  # 保存结果的文件名

    calculate_psnr_ssim(original_dir, restored_dir, output_file)
