from PIL import Image
import os


def resize_and_save(input_dir, output_dir, target_size=(512, 512)):
    """
    将256x256图像调整为512x512，然后保存到另一个目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像路径
    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if
                   fname.endswith('.png') or fname.endswith('.jpg')]
    for img_path in image_paths:
        img = Image.open(img_path)

        # 将图像从256x256放大到512x512
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)  # 使用高质量的插值

        # 保存调整后的图像
        img_name = os.path.basename(img_path)
        img_resized.save(os.path.join(output_dir, img_name))

    print(f"所有图像已保存到 {output_dir}。")


def process_model_output(output_dir, final_output_dir, target_size=(256, 256)):
    """
    将512x512的修复结果图像调整为256x256
    """
    os.makedirs(final_output_dir, exist_ok=True)

    # 获取模型输出的图像路径
    image_paths = [os.path.join(output_dir, fname) for fname in os.listdir(output_dir) if
                   fname.endswith('.png') or fname.endswith('.jpg')]
    for img_path in image_paths:
        img = Image.open(img_path)

        # 将512x512的图像缩小到256x256
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)  # 使用平滑的插值方法

        # 保存缩小后的图像
        img_name = os.path.basename(img_path)
        img_resized.save(os.path.join(final_output_dir, img_name))

    print(f"所有图像已缩小并保存到 {final_output_dir}。")


# 输入和输出目录设置
input_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/images_val_png"  # 存放256x256图像的目录
output_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/images_val_png_to_512"  # 用于存放调整后的512x512图像
final_output_dir = "/path/to/final_output"  # 存放模型修复后的256x256图像

# 调整图像为512x512
resize_and_save(input_dir, output_dir, target_size=(512, 512))

# 这里调用模型进行修复（假设修复结果存放在output_dir）

# 然后调整修复后的图像为256x256
# process_model_output(output_dir, final_output_dir, target_size=(256, 256))
