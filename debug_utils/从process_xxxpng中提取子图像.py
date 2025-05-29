from PIL import Image
import os


def extract_subimages_with_gap(image_path, subimage_size=(256, 256), gap=2, save_dir="subimages"):
    # 打开大图像
    image = Image.open(image_path)

    # 获取大图像的尺寸
    width, height = image.size

    # 考虑边框的有效区域
    effective_width = width - 2  # 去除左右边框
    effective_height = height - 2  # 去除上下边框

    # 创建保存子图像的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 计算每个小图像的间隔后的占用空间
    subimage_with_gap_width = subimage_size[0] + gap  # 每个小图像加上间隔后的宽度
    subimage_with_gap_height = subimage_size[1] + gap  # 每个小图像加上间隔后的高度

    # 提取子图像
    subimages = []
    subimage_count = 0
    for top in range(2, effective_height, subimage_with_gap_height):  # 从第二行开始，跳过边框
        for left in range(2, effective_width, subimage_with_gap_width):  # 从第二列开始，跳过边框
            # 计算每个小图像的区域
            right = min(left + subimage_size[0], width - 2)  # 加边框后的右边界
            bottom = min(top + subimage_size[1], height - 2)  # 加边框后的下边界

            # 获取子图像并保存
            subimage = image.crop((left, top, right, bottom))
            subimage_filename = os.path.join(save_dir, f"subimage_{subimage_count}.png")
            subimage.save(subimage_filename)
            subimages.append(subimage_filename)
            subimage_count += 1

    print(f"提取完成，共提取了 {subimage_count} 个子图像。")
    return subimages


# 使用示例
image_path = "C:\\Users\Administrator\Desktop\paper_use\process\source\\500\\Process_162982.png"  # 输入你的大图像路径
save_dir="C:\\Users\Administrator\Desktop\paper_use\process\image\\500\Process_162982"
subimages = extract_subimages_with_gap(image_path, save_dir=save_dir)
