import os


# 定义自定义的字符串路径
custom_path = "E:\Dataset\Daxu\CelebA-HQ\celeba-hq\celeba-celeba256"

# 读取flist文件
with open(r"D:\All_User_Code\daxu\Palette-Image-to-Image-Diffusion-Models\valid.flist", "r") as file:
    lines = file.readlines()

# 处理每一行，构建新的路径
new_lines = []
for line in lines:
    # 去除行末的换行符
    line = line.strip()
    # 提取文件名
    filename = line.split('/')[-1]
    # 构建新路径
    new_path = os.path.join(custom_path, filename)
    # 添加到新的行列表中
    new_lines.append(new_path)

# 将新的路径写入新文件
with open("datasets/celebahq/flist/valid.flist", "w") as file:
    file.write("\n".join(new_lines))

print("处理完成！新文件已保存")
