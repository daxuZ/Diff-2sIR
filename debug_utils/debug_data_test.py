import matplotlib.pyplot as plt
import data.util.mask as mk
from PIL import Image
import torch
from torchvision import transforms

img_path = "D:\\Users\Administrator\Pycharm\Palette-Image-to-Image-Diffusion-Models\misc\image\Out_Places365_test_00143399.jpg"


def pil_loader(path):
    return Image.open(path).convert('RGB')


def get_mask():
    random_box = mk.random_bbox()
    mask = mk.bbox2mask((256, 256), random_box)
    return torch.from_numpy(mask).permute(2,0,1)


def tfs(img):
    tfs = transforms.Compose([
          transforms.Resize((256, 256)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    img = tfs(img)
    return img

img = pil_loader(img_path)
image = tfs(img)
mask = get_mask()
cond_image = image * (1. - mask) + mask * torch.randn_like(image)
mask_img = image * (1. - mask) + mask
mask_img1 = image * (1. - mask)

image_np = image.permute(1, 2, 0).numpy()
cond_image_np = cond_image.permute(1, 2, 0).numpy()
mask_img_np = mask_img.permute(1, 2, 0).numpy()
mask_img1_np = mask_img1.permute(1, 2, 0).numpy()

plt.imshow(mask.squeeze(), cmap='gray')
plt.show()
plt.imshow(cond_image_np)
plt.show()
plt.imshow(mask_img_np)
plt.show()

plt.imshow(mask_img1_np)
plt.show()
