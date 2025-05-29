"""
1. Download model and save the model to git_root/model/celebahq/200_Network.pth
2. Modify inpainting_celebahq.json
    ["path"]["resume_state"]: "model/celebahq/200"
    ["datasets"]["test"]["args"]["data_root"]: "<Folder Constains Inference Images>"

    (optinally) change ["model"]["which_networks"]["args"]["beta_schedule"]["test"]["n_timestep"] value to reduce # steps inference should take
                more steps yields better results
3. Modify in your particular case in this code:
    model_pth = "<PATH-TO-MODEL>/200_Network.pth"
    input_image_pth = "<PATH-TO-DATASET_PARENT_DIT>/02323.jpg"
5. Run inpainting code (assume save this code to git_root/inference/inpainting.py)
    cd inference
    python inpainting.py -c ../config/inpainting_celebahq.json -p test
"""

import argparse

import core.praser as Praser
import torch
from core.util import set_device, tensor2img
from data.util.mask import get_irregular_mask
from PIL import Image
from torchvision import transforms

model_pth = "/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Models/experiments/checkpoint/200/200_Network.pth"
input_image_pth = "<PATH-TO-DATASET_PARENT_DIT>/02323.jpg"


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Models/config/inpainting_celebahq.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str,
                        choices=['train', 'test'], help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int,
                        default=16, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    args = parser.parse_args()
    opt = Praser.parse(args)
    return opt


# config arg
opt = parse_config()
if opt["model"]["which_networks"][0]["use_pred_v"]:
    from models.v_network import Network
else:
    from models.network import Network
model_args = opt["model"]["which_networks"][0]["args"]

# initializa model
model = Network(**model_args)
state_dict = torch.load(model_pth)
model.load_state_dict(state_dict, strict=False)
device = torch.device('cuda:0')
model.to(device)
model.set_new_noise_schedule(phase='test')
model.eval()

tfs = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# read input and create random mask
img_pillow = Image.open(input_image_pth).convert('RGB')
img = tfs(img_pillow)
mask = get_irregular_mask([256, 256])
mask = torch.from_numpy(mask).permute(2, 0, 1)
cond_image = img*(1. - mask) + mask*torch.randn_like(img)
mask_img = img*(1. - mask) + mask

# save conditional image used a inference input
cond_image_np = tensor2img(cond_image)
Image.fromarray(cond_image_np).save("./result/cond_image.jpg")

# set device
cond_image = set_device(cond_image)
gt_image = set_device(img)
mask = set_device(mask)

# unsqueeze
cond_image = cond_image.unsqueeze(0).to(device)
gt_image = gt_image.unsqueeze(0).to(device)
mask = mask.unsqueeze(0).to(device)

# inference
with torch.no_grad():
    output, visuals = model.restoration(cond_image, y_t=cond_image,
                                        y_0=gt_image, mask=mask, sample_num=8)

# save intermediate processes
output_img = output.detach().float().cpu()
for i in range(visuals.shape[0]):
    img = tensor2img(visuals[i].detach().float().cpu())
    Image.fromarray(img).save(f"./result/process_{i}.jpg")

# save output (output should be the same as last process_{i}.jpg)
img = tensor2img(output_img)
Image.fromarray(img).save("./result/output.jpg")
