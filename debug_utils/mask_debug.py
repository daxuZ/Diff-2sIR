import numpy as np
import torch
from data.util.mask import RandomMask, get_irregular_mask

mask = RandomMask(256, hole_range=[0, 1]).transpose(1, 2, 0)
mask1 = get_irregular_mask([256, 256])
print(mask1)
