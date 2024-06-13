import os
import torch
from glob import glob
from os.path import join
from torchvision.transforms import transforms

# initialize the reliable bank

input_dir = 'data/unlabeled/testA'
result_dir = 'data/unlabeled/candidate'
input_lists = glob(join(input_dir, '*.*'))
for gen_path in zip(input_lists):
    # img = torch.zeros((3,96,96))
    img = torch.zeros((3,192,192))
    img_name = gen_path[0].split('/')[3]
    print(img_name)
    toPil = transforms.ToPILImage()
    res = toPil(img).convert('RGB')
    res.save(os.path.join(result_dir, img_name))
