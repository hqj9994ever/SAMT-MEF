import torch
import random
from math import log10
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import os
import torchvision
import glob


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


# recommend
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal(m.weight.data, mode='fan_out')
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_psnr_ssim(output, label):
    assert output.shape == label.shape
    output = np.clip(output.detach().cpu().numpy(), 0, 1)
    label = np.clip(label.detach().cpu().numpy(), 0, 1)
    output = output.transpose(0, 2, 3, 1)
    label = label.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(output.shape[0]):
        psnr += peak_signal_noise_ratio(label[i], output[i], data_range=1)
        ssim += structural_similarity(label[i], output[i], data_range=1, multichannel=True, channel_axis=-1)

    return psnr / output.shape[0], ssim / output.shape[0], output.shape[0]


def fusion(lowlight_image_path,overlight_image_path,image_list_path,result_list_path,DCE_net,size=256): 
	data_l = Image.open(lowlight_image_path)
	data_o = Image.open(overlight_image_path)
	data_l = data_l.resize((size,size), Image.LANCZOS)
	data_o = data_o.resize((size,size), Image.LANCZOS)
	data_l = (np.asarray(data_l)/255.0) 
	data_o = (np.asarray(data_o)/255.0)
     
	data_l = torch.from_numpy(data_l).float()
	data_o = torch.from_numpy(data_o).float()
     
	data_l = data_l.permute(2,0,1)
	data_o = data_o.permute(2,0,1)
	data_l = data_l.cuda().unsqueeze(0)
	data_o = data_o.cuda().unsqueeze(0)
	enhanced_image = torch.clamp(DCE_net(data_l, data_o)[0], 0, 1)
	
	image_path = lowlight_image_path.replace(image_list_path,result_list_path)
     
	image_path = image_path.replace('.JPG','.png')  #?
	output_path = image_path
	if not os.path.exists(output_path.replace('/'+image_path.split("/")[-1],'')): 
		os.makedirs(output_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, output_path)

def inference_fusion(lowlight_image_list_path,overlight_image_list_path,result_list_path,DCE_net,size=256):
    with torch.no_grad():
        filePath_low = lowlight_image_list_path
        filePath_over = overlight_image_list_path
        file_list_low = sorted(os.listdir(filePath_low))
        file_list_over = sorted(os.listdir(filePath_over))
        
        print("Inferencing...")
        for file_name_a, file_name_b in zip(file_list_low,file_list_over):
            lowlight_image = glob.glob(filePath_low+file_name_a)[0]
            overlight_image = glob.glob(filePath_over+file_name_b)[0]
            fusion(lowlight_image,overlight_image,lowlight_image_list_path,result_list_path,DCE_net,size)