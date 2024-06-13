import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
# my import
from models.Network import Network
from dataset import TestData
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bz = 1
model_root = 'model/ckpt/model_e200.pth'
input_root = 'data'
save_path = 'result'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root, phase='test')
data_load = data.DataLoader(Mydata_, batch_size=bz)
model = Network().cuda()
model = nn.DataParallel(model, device_ids=[0])
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
model.eval()
print('START!')
if 1:
    print('Load model successfully!')
    total_time = 0.0
    for data_ in data_load:
        begin = time.time()
        data_u, data_o, name = data_['under'], data_['over'], *data_['name']
        data_u = Variable(data_u).cuda()
        data_o = Variable(data_o).cuda()
        with torch.no_grad():
            result = model(data_u, data_o)
            temp_res1 = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res1[temp_res1 > 1] = 1
            temp_res1[temp_res1 < 0] = 0
            temp_res1 = (temp_res1*255).astype(np.uint8)
            temp_res1 = Image.fromarray(temp_res1)
            end = time.time()
            print("elapsed_time {:.5f}".format(end-begin))
            temp_res1.save('%s/%s' % (save_path, name))
            print('result saved!')
print('finished!')
