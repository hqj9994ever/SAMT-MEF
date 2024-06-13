import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
from torchvision import transforms
import torch.distributed as dist
from torch.optim import lr_scheduler
import PIL.Image as Image
from utils import *
from torch.autograd import Variable
from loss.losses import *
from models.Network import GetGradientNopadding, latent2im
import matplotlib.pyplot as plt
from kornia.losses import SSIMLoss, PSNRLoss
from loss.losses import *
import pyiqa
import cv2

class Trainer:
    def __init__(self, model, tmodel, args, supervised_loader, unsupervised_loader, val_loader, iter_per_epoch):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.model = model
        self.tmodel = tmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.num_epochs
        self.save_period = 20
        self.loss_unsup = MyLoss().cuda()  # nn.L1Loss()
        self.loss_str = MyLoss().cuda()
        self.loss_grad = nn.L1Loss().cuda()
        self.loss_cr = ContrastiveLeaning().cuda()
        self.loss_psnr = PSNRLoss(max_val=1.0)
        self.loss_ssim = SSIMLoss(window_size=7).cuda()
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.loss_per = PerceptualLoss().cuda()
        self.curiter = 0
        self.model.cuda()
        self.tmodel.cuda()
        self.device, available_gpus = self._get_available_devices(self.args.gpus)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
        self.toPIL = transforms.ToPILImage()
        # set optimizer and learning rate
        self.optimizer_s = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[self.epochs/3, self.epochs/3*2], gamma=0.1)

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.5):
        # exponential moving average(EMA)
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image_u, image_o):
        with torch.no_grad():
            predict_target_ul = self.tmodel(image_u, image_o)

        return predict_target_ul

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    def get_reliable(self, teacher_predict, student_predict, positive_list, p_name):
        N = teacher_predict.shape[0]
        score_t_ag, score_t_sd = self.get_score(teacher_predict)
        score_s_ag, score_s_sd = self.get_score(student_predict)
        score_r_ag, score_r_sd = self.get_score(positive_list)
        positive_sample = positive_list.clone()
        for idx in range(0, N):
            if score_t_ag[idx] > score_s_ag[idx] and score_t_sd[idx] > score_s_sd[idx]:
                if score_t_ag[idx] > score_r_ag[idx] and score_t_sd[idx] > score_r_sd[idx]:
                    positive_sample[idx] = teacher_predict[idx]
                    # update the reliable bank
                    temp_c = np.transpose(teacher_predict[idx].detach().cpu().numpy(), (1, 2, 0))
                    temp_c = np.clip(temp_c, 0, 1)
                    arr_c = (temp_c*255).astype(np.uint8)
                    arr_c = Image.fromarray(arr_c)
                    arr_c.save('%s' % p_name[idx])
        del N, score_r_ag, score_s_ag, score_t_ag, score_r_sd, score_s_sd, score_t_sd, teacher_predict, student_predict, positive_list
        return positive_sample

    def train(self):
        self.freeze_teachers_parameters()
        if self.start_epoch == 1:
            initialize_weights(self.model)
        else:
            checkpoint = torch.load(self.args.resume_path)
            self.model.load_state_dict(checkpoint['state_dict'])
        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_ave, psnr_train = self._train_epoch(epoch)
            loss_val = loss_ave.item() / self.args.crop_size * self.args.train_batchsize
            train_psnr = sum(psnr_train) / len(psnr_train)
            psnr_val = self._valid_epoch(max(0, epoch))
            val_psnr = sum(psnr_val) / len(psnr_val)


            print('[%d] main_loss: %.6f, train psnr: %.6f, val psnr: %.6f, lr: %.8f' % (
                epoch, loss_val, train_psnr, val_psnr, self.lr_scheduler_s.get_last_lr()[0]))

            # Save checkpoint
            if epoch % self.save_period == 0 and self.args.local_rank <= 0:
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict()}
                ckpt_name = str(self.args.save_path) + 'model_e{}.pth'.format(str(epoch))
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)
# 消cr，消可信赖库，消可信赖库和cr，消无标签数据

    def _train_epoch(self, epoch):
        sup_loss = AverageMeter()
        unsup_loss = AverageMeter()
        loss_total_ave = 0.0
        psnr_train = []
        self.model.train()
        self.freeze_teachers_parameters()
        train_loader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        tbar = range(len(self.unsupervised_loader))
        tbar = tqdm(tbar, ncols=130, leave=True)
        for i in tbar:
            (img_data_u, img_data_o, label), (unpaired_data_w_u, unpaired_data_w_o,
                                              unpaired_data_s_u, unpaired_data_s_o, p_list, p_name) = next(train_loader)
            img_data_u = Variable(img_data_u).cuda(non_blocking=True)
            img_data_o = Variable(img_data_o).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            unpaired_data_s_u = Variable(unpaired_data_s_u).cuda(non_blocking=True)
            unpaired_data_s_o = Variable(unpaired_data_s_o).cuda(non_blocking=True)
            unpaired_data_w_u = Variable(unpaired_data_w_u).cuda(non_blocking=True)
            unpaired_data_w_o = Variable(unpaired_data_w_o).cuda(non_blocking=True)
            p_list = Variable(p_list).cuda(non_blocking=True)
            # teacher output
            predict_target_u = self.predict_with_out_grad(unpaired_data_w_u, unpaired_data_w_o)
            N = predict_target_u.shape[0]
            origin_predict = predict_target_u.detach().clone()
            # student output
            outputs_l = self.model(img_data_u, img_data_o)
            outputs_ul = self.model(unpaired_data_s_u, unpaired_data_s_o)
            structure_loss, l2_loss, l1_loss = self.loss_str(outputs_l, label)
            perceptual_loss = self.loss_per(outputs_l, label)
            get_grad = GetGradientNopadding().cuda()
            gradient_loss = self.loss_grad(get_grad(outputs_l), get_grad(label))
            ssim_loss = self.loss_ssim(outputs_l, label)
            loss_sup = 1.2*structure_loss + 0.3 * perceptual_loss + 0.1 * gradient_loss + 0.8*ssim_loss
            sup_loss.update(loss_sup.mean().item())
            p_sample = self.get_reliable(predict_target_u, outputs_ul, p_list, p_name)
            unsu_str_loss, unsu_loss_l1, unsu_loss_l2 = self.loss_unsup(outputs_ul, p_sample)
            cr_loss = self.loss_cr(outputs_ul, p_sample, unpaired_data_s_u, unpaired_data_s_o)
            loss_unsu = unsu_str_loss + 0.1 * cr_loss
            unsup_loss.update(loss_unsu.mean().item())
            consistency_weight = self.get_current_consistency_weight(epoch)
            total_loss = loss_sup + consistency_weight * loss_unsu
            print("loss_unsup: %.5f  loss_sup: %.5f" % (consistency_weight * loss_unsu, loss_sup))
            total_loss = total_loss.mean()
            psnr_train.extend(to_psnr(outputs_l, label))
            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            tbar.set_description('Train-Student Epoch {} | Ls {:.4f} Lu {:.4f}|'
                                 .format(epoch, sup_loss.avg, unsup_loss.avg))

            del img_data_u, img_data_o, label, unpaired_data_w_u, unpaired_data_w_o, unpaired_data_s_o, unpaired_data_s_u
            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                self.curiter = self.curiter + 1

        loss_total_ave = loss_total_ave + total_loss

        self.lr_scheduler_s.step(epoch=epoch - 1)
        return loss_total_ave, psnr_train

    def _valid_epoch(self, epoch):
        psnr_val = []
        self.model.eval()
        self.tmodel.eval()
        val_psnr = AverageMeter()
        val_ssim = AverageMeter()
        total_loss_val = AverageMeter()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for i, (val_data_u, val_data_o, val_label) in enumerate(tbar):
                val_data_u = Variable(val_data_u).cuda()
                val_data_o = Variable(val_data_o).cuda()
                val_label = Variable(val_label).cuda()
                # forward
                val_output = self.model(val_data_u, val_data_o)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(val_output, val_label)
                val_psnr.update(temp_psnr, N)
                val_ssim.update(temp_ssim, N)
                psnr_val.extend(to_psnr(val_output, val_label))
                tbar.set_description('{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f}|'.format(
                    "Eval-Student", epoch, val_psnr.avg, val_ssim.avg))

            del val_output, val_label, val_data_u, val_data_o
            return psnr_val

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_score(self, p_list):
        N = p_list.shape[0]
        score_ag = []
        score_sd = []
        for i in range(N):
            img = latent2im(p_list[i])
            img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            score_ag.append(self.avgGradient(img_grey))
            mean, std = cv2.meanStdDev(img_grey)
            score_sd.append(std)
        score_ag = np.array(score_ag)
        score_sd = np.array(score_sd)
        return score_ag, score_sd
    
    def avgGradient(self, image):
        width = image.shape[1]
        width = width - 1
        heigt = image.shape[0]
        heigt = heigt - 1
        tmp = 0.0

        for i in range(width):
            for j in range(heigt):
                dx = float(image[i, j+1])-float(image[i, j])
                dy = float(image[i+1, j])-float(image[i, j])
                ds = math.sqrt((dx*dx+dy*dy)/2)
                tmp += ds
        imageAG = tmp/(width*heigt)
        return imageAG
