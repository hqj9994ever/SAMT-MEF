import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.models



class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

    def forward(self, xs, ys):
        L2_temp = 0.2 * self.L2(xs, ys)
        L1_temp = self.L1(xs, ys)
        L_total = L1_temp + L2_temp
        return L_total, L2_temp, L1_temp


class GradientLoss(nn.Module):
    # 梯度loss
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.L1loss = nn.L1Loss()

    def forward(self, x, s1):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        grad_f = torch.abs(sobelx) + torch.abs(sobely)

        sobelx_s1 = F.conv2d(s1, self.weightx, padding=1)
        sobely_s1 = F.conv2d(s1, self.weighty, padding=1)
        grad_s1 = torch.abs(sobelx_s1) + torch.abs(sobely_s1)

        loss = self.L1loss(grad_f, grad_s1)
        return loss


class FeatureHook:
    # 给感知loss用的
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()



class PerceptualLoss(nn.Module):
    def __init__(self, blocks=[0, 1, 2, 3]):
        super().__init__()
        self.feature_loss = nn.MSELoss()

        # VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        vgg = torchvision.models.vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        B, C, H, W = inputs.shape
        if C == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)
        inputs = F.normalize(inputs, dim=1)
        targets = F.normalize(targets, dim=1)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0

        for lhs, rhs in zip(input_features, target_features):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs)

        return loss


def Kaiming_mask(x):

    B, C, H, W = x.size()
    mask = torch.zeros((B, C, H, W)).cuda()
    assert H == W
    t = int(H / 4)  # 区间大小
    x = int(H / t)  # 区间个数
    y = int(W / t)
    for c in range(C):
        for i in range(x):
            for j in range(y):
                a = random.randint(0, 100)
                if a >= 50:
                    mask[:, c, i*t:i*t+t, j*t:j*t+t] = 1.0
    masked_x = x * mask

    return masked_x


class ContrastiveLeaning(nn.Module):
    def __init__(self, blocks=[0, 1, 2, 3, 4]):
        super().__init__()
        self.feature_loss = nn.MSELoss()

        # VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end

        vgg = torchvision.models.vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, x, targets, s1, s2):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        x = F.normalize(x, dim=1)
        targets = F.normalize(targets, dim=1)
        s1 = F.normalize(s1, dim=1)
        s2 = F.normalize(s2, dim=1)

        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)
            s1 = s1.repeat(1, 3, 1, 1)
            s2 = s2.repeat(1, 3, 1, 1)

        # 架设mask
        x = Kaiming_mask(x)
        targets = Kaiming_mask(targets)
        s1 = Kaiming_mask(s1)
        s2 = Kaiming_mask(s2)

        # extract feature maps
        self.features(x.float())
        x_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets.float())
        target_features = [hook.features for hook in self.hooks]

        self.features(s1.float())
        s1_features = [hook.features for hook in self.hooks]

        self.features(s2.float())
        s2_features = [hook.features for hook in self.hooks]

        loss = 0.0

        # compare their weighted loss
        for f, g, o, u in zip(x_features, target_features, s1_features, s2_features):
            f = f.view(f.size(0), -1)
            g = g.view(g.size(0), -1)
            o = o.view(o.size(0), -1)
            u = u.view(u.size(0), -1)

            N = self.feature_loss(f, g)
            D = self.feature_loss(f, o) + self.feature_loss(f, u) + 0.003
            loss = loss + (N / D)
            loss.requires_grad_(True)

        return loss
    

class ColorLoss(nn.Module):
    # 低曝曝光还原色彩Loss
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x, gt):
        img1 = torch.reshape(x, [3, -1])
        img2 = torch.reshape(gt, [3, -1])
        clip_value = 0.999999
        norm_vec1 = torch.nn.functional.normalize(img1, p=2, dim=0)
        norm_vec2 = torch.nn.functional.normalize(img2, p=2, dim=0)
        temp = norm_vec1 * norm_vec2
        dot = temp.sum(dim=0)
        dot = torch.clamp(dot, -clip_value, clip_value)
        angle = torch.acos(dot) * (180 / math.pi)
        return 0.1*torch.mean(angle)
    


