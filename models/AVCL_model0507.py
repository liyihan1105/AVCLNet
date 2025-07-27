from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from tools import ops
import cv2
import math
import torchvision.models as models
from torchvision.models import resnet50

class KDNet(nn.Module):
    def __init__(self, net_path_vi=None, net_path_au=None, net_path_au_student=None):
        super(KDNet, self).__init__()

        # 视觉教师模型 net1
        self.net1 = Net(
            backbone=AlexNetV1(),
            head=SiamFC())
        ops.init_weights(self.net1)
        if net_path_vi is not None:
            self.net1.load_state_dict(torch.load(
                net_path_vi, map_location=lambda storage, loc: storage))  # 模型是CPU，预加载的训练参数是GPU

        # 视觉学生模型 net3
        self.net3 = ModifiedResNet50()  # 使用预训练的 ResNet50 模型

        # SiamRPNNetEncoder
        self.net_encoder_vi = teacherviEnocder(backbone=AlexNetV1())

        # 教师听觉模型 net2
        self.net2 = audioNet(
            net_head=AlexNetV1_au(),
            predNet=GCFpredictor())
        ops.init_weights(self.net2)
        if net_path_au is not None:
            self.net2.load_state_dict(torch.load(
                net_path_au, map_location=lambda storage, loc: storage)['model'])

        # 学生听觉模型 net4
        self.net4 = audiostudentNet()
        ops.init_weights(self.net4)
        if net_path_au_student is not None:
            self.net4.load_state_dict(torch.load(net_path_au_student))

        # AlexNetV1_auEncoder
        self.net_encoder_au = AlexNetV1_au()

        self.netMHA = MultiHeadAttention()  # 初始化多头注意力机制模块
        ops.init_weights(self.netMHA)
        self.predNet = Predictor()
        ops.init_weights(self.predNet)
        # 使用了 LayerNorm 层来对不同的特征进行归一化处理
        self.LN1_vi = nn.LayerNorm([256, 35, 35], eps=1e-6)
        self.LN1_au = nn.LayerNorm([256, 35, 35], eps=1e-6)  # 表示视觉和听觉特征的通道数为256，特征图的大小为35x35
        self.LN_vi = nn.LayerNorm([1225, 256], eps=1e-6)
        self.LN_au = nn.LayerNorm([1225, 256], eps=1e-6)  # 特征图的像素数量为1225，通道数为256
        self.LN_av = nn.LayerNorm([1225, 256], eps=1e-6)  # 对视觉和听觉特征的融合结果进行归一化处理


    def forward(self, ref, img0, img1, img2, auFr):
        # 视觉教师模型 net1
        Fvi_teacher_encoder = self.net_encoder_vi(ref, img0, img1, img2)
        # print("Shape of teacher_vi_output:", Fvi_teacher_encoder.shape)
        Fvi_teacher_encoder = self.LN1_vi(Fvi_teacher_encoder)

        # 视觉学生模型 net3
        Fvi_student = self.net3(ref, img0, img1, img2)
        # print("Shape of student_vi_output:", Fvi_student.shape)
        Fvi_student = self.LN1_vi(Fvi_student)

        # 听觉教师模型
        Fau_teacher_encoder = self.net_encoder_au(auFr)
        # print("Shape of teacher_au_output:", Fau_teacher_encoder.shape)
        Fau_teacher_encoder = self.LN1_au(Fau_teacher_encoder)

        # 听觉学生模型
        Fau_student = self.net4(auFr)
        # print("Shape of student_au_output:", Fau_student.shape)
        Fau_student = self.LN1_au(Fau_student)

        # 计算视觉教师和学生之间的特征对损失MSE损失
        vision_feature_loss = F.mse_loss(Fvi_student, Fvi_teacher_encoder.detach())

        # 计算听觉教师和学生之间的特征对损失MSE损失
        audio_feature_loss = F.mse_loss(Fau_student, Fau_teacher_encoder.detach())

        # 融合对齐后的学生视觉和听觉
        b, c = Fvi_teacher_encoder.shape[0], Fvi_teacher_encoder.shape[1]
        Fvi_student = Fvi_student.permute(0, 2, 3, 1).view(b, -1, c)  # [b,N=h*w,c]
        Fau_student = Fau_student.permute(0, 2, 3, 1).view(b, -1, c)  # [b,N=h*w,c]
        # print("Shape of student_vi_output:", Fvi_student.shape)
        # print("Shape of student_au_output:", Fau_student.shape)
        Fvi_student = self.LN_vi(Fvi_student)
        Fau_student = self.LN_au(Fau_student)
        out_vi = self.netMHA(q=Fau_student, k=Fvi_student, v=Fvi_student)
        out_au = self.netMHA(q=Fvi_student, k=Fau_student, v=Fau_student)
        out_av = out_vi + out_au
        out_av = self.LN_av(out_av)
        # print("Shape of out_av_output:", out_av.shape)

        out_pred = self.predNet(out_av)
        out_pred = torch.squeeze(out_pred)
        # print("Shape of out_pred_output:", out_pred.shape)

        # 视觉教师模型 net1
        Fvi_teacher = self.net1(ref, img0, img1, img2)
        Fvi_teacher = Fvi_teacher.permute(0, 2, 3, 1).view(b, -1, c)
        # print("Shape of Fvi_teacher_output:", Fvi_teacher.shape)
        Fvi_teacher = self.LN_vi(Fvi_teacher)

        # 听觉教师模型
        Fau_teacher = self.net2(auFr).permute(0, 3, 1, 2)
        Fau_teacher = Fau_teacher.permute(0, 2, 3, 1).view(b, -1, c)
        # print("Shape of Fvi_teacher_output:", Fvi_teacher.shape)
        Fau_teacher = self.LN_au(Fau_teacher)

        # 执行知识蒸馏L1损失
        vision_distillation_loss = F.l1_loss(out_av, Fvi_teacher)
        audio_distillation_loss = F.l1_loss(out_av, Fau_teacher)

        total_loss = vision_feature_loss + audio_feature_loss + vision_distillation_loss + audio_distillation_loss

        return out_pred, total_loss

class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的全局平均池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((35, 35))  # 使用自适应平均池化层，将输出大小设置为 35x35
        self.conv = nn.Conv2d(2048, 256, kernel_size=1)  # 添加一个卷积层来获得大小为 256x35x35 的特征图

    def forward(self, z, x0, x1, x2):
        z = self.features(z)
        z = self.avgpool(z)
        z = self.conv(z)

        x0 = self.features(x0)
        x0 = self.avgpool(x0)
        x0 = self.conv(x0)

        x1 = self.features(x1)
        x1 = self.avgpool(x1)
        x1 = self.conv(x1)

        x2 = self.features(x2)
        x2 = self.avgpool(x2)
        x2 = self.conv(x2)

        return x0+x1+x2


class teacherviEnocder(nn.Module):  # 在上面stnet中定义
    def __init__(self, backbone):
        super(teacherviEnocder, self).__init__()
        self.backbone = backbone

    def updownsample(self, x):
        return F.interpolate(x,size=(35,35),mode='bilinear',align_corners=False)  # 将图片的大小改为35*35

    def forward(self, z, x0, x1, x2):
        z = self.backbone(z)

        fx0 = self.backbone(x0)
        fx1 = self.backbone(x1)
        fx2 = self.backbone(x2)

        n0 = self.updownsample(fx0)
        n1 = self.updownsample(fx1)
        n2 = self.updownsample(fx2)

        return n0+n1+n2
class _AlexNet(nn.Module):  # 视觉backbones

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class AlexNetV1(_AlexNet):  # 视觉backbones
    output_stride = 8
    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))




class Net(nn.Module):  # 在上面stnet中定义
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def updownsample(self, x):
        return F.interpolate(x,size=(35,35),mode='bilinear',align_corners=False)  # 将图片的大小改为35*35

    def forward(self, z, x0, x1, x2):
        z = self.backbone(z)

        fx0 = self.backbone(x0)
        fx1 = self.backbone(x1)
        fx2 = self.backbone(x2)

        h0 = self.head(z, fx0)
        h1 = self.head(z, fx1)
        h2 = self.head(z, fx2)

        n0 = self.updownsample(h0)
        n1 = self.updownsample(h1)
        n2 = self.updownsample(h2)

        return n0+n1+n2

class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        z0 = z[0]
        x0 = x[0]
        out = F.conv2d(x0.unsqueeze(0), z0.unsqueeze(1), groups=c)

        for i in range(1, nz):
            zi = z[i]
            xi = x[i]
            outi = F.conv2d(xi.unsqueeze(0), zi.unsqueeze(1), groups=c)
            out = torch.cat([out, outi], dim=0)

        return out


class _AlexNet(nn.Module):  # 视觉backbones

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class AlexNetV1(_AlexNet):  # 视觉backbones
    output_stride = 8
    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))



class audiostudentNet(nn.Module):
    def __init__(self):
        super(audiostudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 50 * 50, 512)  # 根据输入输出的维度进行调整
        self.fc2 = nn.Linear(512, 256 * 35 * 35)  # 输出维度调整为 256 x 35 x 35
        self.relu = nn.ReLU()

    def forward(self, x):  # [16, 3, 400, 400]
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # [16, 64, 200, 200]
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # [16, 128, 100, 100]
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # [16, 256, 50, 50]
        x = x.view(x.size(0), -1)  # 将特征展平成一维向量，自动计算特征维度大小
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.view(x.size(0), 256, 35, 35)  # 重新reshape成 [batch_size, channels, height, width] 的形状
        return x

class audioNet(nn.Module):  # 在上面stnet中定义
    def __init__(self,net_head, predNet):
        super(audioNet, self).__init__()
        self.net_head = net_head
        self.predNet = predNet

    def forward(self, x):
        x = self.net_head(x).permute(0, 2, 3, 1)  #[b,h,w,c]
        x = self.predNet(x)
        return x

class GCFpredictor(nn.Module):  # 具体network在GCFnet中介绍
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        return x
class _AlexNet_au(nn.Module):  # GCFnet中定义

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x
class _BatchNorm2d(nn.BatchNorm2d):  # GCFnet中定义

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)
class AlexNetV1_au(_AlexNet_au):  # GCFnet中定义
    output_stride = 8
    def __init__(self):
        super(AlexNetV1_au, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 6, 1, groups=2))


class Predictor(nn.Module):  # 将归一化处理后的特征再次输入预测

    def __init__(self):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=(1225, 1), stride=1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU())
        self.fc3 = nn.Linear(256, 2, bias=False)

    def forward(self, x):
        x = self.maxPool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):  # 两个参数 temperature 和 attn_dropout，分别用于指定缩放因子和注意力层中的 dropout 比例
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output

class MultiHeadAttention(nn.Module):  # 将视觉和听觉的特征融合

    def __init__(self, n_head=8, d_model=256, d_k=32, d_v=32, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = k

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q
