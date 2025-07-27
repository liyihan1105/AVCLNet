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
from torchvision.models import ResNet50_Weights
import yoloface.face_detector as YoloDetector


class AVCLNet(nn.Module):
    def __init__(self, yoloface_weights_path=None, yoloface_config_path=None, net_path_au=None):
        super(AVCLNet, self).__init__()

        # 视觉检测模块：加载预训练YOLOFace模型
        self.net1 = YoloDetector(weights_name=yoloface_weights_path,
                                 config_name=yoloface_config_path,
                                 device='cuda:0')
        # 初始化 YOLOFace 模型权重
        ops.init_weights(self.net1)
        if yoloface_weights_path is not None:
            self.net1.load_state_dict(
                torch.load(yoloface_weights_path, map_location='cpu')
            )

        # 声源定位
        self.net2 = audioNet(
            net_head=AlexNetV1_au(),
            predNet=GCFpredictor())
        ops.init_weights(self.net2)
        if net_path_au is not None:
            self.net2.load_state_dict(torch.load(
                net_path_au, map_location=lambda storage, loc: storage)['model'])

        # ResNet50 编码器
        self.visual_encoder = VisualEncoder(num_classes=256)
        self.audio_encoder = PANNsEncoder(feature_dim=256)

        # 冻结声源定位模块的参数
        for param in self.net2.parameters():
            param.requires_grad = False

        # 视听跨膜态对比学习模块
        self.av_criterion = ContrastiveLoss()
        # 视觉模态内对比学习模块
        self.vi_criterion = VisualContrastiveLoss()

        self.predNet = Predictor()
        ops.init_weights(self.predNet)

    def forward(self, img, auFr, visual_ids, anchor_idx):
        # YOLOFace 提取 bboxes
        bboxes, _ = self.net1(img)
        heatmap = generate_heatmap(bboxes, img_size=(img.shape[2], img.shape[3]), num_classes=1)
        visual_feat = self.visual_encoder(heatmap)

        # stGCF 提取方向图
        gcf_map = self.net2(auFr)
        audio_feat = self.audio_encoder(gcf_map)

        loss1 = self.av_criterion(visual_feat, audio_feat)
        loss2 = self.vi_criterion(visual_feat, visual_ids, anchor_idx)
        visual_feat_optimized = F.normalize(visual_feat, dim=1)

        loss=loss1 + loss2

        output = self.predNet(visual_feat_optimized)
        return output, loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, visual_feat, audio_feat):
        """
        输入: visual_feat, audio_feat [batch_size, feature_dim]
        输出: 对比学习的损失值
        """
        batch_size = visual_feat.size(0)

        # L2 标准化
        visual_feat = F.normalize(visual_feat, dim=1)
        audio_feat = F.normalize(audio_feat, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(visual_feat, audio_feat.T) / self.temperature

        # 构造对比学习目标
        labels = torch.arange(batch_size).cuda()
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

class VisualContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(VisualContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, visual_feat, visual_ids, anchor_idx):
        """
        输入: visual_feat [batch_size, feature_dim]
               visual_ids [batch_size], 物体ID
               anchor_idx [batch_size], 锚点样本的索引
        输出: 对比学习的损失值
        """
        batch_size = visual_feat.size(0)

        # L2 标准化
        visual_feat = F.normalize(visual_feat, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(visual_feat, visual_feat.T) / self.temperature

        # 构造正负样本对的标签
        labels = self.construct_visual_labels(visual_ids, anchor_idx, batch_size)

        # 计算对比损失
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def construct_visual_labels(self, visual_ids, anchor_idx, batch_size):
        """
        基于物体ID构建视觉正负样本对的标签。
        正样本: 锚点目标的前后帧内ID相同的样本。
        负样本: 锚点目标与其他目标的样本。
        """
        labels = torch.zeros(batch_size).long().cuda()  # 默认全为负样本（标签为0）

        # 遍历每个样本
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # 锚点目标为 anchor_idx，正样本：ID相同，负样本：ID不同
                if visual_ids[i] == visual_ids[j]:
                    if visual_ids[i] == visual_ids[anchor_idx]:
                        labels[i] = 1  # 如果两个视觉ID相同，视为正样本
                        labels[j] = 1  # 负样本对应的也是正样本对
                else:
                    if visual_ids[i] == visual_ids[anchor_idx] or visual_ids[j] == visual_ids[anchor_idx]:
                        # 如果目标ID不同于锚点目标，视为负样本
                        if visual_ids[i] != visual_ids[j]:
                            labels[i] = 0  # 负样本
                            labels[j] = 0  # 负样本

        return labels


def generate_heatmap(bboxes, img_size, num_classes):
    B, N, _ = bboxes.shape
    H, W = img_size
    heatmap = torch.zeros(B, num_classes, H, W)  # [B, C, H, W]
    for b in range(B):
        for n in range(N):
            x, y, w, h, cls = bboxes[b, n, :5]
            x1, y1, x2, y2 = int(x - w // 2), int(y - h // 2), int(x + w // 2), int(y + h // 2)
            heatmap[b, int(cls), y1:y2, x1:x2] = 1.0
    return heatmap
class VisualEncoder(nn.Module):
    def __init__(self, num_classes):
        super(VisualEncoder, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, heatmap):
        return self.backbone(heatmap)


# 加载预训练的 PANNs 模型（如 Cnn14 模型）
def load_cnn14():
    model = torch.hub.load('qiuqiangkong/panns', 'cnn14')  # 通过torch hub加载
    return model
class PANNsEncoder(nn.Module):
    def __init__(self, pretrained=True, feature_dim=512, model_path="path_to_local_downloaded_file.pth"):
        super(PANNsEncoder, self).__init__()
        # 加载本地保存的 PANNs 权重文件
        state_dict = torch.load(model_path, map_location="cpu")  # 使用本地文件路径

        # 定义模型结构
        self.model = load_cnn14()  # Cnn14 模型结构可以从 PANNs 源代码中导入
        self.model.load_state_dict(state_dict, strict=False)

        # 移除最后的分类层，提取中间特征
        self.model.fc_audioset = nn.Identity()  # 替换分类层
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        输入 GCFMap: [batch_size, 1, height, width]
        返回特征: [batch_size, feature_dim]
        """
        x = self.model(x)  # PANNs 提取的音频特征
        return x


class audioNet(nn.Module):
    def __init__(self, net_head, predNet):
        super(audioNet, self).__init__()
        self.net_head = net_head
        self.predNet = predNet

    def forward(self, x):
        x = self.net_head(x).permute(0, 2, 3, 1)  # [b,h,w,c]
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

