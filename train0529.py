import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from models.AVCL_model import AVCLNet
from models.my_dataset import myDataset
from tools import ops


class KDNetTrainer():
    def main(self):
        self.loaddata()
        self.setmodel()
        self.train()

    def __init__(self):
        self.device_ids = [3]
        '''set seq and path'''
        self.trainSeqList = ['seq01-1p-0000', 'seq02-1p-0000', 'seq03-1p-0000']
        self.datasetPath = '/amax/tyut/user/lyh/lyh/STNet'
        '''set log path'''
        self.date = 'kdnet'
        self.BASE_DIR = '/amax/tyut/user/lyh/lyh/KD'
        self.log_dir = os.path.abspath(os.path.join(self.BASE_DIR, 'log6', "model_{0}.pth".format(self.date)))
        self.checkpoint_path = os.path.abspath(os.path.join(self.BASE_DIR, 'log6', 'checkpoint.pth'))
        '''set flag : train/test'''
        self.train_flag = True
        self.saveNetwork_flag = True
        self.drawCurve_flag = True
        ops.set_seed(1)
        self.MAX_EPOCH = 50
        self.BATCH_SIZE = 8
        self.LR = 0.0001
        self.log_interval = 16
        self.val_interval = 1
        self.save_interval = 1

    def loaddata(self):
        trainList, validList = ops.splitDataset(self.datasetPath, self.trainSeqList, splitType='train&valid',
                                                trainPct=0.8)
        refpath = f'{self.datasetPath}/AVsample/ref_seq123.pkl'
        with open(refpath, 'rb') as data:
            refData = pickle.load(data)
        train_data = myDataset(dataList=trainList, refData=refData)
        valid_data = myDataset(dataList=validList, refData=refData)
        self.train_loader = DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
        self.valid_loader = DataLoader(dataset=valid_data, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

    def setmodel(self):
        yoloface_weights_path = os.path.join(self.BASE_DIR, '/yoloface/weights', 'yoloface_weights.pth')
        yoloface_config_path = os.path.join(self.BASE_DIR, '/yoloface/models', 'yoloface_config.yaml')
        net_path_au = os.path.join(self.BASE_DIR, '/GCF/weights', 'GCFnet_pre.pth')
        net = AVCLNet(yoloface_weights_path=yoloface_weights_path, yoloface_config_path=yoloface_config_path, net_path_au=net_path_au)
        net = torch.nn.DataParallel(net, device_ids=self.device_ids)
        self.net = net.cuda(device=self.device_ids[0])
        torch.enable_grad()
        self.lossFn1 = nn.MSELoss(reduction='mean')
        self.lossFn2 = nn.MSELoss(reduction='mean')
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.LR, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Load checkpoint if it exists
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.net.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            self.start_epoch = 0

    def train(self):
        if self.train_flag:
            train_curve = list()
            valid_curve = list()
            for epoch in range(self.start_epoch, self.MAX_EPOCH):
                loss_mean = 0.
                self.net.train()
                for i, data in enumerate(self.train_loader):
                    refImg, sampleImg0, sampleImg1, sampleImg2, GCFmap, sampleFace = data
                    imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    auFr = Variable(torch.as_tensor(GCFmap, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    labels = Variable(torch.as_tensor(sampleFace, dtype=torch.float32), requires_grad=False) \
                        .cuda(device=self.device_ids[0])
                    # 梯度置零
                    self.optimizer.zero_grad()

                    # 前向传播
                    outputs, evl_factor, total_loss = self.net(imgRef, img0, img1, img2, auFr)
                    loss1 = self.lossFn1(outputs, labels)
                    r = outputs.detach()
                    dist = torch.sqrt(torch.sum((r - labels) ** 2, axis=1))
                    label2 = torch.div(2, torch.exp(0.05 * dist) + 1)
                    loss2 = self.lossFn2(evl_factor.squeeze(), label2)
                    total_loss = total_loss + loss1 + loss2

                    # 反向传播和参数更新
                    total_loss.backward()
                    self.optimizer.step()

                    # 记录损失
                    loss_mean += total_loss.item()
                    train_curve.append(total_loss.item())
                    if (i + 1) % self.log_interval == 0:
                        loss_mean = loss_mean / self.log_interval
                        print("[{}] Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                            self.date, epoch, self.MAX_EPOCH, i + 1, len(self.train_loader), loss_mean))
                        loss_mean = 0.
                # 调整学习率
                self.scheduler.step()

                # 验证模型
                if (epoch + 1) % self.val_interval == 0:
                    loss_val = 0.
                    with torch.no_grad():
                        for j, data in enumerate(self.valid_loader):
                            refImg, sampleImg0, sampleImg1, sampleImg2, GCFmap, sampleFace = data
                            imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            auFr = Variable(torch.as_tensor(GCFmap, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            labels = Variable(torch.as_tensor(sampleFace, dtype=torch.float32), requires_grad=False) \
                                .cuda(device=self.device_ids[0])
                            # 梯度置零
                            self.optimizer.zero_grad()

                            # 前向传播
                            outputs, evl_factor, total_loss = self.net(imgRef, img0, img1, img2, auFr)
                            loss1 = self.lossFn1(outputs, labels)
                            r = outputs.detach()
                            dist = torch.sqrt(torch.sum((r - labels) ** 2, axis=1))
                            label2 = torch.div(2, torch.exp(0.05 * dist) + 1)
                            loss2 = self.lossFn2(evl_factor.squeeze(), label2)
                            total_loss = total_loss + loss1 + loss2

                            # 记录损失
                            loss_val += total_loss.item()

                    # 计算并记录验证集损失
                    loss_val_epoch = loss_val / len(self.valid_loader)
                    valid_curve.append(loss_val_epoch)
                    print("Valid: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                        epoch, self.MAX_EPOCH, j + 1, len(self.valid_loader), loss_val_epoch))

                # 保存模型
                if (epoch + 1) % self.save_interval == 0:
                    log_dir = os.path.abspath(
                        os.path.join(self.BASE_DIR, "log6", f"model_{self.date}_ep{epoch + 1}.pth"))
                    state = {'model': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, log_dir)
                    torch.save(state, self.checkpoint_path)

            # 绘制损失曲线
            if self.drawCurve_flag:
                train_x = range(len(train_curve))
                train_y = train_curve
                train_iters = len(self.train_loader)
                valid_x = [i * train_iters * self.val_interval for i in range(1, len(valid_curve) + 1)]
                valid_y = valid_curve
                plt.plot(train_x, train_y, label='Train')
                plt.plot(valid_x, valid_y, label='Valid')
                plt.legend(loc='upper right')
                plt.ylabel('Loss value')
                plt.xlabel('Iteration')
                plt.show()
                print('End training')


if __name__ == '__main__':
    kdnet_trainer = KDNetTrainer()
    kdnet_trainer.main()