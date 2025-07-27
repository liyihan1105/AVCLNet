from tools import ops
import pickle
import glob
import cv2
import os
import numpy as np
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
from GCF.GCF_extract_vg import getGCF
import matplotlib.pyplot as plt
import time
import random
#set seq
seqList = ['seq24-2p-0111']
datasetPath = '/amax/tyut/user/lyh/lyh/AVCLData'
au_observe = getGCF()
for sequence in seqList:
    GCCdata = ops.loadGCC(sequence, datasetPath)
    audioDATA = ops.loadaudioDATA(sequence, datasetPath)
    for cam_number in range(1,4):
        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']
        folderPath = f'{datasetPath}/AVsample/imgSample/{sequence}_cam{cam_number}/'
        fileList = sorted(glob.glob(folderPath + '*.pkl'))
        error_curve = list()
        error_total = 0
        MAE_curve = list()
        index = random.sample(range(0, len(fileList)), round(len(fileList) * 0.75))  # index of face guide
        start = time.time()


        for i in range(len(fileList)):  # 遍历每一帧
            pkl_file = open(f'{fileList[i]}', 'rb')
            frame_dataIf = pickle.load(pkl_file)
            img = ops.read_image(frame_dataIf['imgPath'])  # 原始图像
            person_info = frame_dataIf['person_info']  # 获取每帧的多目标信息
            # 初始化误差
            error_total = np.zeros(len(person_info))  # 每个声源的总误差
            MAE_curve = [[] for _ in range(len(person_info))]  # 每个声源的 MAE 曲线

            # 获取每个目标的预测坐标和 GT 坐标
            for idx, person in enumerate(person_info):  # 遍历每个声源目标
                sampleFace = person['gtbox']  # 真实值（bounding box）
                gt2d = np.array([sampleFace[0] + sampleFace[2] / 2,
                                 sampleFace[1] + sampleFace[3] / 2])  # 中心坐标作为真实值

                # 获取预测值 loc2d
                GCFmap, depth_ind  = au_observe.au_observ(DATA, img, GCC, cam_number, frame_dataIf['frameNum'],
                                                     box=sampleFace)
                gcfData = {
                    'GCFmap': GCFmap,
                    'depth_ind': depth_ind,
                }
                ###---calculate the errors
                gcf_t = cv2.resize(np.sum(GCFmap, axis=0), (120, 120))
                ind = np.unravel_index(gcf_t.argmax(), gcf_t.shape)
                loc2d = np.array([ind[1], ind[0]])
                error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))  # 欧氏距离
                error_total[idx] += error2d  # 累计误差
                MAE = error_total[idx] / (i + 1)  # 当前声源的 MAE
                MAE_curve[idx].append(MAE)  # 更新 MAE 曲线

        # 输出每个声源的 MAE
        for source_idx in range(len(person_info)):
            print(f"Cam {cam_number} Source {source_idx + 1}: Final MAE = {MAE_curve[source_idx][-1]:.4f}")

        print("[{}_cam{} sample:{:0>3}/{:0>3}] [error2d:{:.4f} MAE:{:.4f}] ".format(
                sequence, cam_number, i + 1,len(fileList), error2d, MAE))


        plt.plot(MAE_curve)
        plt.title(f'MAE of {sequence}cam_{cam_number} MAE={MAE}')
        plt.show()
