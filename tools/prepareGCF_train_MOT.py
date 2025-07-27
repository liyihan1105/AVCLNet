from numpy.core._multiarray_umath import ndarray

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
'''the training step, we use annotation facebox to get GCF'''

#set seq
seqList = ['seq30-2p-1101']#'seq18-2p-0101''seq24-2p-0111''seq25-2p-0111'
person = 2
datasetPath = '/data3/liyd/dataset/MPTracker'
#load audio data
audioData = ops.loadAudio(seqList, datasetPath)
au_observe = getGCF(audioData)
for sequence in seqList:
    for cam_number in range(1,4):  # (1, 4)
        for person_number in range(1,person+1):
            folderPath = f'{datasetPath}/AVsample1/imgSample/{sequence}_cam{cam_number}/'
            key = f'p{person_number}'
            l = [os.path.join(folderPath, x) for x in os.listdir(folderPath)]
            fileList = sorted([x for x in l if os.path.isfile(x) if key in os.path.splitext(os.path.basename(x))[0]])

            error_curve = list()
            error_total = 0
            MAE_curve = list()
            for i in range(len(fileList)):#len(fileList)
                pkl_file = open(f'{fileList[i]}', 'rb')
                sampleIf = pickle.load(pkl_file)
                sampleImg    = sampleIf['sampleImg']
                imgAnno      = sampleIf['imgAnno']
                refImg       = sampleIf['refImg']
                frameNum     = sampleIf['frameNum']
                imgPath      = sampleIf['imgPath']
                img          = ops.read_image(imgPath)  # org img
                sampleFace = sampleIf['sampleFace']
                sampleImgBox = sampleIf['sampleImgBox']##sample_img_box for square_img (square coordinate)
            # box, re_box(in sample coordinate) turn to img coordinate, first.
                new_x = sampleImgBox[0]
                new_y = sampleImgBox[1]
                sampleInImg = ops.square2img(sampleImgBox)
                # ops.showRecimg(img, sampleInImg)
            #get GCFmap, and depth index
                GCFmap, depth_ind, gt3d = au_observe.au_observ(sequence, cam_number,person_number, frameNum,
                                              img, box=imgAnno, spl_box=sampleInImg)
                gcfData = {
                    'GCFmap': GCFmap,
                    'depth_ind': depth_ind,
                    'GT3D':gt3d,
                }
            ###---calculate the errors
                gcf_t = cv2.resize(np.sum(GCFmap, axis=0), (120, 120))
                ind = np.unravel_index(gcf_t.argmax(), gcf_t.shape)
                loc2d = np.array([ind[1], ind[0]])
                gt2d = np.array([sampleFace[0] + sampleFace[2] / 2, sampleFace[1] + sampleFace[3] / 2])
                error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))
                error_curve.append(error2d)
                error_total += error2d
                MAE = error_total / (i + 1)
                MAE_curve.append(MAE)
                if gt3d.shape[0] == 0: act = 0
                else: act = 1
                print("[{}_cam{} sample:{:0>3}/{:0>3}] [error2d:{:.4f} MAE:{:.4f}] [active: {}]".format(
                    sequence, cam_number, i + 1,len(fileList), error2d, MAE, act))
            ##---show GCF mean and results
                # plt.imshow(gcf_t)
                # plt.plot(loc2d[0], loc2d[1], 'r x', markersize=15)
                # plt.plot(gt2d[0], gt2d[1], 'g x', markersize=15)
                # plt.show()
                # plt.imshow(sampleImg)
                # plt.plot(loc2d[0], loc2d[1], 'r x', markersize=15)
                # plt.plot(gt2d[0], gt2d[1], 'g x', markersize=15)
                # plt.show()
                # print('fileList[i]')

            # ###--- save the imgDataList as {sequence}_sampleList.npz
                folderPath = f'{datasetPath}/AVsample1/GCFmap_train/{sequence}_cam{cam_number}'
                if not os.path.exists(folderPath):
                    os.makedirs(folderPath)
                filename = str(10000 + i)[1:]  # '0000.pkl'
                outputPath = open(f'{folderPath}/p{person_number}_{filename}.pkl', 'wb')
                #!!! save the pkl file!!!!!--------------
                pickle.dump(gcfData, outputPath)
                print(f'save gcfData.pkl for {folderPath}/p{person_number}_{filename}.pkl')

####--plot error and MAE curves
            # plt.plot(error_curve)
            # plt.title(f'error2d of {sequence}cam_{cam_number}_p{person_number} MAE={MAE}')
            # plt.show()
            # plt.savefig(f'{sequence}cam_{cam_number}_error2d.png')

            plt.plot(MAE_curve)
            plt.title(f'MAE of {sequence}cam_{cam_number}_p{person_number} MAE={MAE}')
            plt.show()
            # plt.savefig(f'{sequence}cam_{cam_number}_MAE.png')

print('end')

# # # # # #---------------Verify the .pkl file----------------------
# datasetPath = '/home/liyd/myWork/dataset/MPTracker'
# sequence = 'seq08-1p-0100'
# cam_number = 2

# folderPath = f'{datasetPath}/AVsample1/imgSample/{sequence}_cam{cam_number}/'
# fileList = sorted(glob.glob(folderPath + '*.pkl'))
# for i in range(400,500):
#     pkl_file = open(f'{fileList[i]}', 'rb')
#     sampleIf = pickle.load(pkl_file)
#     sampleImg    = sampleIf['sampleImg']
#     refImg       = sampleIf['refImg']
#     imgPath = sampleIf['imgPath']
#     img          = ops.read_image(imgPath)  # org img
#
#     filename = str(10000+i)[1:]#'0000.pkl'
#     folderPath = f'{datasetPath}/AVsample1/GCFmap_train/{sequence}_cam{cam_number}'
#     pkl_file = open(f'{folderPath}/{filename}.pkl', 'rb')
#     auData = pickle.load(pkl_file)
#     GCFmap = auData['GCFmap']
#     depth_ind = auData['depth_ind']
#
#     ops.showData(sampleImg)
#     ops.showData(GCFmap[0])
#
#     print('end')

# for c in range(1,4):
#     cam_number = c
#     folderPath = f'{datasetPath}/AVsample1/GCFmap_train/{sequence}_cam{cam_number}/'
#     fileList = sorted(glob.glob(folderPath + '*.pkl'))
#     gcfList = []
#     for i in range(len(fileList)):
#         filename = str(10000 + i)[1:]  # '0000.pkl'
#         pkl_file = open(f'{folderPath}/{filename}.pkl', 'rb')
#         auData = pickle.load(pkl_file)
#         GCFmap = auData['GCFmap']
#
#         gcfList.append(GCFmap)
#     print(np.max(gcfList))
#     print(np.min(gcfList))
# print('end')
