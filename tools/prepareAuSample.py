from tools import ops
import pickle
import glob
import os
import numpy as np
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
from GCF.GCF_extract_vg import getGCF

seqList = ['seq03-1p-0000'] # 'seq01-1p-0000', 'seq02-1p-0000', 'seq03-1p-0000'
datasetPath = '/amax/tyut/user/lyh/lyh/AVCLData'
# load audio data 从视频序列中提取样本图像，并对样本图像进行处理，包括获取样本图像、人脸信息、转换坐标等 将处理后的数据保存为 .pkl 文件

au_observe = getGCF()
for sequence in seqList:
    audioDATA = ops.loadaudioDATA(sequence, datasetPath)  # 加载音频数据
    GCCdata = ops.loadGCC(sequence, datasetPath)  # 加载GCC数据
    for cam_number in range(2,4):
        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']
        folderPath = f'{datasetPath}/yoloFaceSamples/{sequence}_cam{cam_number}/'
        fileList = sorted(glob.glob(folderPath + '*.pkl'))  # 使用 glob 模块列出指定路径下的所有pickle文件，并按文件名排序
        error_curve = list()
        error_total = 0
        MAE_curve = list()
        for i in range(len(fileList)):  # 第3000个文件索引是2999
            pkl_file = open(f'{fileList[i]}', 'rb')  # 打开当前文件，并以二进制读取模式打开pickle文件
            sampleIf = pickle.load(pkl_file)  # 加载pickle文件中的数据
            frameNum     = sampleIf['frameNum']  # 获取帧数
            imgPath      = sampleIf['imgPath']  # 获取图像路径
            img          = ops.read_image(imgPath)  # 读取图像
            loc, GCFmap = au_observe.au_observ(img, DATA, GCC, cam_number, frameNum)
            gcfData = {
                'GCFmap': GCFmap,
            }
        ###--- save the imgDataList as {sequence}_sampleList.npz
            folderPath = f'{datasetPath}/GCFmap_train_dou/{sequence}_cam{cam_number}'
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            filename = str(10000 + i)[1:]  # '0000.pkl'
            outputPath = open(f'{folderPath}/{filename}.pkl', 'wb')
            pickle.dump(gcfData, outputPath)  # 将处理后的数据 gcfData 写入pickle文件中
            print(f'save gcfData.pkl for {sequence}_cam{cam_number}_{filename}')

