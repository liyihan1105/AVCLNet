import numpy as np
import random
from torch.utils.data import Dataset
import cv2
import pickle

random.seed(1)
class myDataset(Dataset):
    def __init__(self,dataList, refData):
        self.refData = refData
        self.dataList = dataList
        self.minGCF = -0.01
        self.maxGCF = 0.50

    def __getitem__(self, index):
        imgSamplePath= self.dataList[index]
        # img_file = open(imgSamplePath, 'rb')
        # imgData = pickle.load(img_file) # get whole img data
        # refname = imgSamplePath.split('/')[-2]
        # refImg = self.refData[refname]#
        #
        # auSamplePath = imgSamplePath.replace('imgSample', "GCFmap_train")
        # au_file = open(auSamplePath, 'rb')
        # auData = pickle.load(au_file)


        try:
            with open(imgSamplePath, 'rb') as img_file:
                imgData = pickle.load(img_file)  # get whole img data
        except EOFError:
            print(f"EOFError: File {imgSamplePath} is corrupted or incomplete.")
            raise

        refname = imgSamplePath.split('/')[-2]
        refImg = self.refData[refname]

        imgSamplePathMask = imgSamplePath.replace('imgSample', "imgSampleMask4")
        try:
            with open(imgSamplePathMask, 'rb') as imgMask_file:
                imgDataMask = pickle.load(imgMask_file)
        except EOFError:
            print(f"EOFError: File {imgSamplePathMask} is corrupted or incomplete.")
            raise

        auSamplePath = imgSamplePathMask.replace('imgSampleMask4', "GCFmap_train")
        try:
            with open(auSamplePath, 'rb') as au_file:
                auData = pickle.load(au_file)
        except EOFError:
            print(f"EOFError: File {auSamplePath} is corrupted or incomplete.")
            raise

        auSamplePathDou = auSamplePath.replace('GCFmap_train', "GCFmap_train_dou3")
        try:
            with open(auSamplePathDou, 'rb') as au_dou_file:
                auDataDou = pickle.load(au_dou_file)
        except EOFError:
            print(f"EOFError: File {auSamplePathDou} is corrupted or incomplete.")
            raise


        #sample need resize
        sampleImg = imgData['sampleImg']
        sampleImg0 = cv2.resize(sampleImg, (300, 300)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
        sampleImg1 = cv2.resize(sampleImg, (400, 400)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
        sampleImg2 = cv2.resize(sampleImg, (500, 500)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)

        sampleFace = imgData['sampleFace']
        gt2d = np.array([sampleFace[0] + sampleFace[2] / 2, sampleFace[1] + sampleFace[3] / 2])

        sampleImg = imgDataMask['sampleImg']
        sampleImgMask0 = cv2.resize(sampleImg, (300, 300)
                                , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        sampleImgMask1 = cv2.resize(sampleImg, (400, 400)
                                , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        sampleImgMask2 = cv2.resize(sampleImg, (500, 500)
                                , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        GCFmap = auData['GCFmap']  # 3*H*W
        GCF_nor = (GCFmap - self.minGCF) / (self.maxGCF - self.minGCF)
        GCF_nor_expanded = np.expand_dims(GCF_nor, axis=-1)

        # 将新的维度插入到指定位置
        GCF_nor = np.repeat(GCF_nor_expanded, 3, axis=-1)
        # GCF_nor = np.transpose(GCF_nor, (2, 0, 1))  # H*W*3
        # print(GCF_nor.shape)
        reGCFmap = cv2.resize(GCF_nor, (400, 400)
                              , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)

        GCFmap_dou = auDataDou['GCFmap']  # 3*H*W
        GCF_nor_dou = (GCFmap_dou - self.minGCF) / (self.maxGCF - self.minGCF)
        GCF_nor_expanded_dou = np.expand_dims(GCF_nor_dou, axis=-1)

        # 将新的维度插入到指定位置
        GCF_nor_dou = np.repeat(GCF_nor_expanded_dou, 3, axis=-1)
        # GCF_nor = np.transpose(GCF_nor, (2, 0, 1))  # H*W*3
        # print(GCF_nor.shape)
        reGCFmap_dou = cv2.resize(GCF_nor_dou, (400, 400)
                              , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

        return refImg, sampleImg0, sampleImg1, sampleImg2, sampleImgMask0, sampleImgMask1, sampleImgMask2, reGCFmap, reGCFmap_dou, gt2d

    def __len__(self):
        return len(self.dataList)
