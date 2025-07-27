import os
import glob
import random
import cv2
import numpy as np
from tools import ops
from tools import prepareTools
import matplotlib.pyplot as plt
import pickle
random.seed(1)


# # 数据集路径
# seqList = ['seq24-2p-0111']
# person_number = 2
# datasetPath_vi = '/amax/tyut/user/lyh/lyh/AV163'
# datasetPath = '/amax/tyut/user/lyh/lyh/AVCLData'  # 保存路径
#
# for sequence in seqList:
#     for cam_number in range(1, 4):  # (1, 4)
#         startfr, endfr = ops.getSE(sequence, cam_number)  # 1-index
#         seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
#         img_files = sorted(glob.glob(seq_dir + '*.jpg'))[startfr - 2:endfr]
#
#         # 随机选择一个锚点人物
#         anchor_person = random.randint(0, person_number - 1)  # 随机选择一个人物索引作为锚点
#
#         # 初始化保存路径
#         folderPath = f'{datasetPath}/AVsample/imgSample/{sequence}_cam{cam_number}'
#         if not os.path.exists(folderPath):
#             os.makedirs(folderPath)
#
#         # 遍历每帧
#         for i in range(len(img_files)):
#             frame_data = {
#                 'seqName': sequence,
#                 'camNum': cam_number,  # 摄像机编号
#                 'frameNum': i + startfr - 1,  # 当前帧编号，0-index
#                 'imgPath': img_files[i],
#                 'person_info': []  # 用于存储该帧中所有人物的信息
#             }
#
#             # 遍历所有人物，添加该帧的 GT 信息
#             for p in range(person_number):
#                 anno_path = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/{sequence}_cam{cam_number}-person{p + 1}_GT2D.txt'
#                 img_anno = np.loadtxt(anno_path)[startfr - 2:endfr]  # 获取标注
#                 img_anno = prepareTools.indexSwich(img_anno)  # 转为 0-index
#
#                 # 获取该帧的 GT box
#                 gtbox = img_anno[i]
#                 person_data = {
#                     'person_number': p + 1,  # 当前人物编号
#                     'gtbox': gtbox,  # 当前人物的 GT box
#                     'anchor_person': (p == anchor_person)  # 如果是锚点人物，则为 True
#                 }
#                 frame_data['person_info'].append(person_data)
#
#             # 保存该帧数据
#             filename = f'{str(10000 + i)[1:]}.pkl'  # 生成文件名，如 '0001.pkl'
#             outputPath = os.path.join(folderPath, filename)
#             with open(outputPath, 'wb') as output_file:
#                 pickle.dump(frame_data, output_file)
#             print(f'Saved {outputPath}')
#
# print('Save end')


# # # # #---------------Verify the .pkl file----------------------
datasetPath = '/amax/tyut/user/lyh/lyh/AVCLData'
sequence = 'seq24-2p-0111'
cam_number = 1
datasetPath_vi = '/amax/tyut/user/lyh/lyh/AV163'
startfr, endfr = ops.getSE(sequence, cam_number)#1-index
seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
img_files = sorted(glob.glob(seq_dir + '*.jpg'))[startfr - 2:endfr]
for i in range(len(img_files)):
    filename = str(10000+i)[1:]#'0000.pkl'
    folderPath = f'{datasetPath}/AVsample/imgSample/{sequence}_cam{cam_number}'
    pkl_file = open(f'{folderPath}/{filename}.pkl', 'rb')
    frame_dataIf = pickle.load(pkl_file)
    img_path = frame_dataIf['imgPath']
    img = cv2.imread(img_path)
    person_info = frame_dataIf['person_info']
    # for person in person_info:  # 分别显示单人的
    #     person_number = person['person_number']
    #     gtbox = person['gtbox']  # 当前人物的 GT box
    #     anchor_person = person['anchor_person']  # 是否为锚点人物
    # ops.showRecimg(img, gtbox)
    boxes = [person['gtbox'] for person in person_info]
    ops.showRecimg(img, boxes)
print('end')

