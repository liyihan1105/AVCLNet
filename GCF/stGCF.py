import os
import glob
import cv2
import numpy as np
from tools import ops
from tools import prepareTools
from GCF.GCF_extract_stGCF import getGCF
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
import time
import matplotlib.pyplot as plt
seqList = ['seq24-2p-0111'] # 'seq08-1p-0100', 'seq11-1p-0100','seq12-1p-0100'
datasetPath_vi = '/amax/tyut/user/lyh/lyh/AV163'
datasetPath = '/amax/tyut/user/lyh/lyh/AVCLData'

# 初始化存储MAE值的列表
all_MAE_curve = []

au_observe = getGCF()
total_result = list()
for sequence in seqList:
    # load audio data
    audioDATA = ops.loadaudioDATA(sequence, datasetPath)  # 加载音频数据
    GCCdata = ops.loadGCC(sequence, datasetPath)  # 加载GCC数据
    for cam_number in range(1, 4):
        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']
        startfr, endfr = ops.getSE(sequence, cam_number)  # 1-index 获取序列中当前摄像头的起始帧和结束帧
        seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'  # 设置当前序列摄像头的图像路径
        total_file = sorted(glob.glob(seq_dir + '*.jpg'))  # 获取该摄像头的所有图像文件
        total_GT = np.loadtxt(f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/'
                              f'{sequence}_cam{cam_number}_GT2D.txt')  # 加载当前摄像头的所有图像的GT2D标注
        # 3.img_files: total path of imgs.
        img_files = total_file[startfr - 1:endfr]  #根据起始帧和结束帧获取所需帧的图像文件列表
        # 4.anno is [x,y,w,h] of each frame
        img_anno = total_GT[startfr - 1:endfr]  # 根据起始帧和结束帧获取所需帧的GT2D标注
        img_anno = prepareTools.indexSwich(img_anno)  # "1-index to 0-index,[x,y,w,h]"
        #load total img
        img_data = list()  # 加载图像
        for i in range(len(img_files)):  # 执行任何所需的图像处理操作：颜色转化
            img_org = ops.read_image(img_files[i])
            img_data.append(img_org)  # 将处理后的图像添加到 img_data 列表中
    ###-----TRACKING---------- 用于计算预测结果的误差，并展示预测结果与真实值的对比
        error_curve = list()  # 用于存储每个样本的二维误差
        error_total = 0  # 用于累计总误差
        MAE_curve = list()  # 用于存储每个样本的平均绝对误差
        seq_result = {}  #  存储每个序列的结果
        for i in range(len(img_data)):  # 循环得到每个样本的预测结果和误差信息
            gt2d = np.array([img_anno[i][0]+img_anno[i][2]/2,
                            img_anno[i][1]+img_anno[i][3]/2])  # 计算当前样本的真实二维坐标 取 bounding box 的中心点坐标
            img_org = img_data[i]  # 获取当前样本的图像数据
            frameNum = i + startfr - 1  # 0-index 计算当前帧的编号
            loc2d, GCFmap = au_observe.au_observ(img_org, DATA, GCC, cam_number, frameNum)  # 获取当前样本的预测结果

            # # 可视化热图
            # plt.imshow(GCFmap, cmap='jet', interpolation='nearest')
            # plt.title('Response Heatmap')
            # plt.colorbar()
            # plt.show()

            error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))  # 预测结果与真实值之间的欧氏距离
            error_curve.append(error2d)  # 将当前样本的误差添加到误差曲线列表中
            error_total += error2d
            MAE = error_total / (i + 1)  # 计算当前样本的平均绝对误差
            MAE_curve.append(MAE)  # 当前样本的平均绝对误差添加到平均绝对误差曲线列表中

            # #---show GCF mean and results
            # plt.imshow(GCFmap)
            # plt.plot(loc2d[0], loc2d[1], 'r x', markersize=15)
            # plt.plot(gt2d[0], gt2d[1], 'g x', markersize=15)  # 绘制标记当前位置 loc2d 和真实位置 gt2d 的红色和绿色交叉标记
            # plt.show()
            # print("[{}_cam{} sample:{:0>3}/{:0>3}] [error2d:{:.4f} MAE:{:.4f}]".format(
            #     sequence, cam_number, i + 1, len(img_data), error2d, MAE))

        print("seq:{} cam:{} sample:{:0>3}/{:0>4} [ MAE:{:.4f} ]".
              format(sequence, cam_number, i, len(img_files), MAE))

        #     # 将当前序列的MAE值添加到all_MAE_curve中
        #     all_MAE_curve.append(MAE_curve)
        #
        # # 绘制MAE曲线
        # plt.plot(range(len(MAE_curve)), MAE_curve)
        # plt.xlabel('Frame')
        # plt.ylabel('MAE')
        # plt.title('MAE Curve')
        # plt.show()

#         #total results:
#         seq_result['seq'] = f'{sequence}_cam{cam_number}'
#         seq_result['MAE'] = MAE  # 将当前序列的序列编号和MAE存储在 seq_result 字典中
#         total_result.append(seq_result)  # 保存所有序列的结果
#
# print('[{} MAE:{:.4f} time:{:.0f}s fps:{:.2f} len:{:.0f} update:{:.0f} reset: {:.0f}]'.
#           format(seq_result['seq'], seq_result['MAE'], seq_result['time'], seq_result['fps']))  # 将当前序列的序列编号、MAE、时间、帧率、序列长度、更新次数和重置次数格式化成字符串
# print('avg MAE:{:.4f}'.format(np.mean([total_result[i]['MAE'] for i in range(len(total_result))])))  # 计算列表中所有序列的平均MAE
