import numpy as np
from tools import ops
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from itertools import *
class getGCF(object):
    def GCFextract(self, DATA, img, GCC, fa, cam_number):
#face data is from siamese measure
        #step1: generate 2D sample points
        img_size = img.shape
        w = int(img_size[1] / 3)#360将图像宽度分为3份
        h = int(img_size[0] / 3)#288将图像高度分为3份
        grid_x, grid_y = np.mgrid[0:img_size[1]-1:w*1j, 0:img_size[0]-1:h*1j]  # 生成均匀的网格点
        sample2d = np.concatenate((grid_x.reshape(w*h,-1),grid_y.reshape(w*h,-1)),axis = 1)   # 将网格点组合成2D样本点
    #show img and sampel2d points
        # plt.imshow(img)
        # plt.scatter(sample2d[:,0],sample2d[:,1],color='r',s = 0.15)
        # plt.show()
        #step2: generate 3D sample points
        sample3d = np.zeros((DATA.cfgGCF.map_num, sample2d.shape[0],3))  # 初始化3D样本点的数组
        p2d = np.zeros(shape=(sample2d.shape[0], 3))  # 初始化2D点的数组
        p2d[:, :-1] = sample2d  # 将2D样本点的坐标赋值给p2d数组
        for i in range(DATA.cfgGCF.map_num):
            z = DATA.cfgGCF.Blist[cam_number - 1] + DATA.cfgGCF.Ilist[cam_number - 1] * (i)  # 计算z坐标
            p3d = ops.p2dtop3d_2(p2d, z, DATA.cam, DATA.align_mat, cam_number)  # 将2D点投影到3D空间中
            sample3d[i,:] = p3d.T  # 存储3D样本点
        #find outside samples
        outlier = [[],[]]  # 存储异常点的索引
        for g in range(DATA.cfgGCF.map_num):
            for i in range(sample3d[0].shape[0]):
                p = sample3d[g][i]
                if p[0] <= -1.8  or p[0] >= 1.8 or \
                   p[1] <= -7.2  or p[1] >= 2   or \
                   p[2] <= -0.04 or p[2] >= 1.56:
                    outlier[0].append(g)
                    outlier[1].append(i)
        #step3: tau3d
        pairNum = (1 + len(DATA.audio) - 1) * (len(DATA.audio) - 1) / 2  # 计算音频对的数量
        tau3d = np.zeros((DATA.cfgGCF.map_num, int(pairNum), sample2d.shape[0]))  # 初始化tau3d数组
        interp = 1
        max_shift = int(interp * DATA.audio[0].shape[1])  # 最大偏移
        for g in range(DATA.cfgGCF.map_num):
            tau3dlist = np.zeros(shape=(int(pairNum), sample2d.shape[0]))  # 初始化tau3d列表
            t = 0
            for mici in range(len(DATA.audio)):
                di = np.sqrt(np.sum(np.asarray(DATA.micPos[mici] - sample3d[g]) ** 2, axis=1))  # 计算到每个麦克风的距离
                for micj in range(mici + 1, len(DATA.audio)):
                    dj = np.sqrt(np.sum(np.asarray(DATA.micPos[micj] - sample3d[g]) ** 2, axis=1))  # 计算到每个麦克风的距离
                    tauijk = (di - dj) / DATA.cfgGCF.c  # 计算时间延迟
                    taun = np.transpose(tauijk * DATA.cfgGCF.fs)  # 转置
                    taun = np.rint(taun * interp + max_shift)  # 四舍五入
                    tau3dlist[t, :] = taun  # 存储时间延迟
                    t = t + 1
            tau3d[g, :] = tau3dlist  # 存储时间延迟
    # step4:[fa-m3,fa]
        rGCF, rGCFmax = cal_rGCFmax(DATA, sample2d, tau3d, GCC, outlier, fa)
    # step5:find top-3's t and depth, get indes: max_t_ind, max_d_ind
        top_num = 3
        # 提取前 top_num 个最大值
        max_t_inds = np.argpartition(rGCFmax, -top_num)[-top_num:]  # 返回前 top_num 个时间索引
        locs = []
        combined_heatmap = np.zeros((img_size[0], img_size[1]))

        for max_t_ind in max_t_inds:
            ind = np.unravel_index(rGCF[max_t_ind].argmax(), rGCF[max_t_ind].shape)
            loc = sample2d[ind[-1]]
            locs.append(loc)

            max_map_v = rGCF[max_t_ind, ind[0], :].squeeze()
            max_map = max_map_v.reshape(grid_x.shape[1], grid_x.shape[0], order='F')
            GCFmap = cv2.resize(max_map, (img.shape[1], img.shape[0]))

            # 累加热图
            combined_heatmap += GCFmap
        # 归一化热图
        combined_heatmap = combined_heatmap / combined_heatmap.max()

        return locs, combined_heatmap


    # def au_observ(self, cam_number, frameNum, img):  # 将图像和音频映射到同一区间帧
    def au_observ(self, img, DATA, GCC, cam_number, frameNum):  # 将图像和音频映射到同一区间帧
        fr = frameNum #0-index
        fa = int(2*fr-2)
        gcfmap = self.GCFextract(DATA, img, GCC, fa, cam_number)

        return gcfmap

def cal_tau(DATA, sample2d, sample3d):
    pairNum = (1+len(DATA.audio)-1)*(len(DATA.audio)-1)/2
    tau3d = np.zeros((DATA.cfgGCF.map_num, int(pairNum), sample2d.shape[0]))
    interp = 1
    max_shift = int(interp * DATA.audio[0].shape[1])
    for g in range(DATA.cfgGCF.map_num):
        tau3dlist = np.zeros(shape = (int(pairNum),sample2d.shape[0]))
        t = 0
        for mici in range(len(DATA.audio)):
            di = np.sqrt(np.sum(np.asarray( DATA.micPos[mici]- sample3d[g])**2, axis=1))
            for micj in range(mici + 1, len(DATA.audio)):
                dj = np.sqrt(np.sum(np.asarray( DATA.micPos[micj]- sample3d[g])**2, axis=1))
                tauijk = (di - dj)/DATA.cfgGCF.c
                taun = np.transpose(tauijk * DATA.cfgGCF.fs)
                taun = np.rint(taun * interp + max_shift)
                tau3dlist[t,:] = taun
                t = t + 1
        tau3d[g,:] = tau3dlist
    return tau3d


def cal_rGCFmax(DATA, sample2d, tau3d, GCC, outlier,fa):  # 计算上述rGCFmax 是 rGCF（GCF的空间域表示）的最大值
    m3 = 15
    rGCF = np.zeros(shape=(m3, DATA.cfgGCF.map_num, sample2d.shape[0]))
    rGCFmax = np.zeros(shape=m3)
    gccx = get_chain_np(tau3d.shape[2], 0, tau3d.shape[1], tau3d.shape[0])  # 计算相关性
    gccy = tau3d.reshape(-1).astype(int)  # 三维数组tau3d转换为一维数组，并将其元素类型转换为整数

    for i in range(m3):
        fn = fa - i
        cc = GCC[fn]

        rPHAT= cc[gccx, gccy].reshape(*tau3d.shape)

        rGCForg = np.mean(rPHAT, axis=1)
        rGCForg[outlier[0], outlier[1]] = 0  # set outside sample as 0
        rGCF[i] = rGCForg
        rGCFmax[i] = np.max(rGCF[i]) * (1 - i * 0.0125)  ###punishment for time lag
    return rGCF, rGCFmax

def getnum(mat, x, y):  # 接受一个矩阵和两个索引参数，并返回矩阵中对应索引位置的值
    return mat[x ,y]

def reshape2(mat, *l):
    return mat.reshape(*l)

def cal_GCF2_v5(top_num, sample2d, tau3d, max_t_ind, fa, GCC, outlier, grid_x):
    reGCF = np.zeros(shape=(top_num, sample2d.shape[0]))
    gccx = get_chain_np(tau3d.shape[2], 0, tau3d.shape[1], 1)
    for i in range(top_num):
        t = max_t_ind[i]
        fn = fa - t
        cc = GCC[fn]  # [120*len(cc)] 使用了预先计算的交叉相关性（GCC）来计算每个采样点的GCF值
        gccy = tau3d[i].reshape(-1).astype(int)
        r = getnum(cc, gccx, gccy)  # 根据声源的时间差索引获取对应的GCC值
        rPHAT = reshape2(r, *tau3d[0].shape)

        retGCF = np.mean(rPHAT, axis=0)
        reGCF[i] = retGCF * (1 - t * 0.0125)
    reGCF[outlier[0], outlier[1]] = 0  # set outside sample as 0对于异常值，直接将其设为0
    reGCFmaporg = reGCF.reshape(-1, grid_x.shape[0], grid_x.shape[1], order='F')  # reshape to w*h将得到的GCF值重塑成与图像网格相同大小的矩阵
    return reGCFmaporg


def cal_GCF2_v3(top_num, sample2d, tau3d, max_t_ind, fa, GCC, outlier, grid_x, pairNum):
    reGCF = np.zeros(shape=(top_num, sample2d.shape[0]))
    for i in range(top_num):
        t = max_t_ind[i]
        fn = fa - t
        rPHAT = np.zeros(shape=(sample2d.shape[0], int(pairNum)))
        for j in range(tau3d.shape[1]):
            cc = GCC[fn][j]
            indexij = tau3d[i][j]
            rPHAT[:, j] = cc[indexij.astype(int)]
        retGCF = np.mean(rPHAT, axis=1)
        reGCF[i] = retGCF * (1 - t * 0.0125)
    reGCF[outlier[0], outlier[1]] = 0  # set outside sample as 0
    reGCFmaporg = reGCF.reshape(-1, grid_x.shape[0], grid_x.shape[1], order='F')  # reshape to w*h
    return reGCFmaporg

def get_chain(repeat_times , range_start , range_end , total_repeat_times ):
    p = list(chain(*[[x] * repeat_times for x in range(range_start, range_end)])) * total_repeat_times
    p = np.array(p)
    return p

def get_chain_np(repeat_times , range_start , range_end , total_repeat_times ):  # 该函数的作用是生成一个用于计算相关性的索引数组
    p2 = np.tile(np.repeat(np.arange(range_start, range_end),
                           repeat_times), (1, total_repeat_times)).flatten()
    return p2