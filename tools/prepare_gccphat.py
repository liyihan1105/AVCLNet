import numpy as np
from tools import ops
import os
import pickle
seqList = ['seq24-2p-0111']
# seqList = ['seq01-1p-0000', 'seq02-1p-0000', 'seq03-1p-0000']
datasetPath = '/amax/tyut/user/lyh/lyh/AVCLData'
m3 = 15
interp=1
audioData = ops.loadAudio(seqList, datasetPath)  # 函数的作用是加载音频数据，并将其存储在一个字典中
for sequence in seqList:  # 对每个序列进行循环
    GCCDic = {}  # 用于存储GCC数据的字典
    for cam_number in range(1, 4):
        DATA = audioData[f'{sequence}_cam{cam_number}']
        startfr, endfr = ops.getSE(sequence, cam_number)  # 1-index 获取序列的起始帧和结束帧
        startfr = startfr -1
        startfa = int(2 * startfr - 2) - m3
        endfr = endfr -1
        endfa = int(2 * endfr - 2)
        GCCPHATlist = list()   # 存储每个fa的GCCPHAT数据
        for fa in range(len(DATA.audio[0])):  # 对每个fa进行循环
            GCCPHAT = list()  # 存储每个mici和micj对应的GCCPHAT
            if fa>= startfa and fa<= endfa:  # 如果fa在起始fa和结束fa之间
                for mici in range(len(DATA.audio)):   # 对每个麦克风对进行循环
                    for micj in range(mici + 1, len(DATA.audio)):
                        sig = DATA.audio[mici][fa]   # 获取当前麦克风对应的信号
                        refsig = DATA.audio[micj][fa]
                        cc, _ = ops.gcc_phat(sig, refsig, fs=DATA.cfgGCF.fs, max_tau=None, interp=interp)   # 计算GCC_PHAT
                        GCCPHAT.append(cc)
                        print(f'calculate GCC for {sequence}_cam{cam_number}: {fa}/{len(DATA.audio[0])}')
            GCCPHATlist.append(np.array(GCCPHAT))  # 将GCCPHAT添加到列表中
        data = {f'{sequence}_cam{cam_number}': GCCPHATlist}  # 创建字典存储GCCPHAT数据
        GCCDic.update(data)  # 更新GCC字典
    folderPath = f'{datasetPath}/GCC'  # GCC数据保存的文件夹路径
    if not os.path.exists(folderPath):  # 如果文件夹不存在，则创建
        os.makedirs(folderPath)
    output = open(f'{folderPath}/{sequence}_GCC.pkl', 'wb')  # 用于写入二进制数据
    pickle.dump(GCCDic, output)  # 将GCC字典数据写入文件
    print(f'save _GCC.pkl for {sequence}')  # 打印保存了哪个序列的GCC数据

