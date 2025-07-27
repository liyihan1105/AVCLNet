import os
import pickle
from tools.prepareClass import InitGCF
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls

if __name__ == '__main__':
    seqList = ['seq24-2p-0111']
    # seqList = ['seq01-1p-0000', 'seq02-1p-0000', 'seq03-1p-0000']
    datasetPath_au = '/amax/tyut/user/lyh/lyh/AV163'  # 原始数据集的路径
    datasetPath_save = '/amax/tyut/user/lyh/lyh/AVCLData'

    for sequence in seqList:
        audioDataDic= {}
        for cam_number in range(1, 4):
            DATA, CFGforGCF = InitGCF(datasetPath_au, sequence, cam_number)  # 初始化处理音频数据所需的参数
            audioData = {f'{sequence}_cam{cam_number}': DATA}  # 创建一个包含音频数据的字典，键为序列名称和相机编号的组合，值为音频数据
            audioDataDic.update(audioData)  # 更新字典 将每个相机的音频数据添加到其中

        folderPath = f'{datasetPath_save}/audio/{sequence}'  # 构建保存音频数据的文件夹路径
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        output = open(f'{folderPath}/{sequence}_audio.pkl', 'wb')  # 创建一个二进制写入模式的pickle文件对象，准备将音频数据写入其中
        pickle.dump(audioDataDic, output)  # 将音频数据字典写入pickle文件中
        print(f'save audio.npz for {sequence}')
