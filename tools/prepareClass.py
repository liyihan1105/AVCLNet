import os
import glob
import numpy as np
import wave
import matplotlib.pyplot as plt
from collections import namedtuple
import scipy.signal as signal#窗函数
import scipy.io as scio
import glob
import pickle
configCFGAttr = ['audiolaglist','Blist','Ilist','startGTlist','endGTlist','startFRlist','endFRlist',\
                 'map_num','m1','m2','fs','c','nw','inc','winfunc']
configGCF = namedtuple('configGCF', configCFGAttr)

camClsAttr = ['Pmat','K','alpha_c','kc']
camCls = namedtuple('camCls', camClsAttr)
def cfgSeqSet(sequence):
    if sequence == 'seq18-2p-0101':
        cfg = {
            'audiolaglist': [2.32, 1.80, 3.32],
            'Blist': [1.5, 2.5, 2],
            'Ilist': [0.35, 0.25, 0.3],
            'startGTlist': [0, 0, 0],
            'endGTlist': [0, 0, 0],
            'startFRlist': [125, 138, 100],
            'endFRlist': [1326, 1339, 1301]
        }

    elif sequence == 'seq24-2p-0111':
        cfg = {
            'audiolaglist': [0.84,  -0.28, 1.60],
            'Blist': [1.5, 2.75, 2.3],
            'Ilist': [0.35, 0.25, 0.35],
            'startGTlist': [1, 1, 1],
            'endGTlist': [1, 1, 1],
            'startFRlist': [315, 315, 260],#1-index
            'endFRlist': [500, 528, 481]
        }
    elif sequence == 'seq25-2p-0111':
        cfg = {
            'audiolaglist': [0.84, -0.28, 1.60],
            'Blist': [1.5, 2.75, 2.3],
            'Ilist': [0.35, 0.25, 0.35],
            'startGTlist': [1, 1, 1],
            'endGTlist': [1, 1, 1],
            'startFRlist': [125, 210, 80],  # 1-index
            'endFRlist': [225, 351, 270]
        }
    elif sequence == 'seq30-2p-1101':
        cfg = {
            'audiolaglist': [0.84, -0.28, 1.60],
            'Blist': [1.5, 2.75, 2.3],
            'Ilist': [0.35, 0.25, 0.35],
            'startGTlist': [1, 1, 1],
            'endGTlist': [1, 1, 1],
            'startFRlist': [128, 90, 60],  # 1-index
            'endFRlist': [248, 195, 145]
        }
    elif sequence == 'seq40-3p-0111':
        cfg = {
            'audiolaglist': [0.84, -0.28, 1.60],
            'Blist': [1.5, 2.75, 2.3],
            'Ilist': [0.35, 0.25, 0.35],
            'startGTlist': [1, 1, 1],
            'endGTlist': [1, 1, 1],
            'startFRlist': [],  # 1-index
            'endFRlist': []
        }
    elif sequence == 'seq45-3p-1111':
        cfg = {
            'audiolaglist': [0.84, -0.28, 1.60],
            'Blist': [1.5, 2.75, 2.3],
            'Ilist': [0.35, 0.25, 0.35],
            'startGTlist': [1, 1, 1],
            'endGTlist': [1, 1, 1],
            'startFRlist': [302, 360, 360],  # 1-index
            'endFRlist': [900, 900, 900]
        }

    cfg['map_num'] = 9
    cfg['m1'] = 14
    cfg['m2'] = 5
    cfg['fs'] = 16000
    cfg['c'] = 340
    cfg['nw'] = 640
    cfg['inc'] = 320
    cfg['winfunc'] = signal.hamming(cfg['nw'])
    cfgGCF = configGCF(**cfg)
    return cfgGCF

def enframe(signal, nw, inc, winfunc):
  '''
  nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
  inc:相邻帧的间隔（同上定义）
  '''
  signal_length=len(signal) #信号总长度
  if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
    nf=1
  else: #否则，计算帧的总长度
    nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))

  pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
  zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
  pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
  indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
  indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
  frames=pad_signal[indices] #得到帧信号
  win=np.tile(winfunc,(nf,1)) #window窗函数，这里默认取1
  return frames*win  #返回帧信号矩阵


class DataMain:
    def __init__(self,cfgGCF):
        self.audio = []
        self.cfgGCF = cfgGCF

    def readSynAudio(self, dataPath, audiolag):
        f = wave.open(dataPath, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        # synchronism
        if audiolag >= 0:
            wave_data = np.fromstring(str_data, dtype=np.short)[int(audiolag * framerate):]
        elif audiolag< 0:
            wave_org = np.fromstring(str_data, dtype=np.short)
            wave_zero = np.zeros((abs(int(audiolag * framerate)),))
            wave_data = np.concatenate((wave_zero,wave_org),axis = 0)
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))
        return wave_data

    def readAudio(self, dataPath, cam_number):
        audiolag = self.cfgGCF.audiolaglist[cam_number - 1]
        frameOrg = self.readSynAudio(dataPath, audiolag)
        frameWin = enframe(frameOrg, self.cfgGCF.nw, self.cfgGCF.inc, self.cfgGCF.winfunc)
        return frameWin

    def loadAudio(self, datasetPath, sequence, cam_number):
        for arrayNum  in range(1,3):
            for micNum in range(1,9):
                dataPath = f'{datasetPath}/{sequence}/{sequence}_array{arrayNum}_mic{micNum}.wav'
                frameWin = self.readAudio(dataPath, cam_number)
                self.audio.append(frameWin)

    def loadmicPos(self, datasetPath):
        dataPath = f'{datasetPath}/gt.mat'
        gtDict = scio.loadmat(dataPath)
        micPosData = gtDict['gt'][0][0][7]
        self.micPos = np.transpose(micPosData)

    def loadImgPath(self, datasetPath, sequence, cam_number):
        img_path = f'{datasetPath}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
        img_files = sorted(glob.glob(img_path + '*.jpg'))
        self.imgPath = img_files

    def loadCamAlign(self,datasetPath):
        dataPath = f'{datasetPath}/cam.mat'
        data = scio.loadmat(dataPath)['cam'][0]
        cam = {
            'Pmat': np.concatenate(([data[0][0]], [data[1][0]], [data[2][0]]), axis=0),
            'K': np.concatenate(([data[0][1]], [data[1][1]], [data[2][1]]), axis=0),
            'alpha_c': np.concatenate(([data[0][2]], [data[1][2]], [data[2][2]]), axis=0),
            'kc': np.concatenate(([data[0][3]], [data[1][3]], [data[2][3]]), axis=0),
        }

        # self.cam = namedtuple('cam',cam.keys())(**cam)
        self.cam = camCls(**cam)
        dataPath = f'{datasetPath}/rigid010203.mat'
        data = scio.loadmat(dataPath)['rigid'][0]
        self.align_mat = data[0][1]

    def loadGT3D(self, datasetPath, sequence, person_number):
        self.GT3D = {}
        for i in range(person_number):
            dataPath = f'{datasetPath}/{sequence}/{sequence}-person{i+1}_myDataGT3D.mat'
            data = scio.loadmat(dataPath)
            GT3DData = {f'{sequence}_p{i+1}': data['DataGT3D']}
            self.GT3D.update(GT3DData)

    def loadGT2D(self, datasetPath, sequence, cam_number, person_number):
        self.GT2D = {}
        for i in range(person_number):
            dataPath = f'{datasetPath}/{sequence}/{sequence}_cam{cam_number}_jpg/' \
                       f'{sequence}_cam{cam_number}-person{i+1}_GT2D.txt'
            GT2DData = {f'{sequence}_p{i+1}': np.loadtxt(dataPath)}
            self.GT2D.update(GT2DData)

def InitGCF(datasetPath, sequence, cam_number, person_number):
    CFGforGCF = cfgSeqSet(sequence)
    DATA = DataMain(CFGforGCF)
    DATA.loadAudio(datasetPath, sequence, cam_number)
    DATA.loadmicPos(datasetPath)
    DATA.loadImgPath(datasetPath, sequence, cam_number)
    DATA.loadCamAlign(datasetPath)
    DATA.loadGT3D(datasetPath, sequence, person_number)
    DATA.loadGT2D(datasetPath, sequence, cam_number,person_number)
    print(f'{sequence}_cam{cam_number}: initialfinished' )
    return DATA
#---------------------------------------------------------
