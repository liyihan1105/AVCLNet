import os
import glob
import random
import cv2
import numpy as np
from tools import ops
from tools import prepareTools
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
from GCF.GCF_extract_stGCF import getGCF
from models.KDnet_model0620 import KDNet

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def getdata(ref_examplar, sample, GCFmap):
    minGCF = -0.01
    maxGCF = 0.50
    refImg = ref_examplar.transpose(2, 0, 1)  # output is a ndarray[W*H*C]
    sampleImg0 = cv2.resize(sample, (300, 300), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    sampleImg1 = cv2.resize(sample, (400, 400), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    sampleImg2 = cv2.resize(sample, (550, 550), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

    GCF_nor = (GCFmap - minGCF) / (maxGCF - minGCF)
    GCF_nor_expanded = np.expand_dims(GCF_nor, axis=-1)
    GCF_nor = np.repeat(GCF_nor_expanded, 3, axis=-1)
    reGCFmap = cv2.resize(GCF_nor, (400, 400), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

    imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32).unsqueeze(0), requires_grad=False).cuda(device=device_ids[0])
    img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32).unsqueeze(0), requires_grad=True).cuda(device=device_ids[0])
    img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32).unsqueeze(0), requires_grad=True).cuda(device=device_ids[0])
    img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32).unsqueeze(0), requires_grad=True).cuda(device=device_ids[0])
    imgMask0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32).unsqueeze(0), requires_grad=True).cuda(device=device_ids[0])
    imgMask1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32).unsqueeze(0), requires_grad=True).cuda(device=device_ids[0])
    imgMask2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32).unsqueeze(0), requires_grad=True).cuda(device=device_ids[0])
    auFr = Variable(torch.as_tensor(reGCFmap, dtype=torch.float32).unsqueeze(0), requires_grad=False).cuda(device=device_ids[0])
    auFr_dou = Variable(torch.as_tensor(reGCFmap, dtype=torch.float32).unsqueeze(0), requires_grad=False).cuda(device=device_ids[0])
    return imgRef, img0, img1, img2, imgMask0, imgMask1, imgMask2, auFr, auFr_dou

date = 'stnet'
device_ids = [0]
random_seed = 1
random.seed(random_seed)
seqList = ['seq08-1p-0100']
datasetPath_vi = '/amax/tyut/user/lyh/lyh/Mask2/AV163'
datasetPath = '/amax/tyut/user/lyh/lyh/STNet'
s_size = 120
au_observe = getGCF()
BASE_DIR = '/amax/tyut/user/lyh/lyh/KD'
log_dir = os.path.abspath(os.path.join(BASE_DIR, 'log0620', 'model_kdnet_ep3.pth'))

# Load network
net = KDNet(net_path_vi=None, net_path_au=None)
net = torch.nn.DataParallel(net, device_ids=device_ids)
net = net.cuda(device=device_ids[0])
checkpoint = torch.load(log_dir, map_location=torch.device(f'cuda:{device_ids[0]}'))
net.load_state_dict(checkpoint['model'])

# Tracking
total_result = list()
frames = []  # Store frames for video

for sequence in seqList:
    audioDATA = ops.loadaudioDATA(sequence, datasetPath)
    GCCdata = ops.loadGCC(sequence, datasetPath)
    for cam_number in range(3, 4):
        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']

        startfr, endfr = ops.getSE(sequence, cam_number)
        seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
        total_file = sorted(glob.glob(seq_dir + '*.jpg'))
        total_GT = np.loadtxt(f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/{sequence}_cam{cam_number}_GT2D.txt')

        ref_file = total_file[startfr - 2]
        ref_anno = total_GT[startfr - 2]
        ref_anno = prepareTools.indexSwich(ref_anno)
        ref_examplar, ref_img = prepareTools.getExamplar(ref_file, ref_anno)

        img_files = total_file[startfr - 1:endfr]
        img_anno = total_GT[startfr - 1:endfr]
        img_anno = prepareTools.indexSwich(img_anno)

        img_data = list()
        for i in range(len(img_files)):
            img_org = ops.read_image(img_files[i])
            img_data.append(img_org)

        error_curve = list()
        error_total = 0
        error3d_total = 0
        MAE_curve = list()
        seq_result = {}

        for i in range(len(img_data)):
            gt2d = np.array([img_anno[i][0] + img_anno[i][2] / 2, img_anno[i][1] + img_anno[i][3] / 2])
            img_org = img_data[i]

            if i == 0:
                center = np.array([ref_anno[1] + ref_anno[3] / 2, ref_anno[0] + ref_anno[2] / 2])
            else:
                center = np.array([loc2d[1], loc2d[0]])

            _, sample = ops.crop_and_resize(
                img_org, center, size=120,
                out_size=120,
                border_value=np.mean(img_org, axis=(0, 1))
            )

            boxInSample, re_boxInSample, scale_id = prepareTools.vi_observ(
                ref_examplar, ref_anno, sample
            )

            box = np.array([
                boxInSample[0] + center[1] - s_size / 2,
                boxInSample[1] + center[0] - s_size / 2,
                boxInSample[2], boxInSample[3]
            ])

            sbox = np.array([
                0 + center[1] - s_size / 2,
                0 + center[0] - s_size / 2,
                s_size, s_size
            ])

            frameNum = i + startfr - 1

            loc, GCFmap = au_observe.au_observ(img_org, DATA, GCC, cam_number, frameNum)

            imgRef, img0, img1, img2, imgMask0, imgMask1, imgMask2, auFr, auFr_dou = getdata(ref_examplar, sample, GCFmap)
            outputs, evl_factor, total_loss, gen_loss_vi, gen_loss_au, dis_loss_vi, dis_loss_au = net(imgRef, img0, img1, img2, imgMask0, imgMask1, imgMask2, auFr, auFr_dou)

            output = outputs.detach().cpu().numpy()
            evl_factor = evl_factor.squeeze().detach().cpu().numpy()
            loc2d = np.array([output[0] + center[1] - s_size / 2, output[1] + center[0] - s_size / 2])

            error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))
            error_curve.append(error2d)
            error_total += error2d
            MAE = error_total / (i + 1)
            MAE_curve.append(MAE)
            print("seq:{} cam:{} sample:{:0>3}/{:0>4} [error2d:{:.4f} MAE:{:.4f}]".
                 format(sequence, cam_number, i, len(img_files), error2d, MAE))

            img_viz = img_org.copy()
            cv2.rectangle(img_viz, (int(gt2d[0] - img_anno[i][2] / 2), int(gt2d[1] - img_anno[i][3] / 2)),
                          (int(gt2d[0] + img_anno[i][2] / 2), int(gt2d[1] + img_anno[i][3] / 2)), (0, 255, 0), 2)
            cv2.rectangle(img_viz, (int(loc2d[0] - img_anno[i][2] / 2), int(loc2d[1] - img_anno[i][3] / 2)),
                          (int(loc2d[0] + img_anno[i][2] / 2), int(loc2d[1] + img_anno[i][3] / 2)), (0, 0, 255), 2)

            frames.append(img_viz)

# Save video
height, width, layers = frames[0].shape
video_name = '/amax/tyut/user/lyh/lyh/tracking_results.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))

for frame in frames:
    out.write(frame)

out.release()
print("Video has been saved.")
