import random
import cv2
import numpy as np
from tools import ops

random.seed(1)

def indexSwich(annoFromeOne):  # 函数的作用是将标注从1索引转换为0索引
     "1-index to 0-index,[x,y,w,h]"
     if len(annoFromeOne.shape) == 1:# ref_anno dimention = 1 如果输入的标注是一维数组（1维），则将数组中的第一个和第二个元素分别减1
         annoFromeOne[0] = annoFromeOne[0] - 1
         annoFromeOne[1] = annoFromeOne[1] - 1
     else:  # 如果输入的标注是二维数组（2维），则将每一行的第一个和第二个元素分别减1
         annoFromeOne[:,0] = annoFromeOne[:, 0] - 1
         annoFromeOne[:,1] = annoFromeOne[:, 1] - 1
     annoFromZero = annoFromeOne
     return annoFromZero
