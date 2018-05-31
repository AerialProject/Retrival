import os
import time
import numpy as np
from PIL import Image

from utils.roi_pooling import r_mac
from utils.roi_extractor import roi_extractor

from PIL import ImageOps, ImageEnhance
import cv2

w_h = 224
f_size = 7

def Extract_feature(output,L=5):

    # TODO : CLAHE-Contarst Stretching
    '''bgr = cv2.imread(img_path)
    # COLOR_BGR2LAB     # COLOR_BGR2YCrCb   # COLOR_BGR2HSV
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LUV)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LUV2BGR)
    cv2.imwrite('./EXAMPLE/temp/after_clahe.jpg', bgr)
    img_ = Image.open('./EXAMPLE/temp/after_clahe.jpg')
    #img_.show()
    img = ImageOps.autocontrast(img_, cutoff=1)
    #img.show()'''

    # compute output
    #output = model.forward(x)

    # TODO : output 사이즈 체크!
    imagehelper = roi_extractor(L=L)
    rois, _ = imagehelper.rois((1,3,w_h,w_h))
    #rois, _ = imagehelper.rois((1,w_h,w_h,3))


    # feature = torch.FloatTensor(1,2048,7,7)
    #feature = torch.FloatTensor(1, 1024, f_size, f_size)
    #feature[0] = output.data
    out = r_mac(output,rois,spatial_scale=1/32.0) #1/16.0
    #input_feature = out

    print('done')
    return out


