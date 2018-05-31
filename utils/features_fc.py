import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.resnet_fc import resnet101 as resnet

def Transformer():
    # Use this to preprocessing images before the extractor

    #normalize = transforms.Normalize(mean=[average_color[0]/255, average_color[1]/255, average_color[2]/255],
     #                                std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([transforms.Scale(224),transforms.CenterCrop(224),
        transforms.ToTensor(),normalize])
    return trans

def Transformer_perExample(average_color):
    # Use this to preprocessing images before the extractor
    normalize = transforms.Normalize(mean=[average_color[0]/255, average_color[1]/255, average_color[2]/255],
                                     std=[0.229, 0.224, 0.225])
#    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
 #                                    std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
    return trans



def Load_pretrained(path):
    # Prepare ResNet
    model = resnet(pretrained=False).eval()
    model.fc = nn.Linear(model.fc.in_features, 30)

    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    # create new OrderedDict that does not contain `module.`
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model

def Extract_database_features(model,path,npz_name="./features/database_features_fc.npy"):
    # FORWARD images in Database
    bs = 128

    trans =Transformer()
    database = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        database.extend(filenames)
        break # For safety
    length = len(database)

    out_tensor = torch.FloatTensor(length, 2048).zero_()

    start_time = time.time()
    iter_end = int(np.floor(length/bs))
    print("Start to extract database features ")
    for idx_iter in range(iter_end+1):
        print(idx_iter,)
        if idx_iter==iter_end:
            current_bs_size = length-idx_iter*bs
        else:
            current_bs_size = bs
        batches = torch.FloatTensor(current_bs_size, 3, 224, 224).zero_()
        for idx_img in range(current_bs_size):
            img_path = os.path.join(path,database[idx_img+bs*idx_iter])
            img = Image.open(img_path).convert('RGB')
            img = trans(img)
            batches[idx_img] = img
        x = torch.autograd.Variable(batches, volatile=True)
        # compute output
        output = model.forward(x)
        if idx_iter==iter_end:
            out_tensor[idx_iter*bs:length] = output.data
        else:
            out_tensor[idx_iter * bs:idx_iter * bs + bs] = output.data

    database_features = out_tensor.cpu().numpy()

    np.save(npz_name,database_features)
    np.save("./features/database_fc.npy",database)
    # TIME consuming
    Tdiff = time.time()-start_time
    m, s = divmod(Tdiff, 60)
    h, m = divmod(m, 60)
    print('\nForward time for %d train images = %dh:%02dm:%0.2fs' % (length,h,m,s))

    return database, database_features

def Extract_one_feature(model,root,img_name):
    # Forward an image (test)

    start_time = time.time()
    # Prepare an input tensor with the input image
    img_path = os.path.join(root, img_name)
    img_origin = Image.open(img_path).convert('RGB')

    # geunho : autocontrast
    from PIL import ImageOps, ImageEnhance

    '''img = Image.open(img_path)
    enhancer = ImageEnhance.Contrast(img)
    enhancer.enhance(2.0)
    img.show()'''
    import cv2
    bgr = cv2.imread(img_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LUV) #Luv4.0=9, HLS1.0=1
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LUV2BGR)
    #cv2.imshow('Apply CLAHE to the converted image in LAB format to only Lightness component', bgr)
    #cv2.waitKey(0)
    cv2.imwrite('./EXAMPLE/temp/after_clahe.jpg', bgr)

    img_ = Image.open('./EXAMPLE/temp/after_clahe.jpg')
    img = ImageOps.autocontrast(img_, cutoff=1)
    #img.show()

    #for i in range(8):
    #   factor = i / 4.0
    #   enhancer.enhance(factor).show("Sharpness %f" % factor)


    # geunho : per-example normalization test!!!    ------------------------------------------------------------------------
    from statistics import mean
    average_color = [mean(img.getdata(band)) for band in range(3)]

    trans = Transformer()
    #trans = Transformer_perExample(average_color)
    img = trans(img) # original

                    #per-example normalization
    # geunho END ------------------------------------------------------------------------


    img_tensor = torch.FloatTensor(1, 3, 224, 224)
    img_tensor[0] = img
    x = torch.autograd.Variable(img_tensor, volatile=True)

    # compute output
    output = model.forward(x)
    feature = output.data.cpu().numpy()

    # #%% TIME consuming
    Tdiff = time.time() - start_time
    m, s = divmod(Tdiff, 60)
    h, m = divmod(m, 60)
    print('Forward time for %d test image = %dh:%02dm:%0.2fs' % (1, h, m, s))
    return feature


