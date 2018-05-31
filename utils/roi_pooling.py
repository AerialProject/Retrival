#import torch.autograd as ag
#import torch
#from torch.nn import AdaptiveMaxPool2d

import numpy as np
from sklearn import preprocessing
import skimage.measure

#def adaptive_max_pool_temp(input, size):
#    '''geunho : 이부분을 손봐야하겠다.'''
#    return AdaptiveMaxPool2d((size[0],size[1]))(input)

def adaptive_max_pool(input, size): # size is original input size (not downsize!)
    return skimage.measure.block_reduce(input, (1,size[1],size[2],1), np.max)


def roi_pooling(input, rois, size=(1, 1, 14, 14), spatial_scale=1.0):
    output = []
    num_rois = np.shape(rois)[0]

    roisa = rois * spatial_scale
    rois[:, 1:] = rois[:, 1:] * spatial_scale # FIXME

    #rois = rois.long()
    rois = np.int_(rois)
    for i in range(num_rois):
        roi = rois[i]
        #im_idx = roi[0]
        #im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        im = input[:, roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1), :]
        size = np.shape(im) #(im.size, 1, 14, 14)
        temp = adaptive_max_pool(im, size)
        temp1 = np.reshape(temp, (1, 2048))
        output.append(temp1) # FIXME
        #output[i].data = output[i].data.view(1,-1) # FIXME

    return output


def r_mac(input, rois, spatial_scale=1/32.0):
    temp = roi_pooling(input, rois=rois, spatial_scale=spatial_scale)
    temp1 = np.reshape(temp, (30, 2048))
    roi_pooled_l2 = preprocessing.normalize(temp1) #(roi_pooled)
    summed = np.sum(roi_pooled_l2, axis=0)
    summed_l2 = preprocessing.normalize(summed.reshape(1,-1))

    return summed_l2

'''
if __name__ == "__main__":
    input = ag.Variable(torch.rand(1, 3, 10, 10), requires_grad=True)
    rois = ag.Variable(torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8]]), requires_grad=False)

    out1 = adaptive_max_pool(input, (1, 1))
    out2 = roi_pooling(input, rois, size=(1, 1))


    from sklearn import preprocessing
    import numpy as np

    out = preprocessing.normalize(out2.data.numpy())
    print(out)
    print(np.linalg.norm(out[1]))
    print(np.sum(out,axis=0))
'''





