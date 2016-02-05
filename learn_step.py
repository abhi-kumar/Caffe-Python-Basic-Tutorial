import numpy as np
from matplotlib import pyplot as plt
caffe_root = '/home/abhi/caffe/caffe/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from pylab import *
import cv2

caffe.set_mode_cpu()

# deploy file reference /caffe/models/bvlc_alexnet/deploy.prototxt
DEPLOY_FILE ='absValLayer_example.prototxt'

net = caffe.Net(DEPLOY_FILE, caffe.TEST)

print net.blobs['data'].data.shape
print net.blobs['abs_val'].data.shape

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_raw_scale('data',255)
img = caffe.io.load_image ('road.jpg',1)
net.blobs['data'].data[...] = transformer.preprocess('data', img)
net.forward()


def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.savefig('absValLayer.png', data)



feat = net.blobs['abs_val'].data[0, :1]
vis_square(feat, padval=1)





