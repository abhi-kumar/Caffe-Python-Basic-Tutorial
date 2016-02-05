import numpy as np
from matplotlib import pyplot as plt
caffe_root = '/home/abhi/Desktop/Thesis/caffe/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from pylab import *

print dir(caffe)
from caffe import Layer as L
#from caffe import Param as P

caffe.set_mode_cpu()

# caffemodel file
MODEL_FILE ='lenet_iter_10000.caffemodel'
# deploy file reference /caffe/models/bvlc_alexnet/deploy.prototxt
DEPLOY_FILE ='lenet_deploy.prototxt'
net = caffe.Net(DEPLOY_FILE, MODEL_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_raw_scale('data',255)
#transformer.set_channel_swap('data',(2,1,0))
#net.blobs['data'].reshape(64,1,28,28)

img = caffe.io.load_image ('Dataset/Test/55002.png')
net.blobs['data'].data[...] = transformer.preprocess('data', img)

out = net.forward ()
# output probability distribution for each possible classification
predicts = out['prob']

predict = predicts.argmax()

print "Predicted label:"
print predict

