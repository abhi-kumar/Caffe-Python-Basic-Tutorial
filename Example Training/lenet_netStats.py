import numpy as np
from matplotlib import pyplot as plt
caffe_root = '/home/test/Desktop/Abhi/caffe/caffe-master/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from pylab import *
import cv2

print dir(caffe)
from caffe import Layer as L
#from caffe import Param as P

caffe.set_mode_cpu()

DEPLOY_FILE ='lenet_deploy.prototxt'
net = caffe.Net(DEPLOY_FILE, caffe.TEST)

print "input dimensions = "
print net.blobs['data'].data.shape

print "\nDimensions after conv1 layer"
print net.blobs['conv1'].data.shape

print "\nDimensions after pool1 layer"
print net.blobs['pool1'].data.shape

print "\nDimensions after conv2 layer"
print net.blobs['conv2'].data.shape

print "\nDimensions after pool2 layer"
print net.blobs['pool2'].data.shape

print "\nDimensions after ip1 layer"
print net.blobs['ip1'].data.shape

print "\nDimensions after ip2 layer"
print net.blobs['ip2'].data.shape

print "\nDimensions after prob layer"
print net.blobs['prob'].data.shape


print "\n\nconv1 Layer Dimensions"
print net.params['conv1'][0].data.shape

print "\nconv2 Layer Dimensions"
print net.params['conv2'][0].data.shape

print "\nip1 Layer Dimensions"
print net.params['ip1'][0].data.shape

print "\nip2 Layer Dimensions"
print net.params['ip2'][0].data.shape


#to print all blob dimensions at once uncomment line below
#print [(k, v.data.shape) for k, v in net.blobs.items()]

#to print all layer dimensions at once uncomment line below
#print [(k, v[0].data.shape) for k, v in net.params.items()]








