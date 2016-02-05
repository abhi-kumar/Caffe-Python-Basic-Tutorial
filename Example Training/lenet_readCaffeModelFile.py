import google.protobuf
import numpy as np
from matplotlib import pyplot as plt
caffe_root = '/home/abhi/caffe/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from pylab import *

from caffe.proto import caffe_pb2
import struct
print dir(caffe)
from caffe import Layer as L
net = caffe_pb2.NetParameter()
f = open("lenet_iter_5000.caffemodel","rb")

net.ParseFromString(f.read())

print "\n"
#size = net.layer.__len__
length = size(net.layer)


layers = "";
for i in range(0,length):
	print i
	layers += str(net.layer[i]) 

f = open('lenet_iter_5000_caffemodel.txt','w')
f.write(layers)
f.close()

