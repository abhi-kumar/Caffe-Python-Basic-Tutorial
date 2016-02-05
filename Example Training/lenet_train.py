#all the arrays are numpy arrays here
import numpy as np			

#For plotting filters & filtered images	
import matplotlib.pyplot as plt

#Set this as the root folder of where caffe in installed
caffe_root = '/home/abhi/caffe/caffe/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2

#For pyplot to be stable
from pylab import *

#If GPU is not available
caffe.set_mode_cpu()


#This takes the solver file
solver = caffe.get_solver('lenet_solver.prototxt')
#This makes the training move one step ahead
#solver.step(1)


#To do the full training uncomment next line
solver.solve()




