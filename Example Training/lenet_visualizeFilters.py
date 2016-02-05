#all the arrays are numpy arrays here
import numpy as np			

#For plotting filters & filtered images	
import matplotlib.pyplot as plt

#Set this as the root folder of where caffe in installed
caffe_root = '/home/test/Desktop/Abhi/caffe/caffe-master/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2

#For pyplot to be stable
from pylab import *

#If GPU is not available
caffe.set_mode_cpu()


# deploy file reference /caffe/models/bvlc_alexnet/deploy.prototxt
DEPLOY_FILE ='lenet_deploy.prototxt'
MODEL_FILE = 'lenet_iter_10000.caffemodel'

#Initialize the net
net = caffe.Net(DEPLOY_FILE, MODEL_FILE, caffe.TEST)


#Transforming image to suit the need of input as a blob
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#If mean image exists uncomment the line below and add appropriate mean file
#transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('Dataset/Test/55002.png'))



#Go for one forward step
out = net.forward()
out = net.forward()


#Function to map all the filters and the filtered image on a single plot
def vis_square(filename, data, padsize=1, padval=0):
#    data -= data.min()
#    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
#    plt.show()
    plt.savefig(filename)

#If the layer has associated filter
#filters = net.params['conv2'][0].data
#vis_square('dffsd',filters.transpose(0, 2, 3, 1))

#To print set of filtered images after it passes through the layer. data[0, :x] x is the number of channels you want to print.
#feat = net.blobs['conv1'].data[0, :20]
#vis_square(feat, padval=1)


#Gives all the functions that the layer has
#print dir(net.blobs['eltwise'].data)


for k, v in net.blobs.items():
	if(k=='ip1'):
		break
	feat = net.blobs[k].data[0, :36]
	fileN = 'lenet_blob_' + k + '.png'
	vis_square(fileN, feat, padval=1)
	print fileN

num = 0;
for k, v in net.params.items():
	if(k[0]=='c' and num==0):
		fileN = 'lenet_filter_' + k + '.png'
		print fileN
		filters = net.params[k][0].data
		print filters.size
		vis_square(fileN, filters.transpose(0, 2, 3, 1))
		num = num+1
	elif(k[0]=='c' and num>0):
		fileN = 'lenet_filter_' + k + '.png'
		print fileN
		filters = net.params[k][0].data
		print filters[:10].size
		vis_square(fileN, filters[:10].reshape(50*4, 5, 5)) #seeing just 200 filter channels
		num = num+1
		


