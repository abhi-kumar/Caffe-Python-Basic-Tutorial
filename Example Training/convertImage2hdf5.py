import h5py, os
import numpy as np
import cv2

f = h5py.File('train.h5', 'w')
# 10 images, each is a 3*28*28-dim vector
f.create_dataset('data', (10, 28*28*3), dtype='f8')
# Data's labels, each is a 1-dim vector
f.create_dataset('label', (10, 1), dtype='f4')

images = open('Train.txt', 'r')


# Fill in something with fixed pattern
# Regularize values to between 0 and 1, or SigmoidCrossEntropyLoss will not work
for i in range(10):
	a = np.empty(28*28*3)
	content = images.readline()
	paths,labels = content.split()
	print paths
	img = cv2.imread(paths,1)
	img.reshape(1,28*28*3)
	b = np.asarray(img) 
	c = b.reshape(1,28*28*3)
	a = c
#	print int(labels)
	f['data'][i] = a
	f['label'][i] = int(labels)

f.close()
images.close()
