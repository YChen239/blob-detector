import matplotlib.pyplot as plt
import pylab
import numpy as np
import cv2
import LaplacianBlob


#Input image and Convert it to gray scale as double
img_o = cv2.imread('butterfly.jpg')
img=img_o[:,:,1]
img = img / np.max(img)
saveName = 'butterfly.png'
flag = 1
downSample = 1


#Define parameters
numScales =9
sigma = 2
scaleMultiplier = np.sqrt(2)
threshold = np.linspace(0.008, 0.008, numScales)


#Detect blobs
scaleSpace_3D_NMS = LaplacianBlob.detectBlobs(img, numScales, sigma, downSample, scaleMultiplier, threshold )

#Draw circles
radiiByScale = LaplacianBlob.calcRadiiByScale(numScales, scaleMultiplier, sigma)
blobMarkers = LaplacianBlob.retrieveBlobMarkers(scaleSpace_3D_NMS, radiiByScale,numScales)

xPos = blobMarkers[:, 0]
yPos = blobMarkers[:, 1]
radii = blobMarkers[:, 2]


LaplacianBlob.show_all_circles(img_o, img, xPos, yPos, radii, saveName, flag)

