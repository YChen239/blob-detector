import numpy as np
import cv2


#convolve
def convolve(img, fil):
    h = fil.shape[0] // 2
    w = fil.shape[1] // 2
    imgp = np.pad(img, ((h, h), (w, w)), 'edge')
    conv=_convolve(imgp,fil)
    conv=abs(conv)


    return conv


def _convolve(img, fil):
    import cv2
    fil_heigh = fil.shape[0]
    fil_width = fil.shape[1]
    conv_heigh = img.shape[0]-fil_heigh+1
    conv_width = img.shape[1]-fil_width+1

    conv = np.zeros((conv_heigh, conv_width))

    for i in range(conv_heigh):
        for j in range(conv_width):  # 逐点相乘并求和得到每一个点
            conv[i][j] = wise_element_sum(img[i:i + fil_heigh, j:j + fil_width], fil)
    return conv


def wise_element_sum(img, fil):
    fil=FZ(fil)
    res = (img * fil).sum()
    if (res < -1):
        res = -1
    elif res > 1:
        res = 1
    return res

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))

#LoG fliter
def LoG(sigma):
    import scipy
    size=max(1,2*round(sigma*3)+1)
    center = (size -1)/ 2
    xv=np.linspace(-center,center,size)
    yv=np.linspace(-center,center,size)
    x,y=np.meshgrid(xv,yv)
    std2=pow(sigma,2)
    arg=-(pow(x,2)+pow(y,2))/(2*std2)
    h=np.exp(arg)
    h[h<(2**-52)*np.max(h)]=0
    sumh=np.sum(h)
    if sumh:
        h=h/sumh
    # calculate Laplacian
    h1=h*(pow(x,2)+pow(y,2)-2*std2)/pow(std2,2)
    gaus=h1-np.sum(h1)/pow(size,2)
    return gaus


#Dectect Blob
def detectBlobs( img_GrayScale, numScales, sigma, bShouldDownsample, scaleMultiplier, threshold ):
    import math
    h = img_GrayScale.shape[0]
    w = img_GrayScale.shape[1]

    #Generate various scales
    scaleSpace = generateScaleSpace(img_GrayScale, numScales, sigma, scaleMultiplier, bShouldDownsample)

    #2D NonMax Suprresion for Each Individiaul Scale
    scaleSpace_2D_NMS = np.zeros((h, w, numScales))
    for i in range(numScales):
        #r=int(math.floor(sigma*(scaleMultiplier**i)/3))
        scaleSpace_2D_NMS[:,:, i] = nms_2D(scaleSpace[:,:, i],3)

    scaleSpace_3D_NMS = nms_3D(scaleSpace_2D_NMS, scaleSpace, numScales)
    for m in range(numScales):
        for i in range(h):
            for j in range(w):
                if scaleSpace_3D_NMS[i,j,m]<=threshold[m]:
                    scaleSpace_3D_NMS[i, j, m]=0

    return scaleSpace_3D_NMS

def generateScaleSpace(img, numScales, sigma, scaleMultiplier, bShouldDownsample):
    import datetime
    h = img.shape[0]
    w = img.shape[1]
    scaleSpace = np.zeros((h,w ,numScales))
    starttime = datetime.datetime.now()
    if bShouldDownsample is 1:
        LoGKernel = LoG(sigma)
        LoGKernel = pow(sigma,2)*LoGKernel


        for i in range(numScales):
            import cv2
            #Downsample
            if i is 0:
                downsizedImg=img
            else:
                downsizedImg = cv2.resize(img,None, fx=1/(scaleMultiplier**i),fy=1/(scaleMultiplier**i), interpolation=cv2.INTER_CUBIC)

            filteredImage = convolve(downsizedImg, LoGKernel)
            filteredImage = pow(filteredImage,2)

             #Upsample
            reUpscaledImg = cv2.resize(filteredImage, (w,h), interpolation=cv2.INTER_CUBIC)
            scaleSpace[:,:,i] = reUpscaledImg

    else:
        for i in range(numScales):
            scaledSigma = sigma * (scaleMultiplier**i)
            LoGKernel = LoG(scaledSigma)
            LoGKernel = pow(scaledSigma, 2) * LoGKernel
            filteredImage = convolve(img, LoGKernel)
            filteredImage = pow(filteredImage, 2)
            scaleSpace[:, :, i] = filteredImage

    endtime = datetime.datetime.now()
    print (endtime - starttime)

    return scaleSpace

def nms_2D(img,size):
    #extract local maxima
    h = img.shape[0]
    w = img.shape[1]
    img1=np.zeros((h,w))
    for i in range(size,h-size):
        for j in range(size,w-size):
            img1[i][j]=np.max(img[i-size:i+size+1,j-size:j+size+1])
    return img1

def nms_3D(scaleSpace_2D_NMS, originalScaleSpace, numScales):
    h = scaleSpace_2D_NMS.shape[0]
    w = scaleSpace_2D_NMS.shape[1]
    maxVals_InNeighboringScaleSpace = np.zeros((h,w,numScales))
    originalValMarkers= np.zeros((h,w,numScales))
    for i in range(numScales):
        for m in range(h):
            for n in range(w):
                maxVals_InNeighboringScaleSpace[m, n, i] = np.max(scaleSpace_2D_NMS[m, n, 0: numScales-1])
                if maxVals_InNeighboringScaleSpace[m, n, i]==originalScaleSpace[m, n, i]:
                    originalValMarkers[m, n, i]=maxVals_InNeighboringScaleSpace[m, n, i]

    return originalValMarkers


#Draw circles
def calcRadiiByScale(numScales, scaleMultiplier, sigma):
    radiiByScale = np.zeros((1, numScales))
    for i in range(numScales):
        radiiByScale[0,i] = np.sqrt(2) * sigma * (scaleMultiplier**i) #calculate r
    return radiiByScale

def retrieveBlobMarkers(scaleSpace, radiiByScale,numScales):
    blobMarkers=[]
    for i in range(numScales):
        [newMarkerRows, newMarkerCols] = np.nonzero(scaleSpace[:, :, i])
        newMarkers = np.vstack((newMarkerRows, newMarkerCols))
        newMarkers = np.transpose(newMarkers)
        c = np.ones((np.size(newMarkerRows), 1))
        newMarkers = np.c_[newMarkers, c]
        newMarkers[:, 2] = radiiByScale[0, i]
        if i is 0:
            blobMarkers = newMarkers
        blobMarkers = np.vstack((blobMarkers, newMarkers))
    return blobMarkers

def show_all_circles(img_o, I, cx, cy, rad, name, flag):
    from matplotlib.patches import Ellipse, Circle
    import matplotlib.pyplot as plt
    theta = np.arange(0, 2 * np.pi+0.01, 0.01)
    x = np.tile(cx, (np.size(theta), 1))
    x = np.transpose(x)
    y = np.tile(cy, (np.size(theta), 1))
    y = np.transpose(y)
    r = np.tile(rad, (np.size(theta), 1))
    r = np.transpose(r)



    x_d = x + r * np.cos(theta)
    y_d = y + r * np.sin(theta)
    img_o = img_o[:, :, [2, 1, 0]]
    if flag is 1:
        plt.figure(figsize=(16,8))
        plt.subplot(1, 2, 1)
        plt.imshow(img_o)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(I, cmap='gray')
        plt.plot(np.transpose(y_d), np.transpose(x_d), color='red', linewidth=1)
        plt.axis('off')
        plt.savefig(name,dip=300)
        plt.show()
    else:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img_o)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(I, cmap='gray')
        plt.plot(np.transpose(y_d), np.transpose(x_d), color='red', linewidth=1)
        plt.axis('off')
        plt.show()


