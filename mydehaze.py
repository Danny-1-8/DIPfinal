import numpy as np
import cv2
import os
import time

def darkch(img):
    
    # img = np.min(img,axis=2)  #尺寸为（*，*）
    # resimg = np.zeros_like(img)
    # size1 = img.shape[0]
    # size2 = img.shape[1]
    # for i in range(0,size1):
    #     for j in range(0,size2):
    #         resimg[i,j] = np.min(img[((i-7) if (i-7)>=0 else 0):\
    #                               ((i+7) if (i+7)<=size1-1 else size1-1),\
    #                                 ((j-7) if (j-7)>=0 else 0):\
    #                               ((j+7) if (j+7)<=size2-1 else size2-1)])     #用numpy实现的速度很慢！！！！DHAZY 1262_要7.36s!!
    # return resimg

    #这个函数引用了https://github.com/He-Zhang/image_dehaze/blob/master/dehaze.py       DHAZY 1262_只需0.86s   
    img = np.min(img,axis=2)  
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    img = cv2.erode(img,element)
    return img

def boxFilter(img,r):

    #自己实现的box filter非常慢，只能调用cv2的boxfilter！ 但效果是一样的    对于DHAZY的1262_.bmp，需要35s！！
    resimg = np.zeros_like(img)
    size1 = img.shape[0]
    size2 = img.shape[1]
    for i in range(0,size1):
        for j in range(0,size2):
            resimg[i,j] = np.mean(img[((i-7) if (i-7)>=0 else 0):\
                                  ((i+7) if (i+7)<=size1-1 else size1-1),\
                                    ((j-7) if (j-7)>=0 else 0):\
                                  ((j+7) if (j+7)<=size2-1 else size2-1)])    
            a=1
    return resimg 




#何凯明guided filter原文代码，cv2.boxfilter的选择参考了https://github.com/He-Zhang/image_dehaze/blob/master/dehaze.py
def gfilter(p, I, r, e):

    meanI=cv2.boxFilter(I,cv2.CV_64F, (r,r))
    meanp= cv2.boxFilter(p,cv2.CV_64F, (r, r))
    corrI= cv2.boxFilter(I*I,cv2.CV_64F, (r,r))
    corrIp=cv2.boxFilter(I*p,cv2.CV_64F, (r, r))
    # meanI=boxFilter(I,(r, r))
    # meanp=boxFilter(p,(r, r))
    # corrI=boxFilter(I*I,(r, r))
    # corrIp=boxFilter(I*p,(r, r))
    varI= corrI-meanI*meanI
    covIp= corrIp-meanI*meanp
    a =covIp/(varI+e)
    b=meanp-a*meanI
    meana =cv2.boxFilter(a, cv2.CV_64F,(r,r))
    meanb=cv2.boxFilter(b, cv2.CV_64F, (r,r))
    # meana =boxFilter(a, (r,r))
    # meanb=boxFilter(b, (r,r))
    q =meana*I +meanb
    return q

def start(filepath,outputfile):

    orifilepath=filepath
    filepath = os.listdir(filepath)

    timesum =0
    for fn in filepath:
        st = time.time()
        img = cv2.imread(os.path.join(orifilepath,fn))
        dark_im = darkch(img)
        guide_img = np.dot(img[..., ::-1], [0.299, 0.587, 0.114])
        img = np.float64(img)
        dark_im = np.float64(dark_im)
        guide_img = np.float64(guide_img)
        tempsize = dark_im.shape[0]*dark_im.shape[1]
        dark_im=dark_im.reshape(tempsize)
        samples = dark_im.argsort()[::-1]
        samples = samples[:int(0.001*tempsize)]
        img=img.reshape(img.shape[0]*img.shape[1],3)
        atmos_light = np.zeros(3,)
        atmos_light[0] = np.max(img[samples][:,0])
        atmos_light[1] = np.max(img[samples][:,1])
        atmos_light[2] = np.max(img[samples][:,2])
        tx = darkch(img/atmos_light)
        transm_map = 1 - 0.95*tx   
        #在何恺明的原文里使用的是Soft Matting来对map进行滤波，但我实在看不懂原文是怎么实现的；
        #在网上调研后发现大多数人在复现代码时都采用的是guided filter，简单且高效，所以我也采用这种方法
        transm_map = gfilter(transm_map, guide_img, 20, 0.0001) 
        res = (img-atmos_light)/np.expand_dims(np.where(transm_map>0.26,transm_map,0.26),axis=2) +atmos_light 
        et = time.time()
        timesum+=(et-st)
        print(fn+"is ok!"+"time is: "+str(et-st))
        cv2.imwrite(os.path.join(outputfile,fn),res )   

    timesum /=len(filepath)
    with open(os.path.join(outputfile,"time.txt"),'w') as f:
        f.write(str(timesum))





start('/remote-home/lqwang/DHAZE-ANCIENT/DHAZY/img','/remote-home/lqwang/darkchannel/test2')
