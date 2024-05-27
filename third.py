import numpy as np
import cv2
import os
import time

def firstproc(img):
    #白平衡算法公式参考了：https://www.cnblogs.com/pear-linzhu/p/12453985.html 
    img = img.copy()
    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]
    r = np.float64(r)
    g = np.float64(g)
    b = np.float64(b)
    gray = 0.299*r+0.587*g+0.114*b
    gray_av = np.average(gray)
    ra=np.mean(r)
    ga=np.mean(g)
    ba=np.mean(b)
    r*= (gray_av/ra)
    g*= (gray_av/ga)
    b*= (gray_av/ba)
    res = np.zeros_like(img,dtype=np.float64)
    res[:,:,0]=b
    res[:,:,1]=g
    res[:,:,2]=r
    res = np.float64(res)
    res = np.min(res,axis=2)
    return res

def medianfilter(img,ks):
    ori=img.copy()
    temp=np.zeros((img.shape[0]+(ks-1),img.shape[1]+(ks-1)))
    half=int((ks-1)/2)
    for i in range(half,img.shape[0]+half):
        for j in range(half,img.shape[1]+half):
            temp[i,j]=np.median(ori[i-half:i+half,j-half:j+half])
    
    res = temp[half:img.shape[0]+half,half:img.shape[1]+half].copy()
    return res
#自己实现的中值滤波器，速度很慢，对于DHAZY的1262_.bmp，用cv2的medianblur只需要0.26s，而用自己的medianblur要32s！！

def start(filepath,outputfile):
    orifilepath=filepath
    filepath = os.listdir(filepath)

    timesum =0
    for fn in filepath:
        st = time.time()
        img = cv2.imread(os.path.join(orifilepath,fn))
        ori = img.copy()
        whiteb = firstproc(img)
        tempw = np.uint8(whiteb)
        A = cv2.medianBlur(tempw,41)
        # A = medianfilter(tempw,41)
        temp = np.abs(whiteb-A)
        temp = np.uint8(temp)
        B=A-cv2.medianBlur(temp,41)
        # B=A-medianfilter(temp,41)
        B=np.float64(B)
        B*=0.95  
        temp = np.zeros((B.shape[0],B.shape[1],3))
        temp[:,:,0]=B
        temp[:,:,1]=np.float64(whiteb)
        temp[:,:,2]=255
        V= np.min(temp,axis=2)
        V[V<0]=0
        V=np.uint8(V)
        V=np.expand_dims(V,axis=-1)
        ori=np.float64(ori)
        V=np.float64(V)
        temp = ori -V
        temp[temp<0.0]=0.0
        temp[temp>255.0]=255.0
        res = np.uint8(temp) / (1 - V/255)
        res = np.uint8(res)
        et = time.time()
        timesum+=(et-st)
        print(fn+"is ok!"+"time is: "+str(et-st))
        cv2.imwrite(os.path.join(outputfile,fn),res )   
    
    timesum /=len(filepath)
    with open(os.path.join(outputfile,"time.txt"),'w') as f:
        f.write(str(timesum))





start('/remote-home/lqwang/DHAZE-ANCIENT/DHAZY/img','/remote-home/lqwang/darkchannel/test')