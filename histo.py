import cv2  
import numpy as np  
import os
import time


def haha(imgori):
    img = imgori.flatten()
    img,_ = np.histogram(img,256,[0,256])   
    temp = 0
    for (i, pix) in enumerate(img):
        temp += img[i]
        img[i] = temp
    coff = 255/(img.max()-img.min())
    img = (img-img.min())* coff
    result = img[imgori]  

    return result


def start(filepath,outputfile):

    orifilepath=filepath
    filepath = os.listdir(filepath)

    timesum =0
    for fn in filepath:
        
        st = time.time()
        img = cv2.imread(os.path.join(orifilepath,fn))
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        result = np.zeros_like(img)
        result[:,:,0]=haha(r)
        result[:,:,1]=haha(g)
        result[:,:,2]=haha(b)

        et = time.time()
        
        timesum+=(et-st)
        print(fn+"is ok!"+"time is: "+str(et-st))
        cv2.imwrite(os.path.join(outputfile,fn),result )   

    timesum /=len(filepath)
    with open(os.path.join(outputfile,"time.txt"),'w') as f:
        f.write(str(timesum))

start('/remote-home/lqwang/darkchannel/DHAZY/img','/remote-home/lqwang/darkchannel/histoDHAZY')