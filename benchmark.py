import numpy as np
import os
import cv2
from pyciede2000 import ciede2000
from skimage import color


def mse(img , gt):
    imgh = img.shape[0]
    imgw = img.shape[1]

    res = (1/(imgh*imgw*3))*(np.sum((img-gt)**2))
    return res

def singlemse(img,gt,c):
    imgh = img.shape[0]
    imgw = img.shape[1]

    res = (1/(imgh*imgw))*(np.sum((img[:,:,c]-gt[:,:,c])**2))
    return res


def singlessim(img,gt,c):
    imgh = img.shape[0]
    imgw = img.shape[1]

    miuo=1/(imgh*imgw) *(np.sum(img[:,:,c]))
    miug=1/(imgh*imgw) *(np.sum(gt[:,:,c]))

    faio=((1/(imgh*imgw-1)) * np.sum((img[:,:,c] - miuo)**2)) ** (0.5)
    faig=((1/(imgh*imgw-1)) * np.sum((gt[:,:,c] - miug)**2)) ** (0.5)
    faiog=((1/(imgh*imgw-1)) * np.sum((gt[:,:,c] - miug)*(img[:,:,c] - miuo))) 

    log = (2*miuo*miug + (0.01 * 255)**2)/(miuo**2+miug**2+(0.01 * 255)**2)
    cog = (2*faio*faig + (0.03 * 255)**2)/(faio**2+faig**2+(0.03 * 255)**2)
    sog = (faiog + ((0.03 * 255)**2 )/2)/(faio*faig+ ((0.03 * 255)**2 )/2)

    return log*cog*sog


def getbench(filepath,gtpath):

    orifilepath=filepath
    origtpath=gtpath
    filepath = os.listdir(filepath)
    gtpath = os.listdir(gtpath)


    msesum =0
    psnrsum =0
    ssimsum = 0
    ciesum=0
    for fn in filepath:
        if fn not in gtpath:
            continue

        img = cv2.imread(os.path.join(orifilepath,fn))
        gt = cv2.imread(os.path.join(origtpath,fn))
        # if fn[-1]=="g":
        #     gt = cv2.imread(os.path.join(origtpath,fn[:11]+"GT.jpg"))
        # elif fn[-1]=="G":
        #     gt = cv2.imread(os.path.join(origtpath,fn[:11]+"GT.JPG"))
        # else:
        #     continue

        labimg = color.rgb2lab(img)
        labgt = color.rgb2lab(gt)
        cie = ciede2000((np.mean(labimg[:,:,0]),np.mean(labimg[:,:,1]),np.mean(labimg[:,:,2])) ,\
                         (np.mean(labgt[:,:,0]),np.mean(labgt[:,:,1]),np.mean(labgt[:,:,2])))['delta_E_00']

        ciesum += cie


        mseres = mse(img,gt)

        psnr = 0
        for c in range(0,3):
            psnr += 10 * np.log10(   (2**8 - 1)**2 /singlemse(img,gt,c) )
        
        psnr/=3

        ssim = 0
        for c in range(0,3):
            ssim += singlessim(img,gt,c)
        
        ssim/=3

        msesum+=mseres
        psnrsum+=psnr
        ssimsum+=ssim
        print(fn+"mse: "+str(mseres))
        print(fn+"psnr: "+str(psnr))
        print(fn+"ssim: "+str(ssim))
        print(fn+"cie: "+str(cie))
    msesum/=len(gtpath)
    psnrsum/=len(gtpath)
    ssimsum/=len(gtpath)
    ciesum/=len(gtpath)
    with open(os.path.join(orifilepath,"mse.txt"),'w') as f:
        f.write(str(msesum))

    with open(os.path.join(orifilepath,"psnr.txt"),'w') as f:
        f.write(str(psnrsum))

    with open(os.path.join(orifilepath,"ssim.txt"),'w') as f:
        f.write(str(ssimsum))

    with open(os.path.join(orifilepath,"cie.txt"),'w') as f:
        f.write(str(ciesum))




getbench('/remote-home/lqwang/darkchannel/thirdRESIDE','/remote-home/lqwang/DM2F-Net/data/RESIDE/original')