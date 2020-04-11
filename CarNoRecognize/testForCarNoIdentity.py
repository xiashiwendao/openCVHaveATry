import cv2
import numpy as np

def stretch(img):
    '''
    图像拉伸函数，增强对比度
    '''
    maxi=float(img.max()) # 最小灰度
    mini=float(img.min()) # 最大灰度
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]=(255/(maxi-mini)*img[i,j]-(255*mini)/(maxi-mini))
    
    return img

# 二值化只是针对单通道而言，所以这里如果是多通道（有颜色的），还是会保留颜色。
# 这个也是为什么图像处理很多时候需要先灰度化的原因
def dobinaryzation(img):
    '''
    二值化处理函数
    '''
    maxi=float(img.max())
    mini=float(img.min())
    
    x=maxi-((maxi-mini)/2)
    #二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret,thresh=cv2.threshold(img,x,255,cv2.THRESH_BINARY)
    #返回二值化后的黑白图像
    return thresh

if __name__ == "__main__":
    img=cv2.imread('images/0a943.PNG',cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_stretch = stretch(img)
    img_twovalue = dobinaryzation(img)
    cv2.imshow('raw image', img)
    cv2.imshow('stretch img', img_stretch)
    cv2.imshow('two value', img_twovalue)
    r=16
    h=w=r*2+1
    kernel=np.zeros((h,w),np.uint8)
    cv2.circle(kernel,(r,r),r,1,-1)
    openingimg=cv2.morphologyEx(img_stretch,cv2.MORPH_OPEN,kernel)
    strtimg=cv2.absdiff(img_stretch,openingimg)
    cv2.imshow('diff img', strtimg)
    img_twovalue_2 = dobinaryzation(strtimg)    
    cv2.imshow('two value_2', img_twovalue_2)
    
    cv2.waitKey()
    cv2.destroyAllWindows()