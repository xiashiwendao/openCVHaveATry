import cv2
import numpy as np
# 识别角
def recognize_corner():
    img = cv2.imread('images/chessboard.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 29, 0.04)
    img[dst>0.01 * dst.max()] = [0, 0, 255]
    while (True):
        cv2.imshow('corners', img)
        if cv2.waitKey():
            break

    cv2.destroyAllWindows()

import cv2
import sys
import numpy as np

def sift_corner():
        
    imgpath = 'images\\highfloor.jpg'
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(gray,None)
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints = keypoints,
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (51, 163, 236))
    cv2.imshow('sift_keypoints', img)
    if cv2.waitKey():
        cv2.destroyAllWindows()

def fd(algorithm):
    if algorithm == "SIFT":
        return cv2.xfeatures2d.SIFT_create()
    if algorithm == "SURF":
        return cv2.xfeatures2d.SURF_create(float(sys.argv[3]) if len(sys.argv)== 4 else 4000)

def surf_sift(alg):
    imgpath = 'images\\highfloor.jpg'
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd_alg = fd(alg)
    keypoints, descriptor = fd_alg.detectAndCompute(gray,None)
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints = keypoints, flags = 4, color = (51, 163, 236))
    cv2.imshow('keypoints', img)
    while (True):
        if cv2.waitKey():
            break
    cv2.destroyAllWindows()

def surf_compare(alg):
    img1 = cv2.imread('images/manowar_logo.png',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/manowar_single.jpg', cv2.IMREAD_GRAYSCALE)
    fd_alg = fd(alg)
    kp1, des1 = fd_alg.detectAndCompute(img1,None)
    kp2, des2 = fd_alg.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    print('+++++ des1: ', des1)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:40], img2,flags=2)
    plt.imshow(img3)
    plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt

def org_compare():
    # img1 = cv2.imread('images/manowar_logo.png',cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('images/manowar_single.jpg', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('images/cocacola2.jpg',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/cocacola_bottle2.jpg', cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:40], img2,flags=2)
    plt.imshow(img3)
    plt.show()

def org_feature(alg):
    # imgpath = 'images\\highfloor.jpg'
    # img = cv2.imread(imgpath)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.imread('images/manowar_logo.png',cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('images/manowar_single.jpg', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('images/cocacola.jpg',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/cocacola_bottle2.jpg', cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    fd_alg = fd(alg)
    img = cv2.drawKeypoints(image=img1, outImage=img1, keypoints = kp1, flags = 4, color = (51, 163, 236))
    cv2.imshow('keypoints', img)
    while (True):
        if cv2.waitKey():
            break
    cv2.destroyAllWindows()

def flann_detect():
    # queryImage = cv2.imread('images/bathory_album.jpg',0)
    # trainingImage = cv2.imread('images/vinyls.jpg',0)
    trainingImage = cv2.imread('images/cocacola_bottle2.jpg',0)
    queryImage = cv2.imread('images/cocacola2.jpg',0)
    
    # create SIFT and detect/compute
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(queryImage,None)
    kp2, des2 = sift.detectAndCompute(trainingImage,None)
    # FLANN matcher parameters
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    searchParams = dict(checks=50) # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(indexParams,searchParams)
    matches = flann.knnMatch(des1,des2,k=2)
    # prepare an empty mask to draw good matches
    matchesMask = [[0,0] for i in range(len(matches))]
    # David G. Lowe's ratio test, populate the mask
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    drawParams = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)
    resultImage =cv2.drawMatchesKnn(queryImage,kp1,trainingImage,kp2,matches,None,**drawParams)
    
    plt.imshow(resultImage,), plt.show()



if __name__ == "__main__":
    # sift_corner()
    # surf_sift('SURF')
    org_compare()
    # surf_compare('SURF')
    # org_feature('SURF')
    # flann_detect()
