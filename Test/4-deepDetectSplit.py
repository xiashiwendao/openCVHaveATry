import numpy as np
import cv2

def update(val = 0):
    # disparity range is tuned for 'aloe' image pair
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio','disparity'))
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize','disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff','disparity'))
    
    print('computing disparity…')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    
    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)

def test_StereoSGBM():
    window_size = 5
    min_disp = 16
    num_disp = 192-min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400
    imgL = cv2.imread('images/color1_small.jpg')
    imgR = cv2.imread('images/color2_small.jpg')
    cv2.namedWindow('disparity')
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50,update)
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize,200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50,update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250,update)
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = uniquenessRatio,
        speckleRange = speckleRange,
        speckleWindowSize = speckleWindowSize,
        disp12MaxDiff = disp12MaxDiff,
        P1 = P1,
        P2 = P2
    )
    update()
    cv2.waitKey()

def grabcut_test():
    from matplotlib import pyplot as plt
    img = cv2.imread('images/statue_small.jpg')
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    # rect = (100,50,421,378)
    rect = (240,280,660,600)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    plt.subplot(121), plt.imshow(img)
    plt.title("grabcut"), plt.xticks([]), plt.yticks([])
    plt.subplot(122),
    plt.imshow(cv2.cvtColor(cv2.imread('images/statue_small.jpg'),
    cv2.COLOR_BGR2RGB))
    plt.title("original"), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    # test_StereoSGBM()
    grabcut_test()


