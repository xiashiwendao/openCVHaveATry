# coding=utf-8
import sklearn
import cv2
import numpy as np
import matplotlib.pyplot as plt

###############################
######  theme: 车牌识别   ######
######   author: 行歌     ######
######   time: 2018.3.23  ######
################################

# TODO 增加旋转
# TODO 是否可以用手写集来作为文字识别？
PIXEL_SIZE = 400
def imread_photo(filename, flags=cv2.IMREAD_COLOR):
    """
    该函数能够读取磁盘中的图片文件，默认以彩色图像的方式进行读取
    输入： filename 指的图像文件名（可以包括路径）
          flags用来表示按照什么方式读取图片，有以下选择（默认采用彩色图像的方式）：
              IMREAD_COLOR 彩色图像
              IMREAD_GRAYSCALE 灰度图像
              IMREAD_ANYCOLOR 任意图像
    输出: 返回图片的通道矩阵
    """
    return cv2.imread(filename, flags)





def resize_photo(imgArr, MAX_WIDTH=1000):
    """
    这个函数的作用就是来调整图像的尺寸大小，当输入图像尺寸的宽度大于阈值（默认1000），我们会将图像按比例缩小
    输入： imgArr是输入的图像数字矩阵
    输出:  经过调整后的图像数字矩阵
    拓展：OpenCV自带的cv2.resize()函数可以实现放大与缩小，函数声明如下：
            cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
        其参数解释如下：
            src 输入图像矩阵
            dsize 二元元祖（宽，高），即输出图像的大小
            dst 输出图像矩阵
            fx 在水平方向上缩放比例，默认值为0
            fy 在垂直方向上缩放比例，默认值为0
            interpolation 插值法，如INTER_NEAREST，INTER_LINEAR，INTER_AREA，INTER_CUBIC，INTER_LANCZOS4等            
    """
    img = imgArr
    rows, cols = img.shape[:2]  # 获取输入图像的高和宽
    if cols > MAX_WIDTH:
        change_rate = MAX_WIDTH / cols  # 这里实现的比较精妙，这样实现了宽高同步进行缩放
        img = cv2.resize(img, (MAX_WIDTH, int(rows * change_rate)),
                         interpolation=cv2.INTER_AREA)
    return img

def predict(imageArr):
    """
    这个函数通过一系列的处理，找到可能是车牌的一些矩形区域
    输入： imageArr是原始图像的数字矩阵
    输出：gray_img_原始图像经过高斯平滑后的二值图
          contours是找到的多个轮廓
    """
    img_copy = imageArr.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_img_ = cv2.GaussianBlur(gray_img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    kernel = np.ones((23, 23), np.uint8)
    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)
    # 找到图像边缘
    ret, img_thresh = cv2.threshold(
        img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((10, 10), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(
        img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return gray_img_, contours

def chose_licence_plate(contours, Min_Area=2000):
    """
    这个函数根据车牌的一些物理特征（面积等）对所得的矩形进行过滤
    输入：contours是一个包含多个轮廓的列表，其中列表中的每一个元素是一个N*1*2的三维数组
    输出：返回经过过滤后的轮廓集合

    拓展：
    （1） OpenCV自带的cv2.contourArea()函数可以实现计算点集（轮廓）所围区域的面积，函数声明如下：
            contourArea(contour[, oriented]) -> retval
        其中参数解释如下：
            contour代表输入点集，此点集形式是一个n*2的二维ndarray或者n*1*2的三维ndarray
            retval 表示点集（轮廓）所围区域的面积
    （2） OpenCV自带的cv2.minAreaRect()函数可以计算出点集的最小外包旋转矩形，函数声明如下：
             minAreaRect(points) -> retval      
        其中参数解释如下：
            points表示输入的点集，如果使用的是Opencv 2.X,则输入点集有两种形式：一是N*2的二维ndarray，其数据类型只能为 int32
                                    或者float32， 即每一行代表一个点；二是N*1*2的三维ndarray，其数据类型只能为int32或者float32
            retval是一个由三个元素组成的元组，依次代表旋转矩形的中心点坐标、尺寸和旋转角度（根据中心坐标、尺寸和旋转角度
                                    可以确定一个旋转矩形）
    （3） OpenCV自带的cv2.boxPoints()函数可以根据旋转矩形的中心的坐标、尺寸和旋转角度，计算出旋转矩形的四个顶点，函数声明如下：
             boxPoints(box[, points]) -> points
        其中参数解释如下：
            box是旋转矩形的三个属性值，通常用一个元组表示，如（（3.0，5.0），（8.0，4.0），-60）
            points是返回的四个顶点，所返回的四个顶点是4行2列、数据类型为float32的ndarray，每一行代表一个顶点坐标              
    """
    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > Min_Area:
            temp_contours.append(contour)
    car_plate = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        rect_width, rect_height = rect_tupple[1]
        # 这里长宽互换，感觉上是要保证宽度一定要大于高度，可能和车牌本身的性质有关系，但是这样交换不会有问题吗？
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间（从比例上面来考虑，这个点很好！）
        if aspect_ratio > 2 and aspect_ratio < 5.5:
            car_plate.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
            rect_vertices = np.int0(rect_vertices)
    return car_plate

def find_rectangle(contour):
    '''
    寻找矩形轮廓
    '''
    y,x=[],[]
    
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    
    return [min(y),min(x),max(y),max(x)]

def locate_license(car_plates, image_raw):
    '''
    定位车牌号
    '''
    #找出最大的三个区域
    block=[]
    for c in car_plates:
        #找出轮廓的左上点和右下点，由此计算它的面积和长度比
        r=find_rectangle(c)
        a=(r[2]-r[0])*(r[3]-r[1])   #面积
        s=(r[2]-r[0])*(r[3]-r[1])   #长度比
        
        block.append([r,a,s])
    #选出面积最大的3个区域
    # if(len(block) > 3):
    #     block=sorted(block,key=lambda b: b[1])[-3:]
    # else:
    #     block=sorted(block,key=lambda b: b[1])
    #使用颜色识别判断找出最像车牌的区域
    maxweight,maxindex=0,-1
    for i in range(len(block)):
        b=image_raw[block[i][0][1]:block[i][0][3],block[i][0][0]:block[i][0][2]]
        #BGR转HSV
        hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
        #蓝色车牌的范围
        lower=np.array([100,50,50])
        upper=np.array([140,255,255])
        #根据阈值构建掩膜
        mask=cv2.inRange(hsv,lower,upper)
        #统计权值
        w1=0
        for m in mask:
            w1+=m/255
        
        w2=0
        for n in w1:
            w2+=n
            
        #选出最大权值的区域
        if w2>maxweight:
            maxindex=i
            maxweight=w2

    car_plate = car_plates[maxindex]
    row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
    row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
    cv2.rectangle(img, (row_min, col_min),
                    (row_max, col_max), (0, 255, 0), 2)
    card_img = img[col_min:col_max, row_min:row_max, :]
    cv2.imwrite("card_img.jpg", card_img)
    # cv2.imshow("raw_img_with_best_rec", img)        
    # cv2.imshow("card_img.jpg", card_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return "card_img.jpg"   

def license_segment(car_plates):
    print('len(car_plates): ', len(car_plates))
    """
    此函数根据得到的车牌定位，将车牌从原始图像中截取出来，并存在当前目录中。
    输入： car_plates是经过初步筛选之后的车牌轮廓的点集 
    输出:   "card_img.jpg"是车牌的存储名字
    """
    # 经过前面的处理，应该只是保留一个车牌识别区域
    # if len(car_plates) == 1:
    for car_plate in car_plates:
        # 这里car_plate怎么是三维的？为啥操作对象是第二个维度？
        row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
        row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
        cv2.rectangle(img, (row_min, col_min),
                        (row_max, col_max), (0, 255, 0), 2)
        card_img = img[col_min:col_max, row_min:row_max, :]
        # cv2.imshow("img", img)
    # else:
    #     print('+++++ error: for more then distriction return +++++')
    cv2.imwrite("card_img.jpg", card_img)
    # cv2.imshow("card_img.jpg", card_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return "card_img.jpg"


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    '''根据设定的阈值和图片直方图，找出波峰，用于分隔字符'''
    up_point = -1  # 上升点
    is_peak = False  # 是否为峰值范围
    # 分析直方图首个元素，和阈值进行比较，作为后面比较的基准
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            # 距离上次发现up point多于两个点，则认为最近那次up point是峰值
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            # 当前不是峰值范围，并且当前点高于阈值，说明当前点事上升点；
            # 那么，认为当前点可能就是峰值
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    
    return wave_peaks

# 实际测试中发现把一半的字符都切掉了
def remove_plate_upanddown_border(card_img):
    """
    这个函数将截取到的车牌照片转化为灰度图，然后去除车牌的上下无用的边缘部分，确定上下边框
    输入： card_img是从原始图片中分割出的车牌照片
    输出: 在高度上缩小后的字符二值图片
    """
    plate_Arr = cv2.imread(card_img)
    plate_gray_Arr = cv2.cvtColor(plate_Arr, cv2.COLOR_BGR2GRAY)
    # 获得（黑白）二分图
    ret, plate_binary_img = cv2.threshold(
        plate_gray_Arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和（直方图）
    # row_min = np.min(row_histogram)
    # row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    # row_threshold = (row_min + row_average) / 2
    # wave_peaks = find_waves(row_threshold, row_histogram) # 峰值点
    # # 接下来挑选跨度最大的波峰
    # wave_span = 0.0
    # for wave_peak in wave_peaks:
    #     span = wave_peak[1]-wave_peak[0]
    #     if span > wave_span:
    #         wave_span = span
    #         selected_wave = wave_peak
    # # ？这里的selected_wavep[0]和selected_wave[1]各自都是什么意思？
    # plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    #cv2.imshow("plate_binary_img", plate_binary_img)

    return plate_binary_img

    ##################################################
    # 测试用
    # print( row_histogram )
    # fig = plt.figure()
    # plt.hist( row_histogram )
    # plt.show()
    # 其中row_histogram是一个列表，列表当中的每一个元素是车牌二值图像每一行的灰度值之和，列表的长度等于二值图像的高度
    # 认为在高度方向，跨度最大的波峰为车牌区域
    # cv2.imshow("plate_gray_Arr", plate_binary_img[selected_wave[0]:selected_wave[1], :])
    ##################################################


 #####################二分-K均值聚类算法############################


def distEclud(vecA, vecB):
    """
    计算两个坐标向量之间的街区距离 
    """
    return np.sum(abs(vecA - vecB))

# KMeans算法里面的随机获取质心
def randCent(dataSet, k):
    n = dataSet.shape[1]  # 列数
    centroids = np.zeros((k, n))  # 用来保存k个类的质心
    for j in range(n):
        minJ = np.min(dataSet[:, j], axis=0)
        rangeJ = float(np.max(dataSet[:, j])) - minJ
        for i in range(k):
            centroids[i:, j] = minJ + rangeJ * (i+1)/k
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    # 这个簇分配结果矩阵包含两列，一列记录簇索引值，第二列存储误差。这里的误差是指当前点到簇质心的街区距离
    clusterAssment = np.zeros((m, 2))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0] == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    这个函数首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，
    选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。
    输入：dataSet是一个ndarray形式的输入数据集 
          k是用户指定的聚类后的簇的数目
         distMeas是距离计算函数
    输出:  centList是一个包含类质心的列表，其中有k个元素，每个元素是一个元组形式的质心坐标
            clusterAssment是一个数组，第一列对应输入数据集中的每一行样本属于哪个簇，第二列是该样本点与所属簇质心的距离
    """
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    centroid0 = np.mean(dataSet, axis=0).tolist()
    centList = []
    centList.append(centroid0)
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.array(centroid0), dataSet[j, :])**2
    while len(centList) < k:  # 小于K个簇时
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(
                clusterAssment[:, 0] == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(
                clusterAssment[np.nonzero(clusterAssment[:, 0] != i), 1])
            if (sseSplit + sseNotSplit) < lowestSSE:  # 如果满足，则保存本次划分
                bestCentTosplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[
            0], 0] = bestCentTosplit
        centList[bestCentTosplit] = bestNewCents[0, :].tolist()
        centList.append(bestNewCents[1, :].tolist())
        clusterAssment[np.nonzero(clusterAssment[:, 0] == bestCentTosplit)[
            0], :] = bestClustAss
    return centList, clusterAssment


def split_licensePlate_character(plate_binary_img):
    """
    此函数用来对车牌的二值图进行水平方向的切分，将字符分割出来
    输入： plate_gray_Arr是车牌的二值图，rows * cols的数组形式
    输出： character_list是由分割后的车牌单个字符图像二值图矩阵组成的列表
    """
    plate_binary_Arr = np.array(plate_binary_img)
    # 输出像素数据到txt文件中，可以直观的了解像素是如何形成图像的
    with open('img.txt','w') as f:
        for row in plate_binary_Arr:
            f.write(str.replace(str(row), '\n',''))

    # ？这里plate_binary_Arr>=255是什么意思
    # 搞明白了，其实就是=255，获取到所有的像素值为255的索引，
    # 行索引放在row_list, 列索引放在col_list；行列索引相同位置组成的坐标就是值为255的点
    row_list, col_list = np.nonzero(plate_binary_Arr >= 255)
    # dataArr的第一列是列索引，第二列是行索引
    # ？为什么是列放在前面，行放在后面呢，我觉得无所谓，因为基于欧氏距离计算的话，前后结果是一样的
    dataArr = np.column_stack((col_list, row_list))
    # centroids是质心，clusterAssment则是每个（非0）点所对应的质心索引以及距离值
    centroids, clusterAssment = biKmeans(dataArr, 7, distMeas=distEclud)
    # 对质心基于x轴进行排序，这样返回质心就是从左到右排序
    centroids_sorted = sorted(centroids, key=lambda centroid: centroid[0])
    split_list = []
    for centroids_ in centroids_sorted:
        i = centroids.index(centroids_)
        # current_class数据结构是[列索引，行索引]，对应的点属于第i个分类
        current_class = dataArr[np.nonzero(clusterAssment[:, 0] == i)[0], :]
        # 这里x和y的描述是倒置的，可能是和当前处理机制有关系，最后split_list在拼接时候，
        # 是将y放在前面，x放在后面，后面切割的时候也是讲前面作为x，后面作为y
        x_min, y_min = np.min(current_class, axis=0)
        x_max, y_max = np.max(current_class, axis=0)
        # 根据当前分类最左端和最右端信息，可以框处这个分类字符的范围
        split_list.append([y_min, y_max, x_min, x_max])
    
    character_list = []
    for i in range(len(split_list)):
        single_character_Arr = plate_binary_img[split_list[i][0]: split_list[i][1], split_list[i][2]:split_list[i][3]]
        character_list.append(single_character_Arr)
        # cv2.imshow('character'+str(i), single_character_Arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return character_list  # character_list中保存着每个字符的二值图数据

    ############################
    # 测试用
    #print(col_histogram )
    #fig = plt.figure()
    #plt.hist( col_histogram )
    # plt.show()
    ############################


############################机器学习识别字符##########################################
# 这部分是支持向量机的代码


def load_data(filename_1):
    """
    这个函数用来加载数据集，其中filename_1是一个文件的绝对地址
    """
    img_dir = r'C:\Users\python_gay\MySpace\Code_Space\openCVHaveATry\images\car_no\train_no'
    with open(filename_1, 'r') as fr_1:
        temp_address = [row.strip() for row in fr_1.readlines()]
        # print(temp_address)
        # print(len(temp_address))
    # 车牌中没有“O"和"I"两个字母，避免和0,1混淆
    middle_route = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    sample_number = 0  # 用来计算总的样本数
    dataArr = np.zeros((13156, 400))
    label_list = []
    for i in range(len(temp_address)):
        with open(r'C:\Users\Administrator\Desktop\python code\OpenCV\121\\' + temp_address[i], 'r') as fr_2:
            temp_address_2 = [row_1.strip() for row_1 in fr_2.readlines()]
        # print(temp_address_2)
        # sample_number += len(temp_address_2)
        for j in range(len(temp_address_2)):
            sample_number += 1
            # print(middle_route[i])
            # print(temp_address_2[j])
            temp_img = cv2.imread(
                'C:\\Users\Administrator\Desktop\python code\OpenCV\plate recognition\\train\chars2\chars2\\' +
                middle_route[i] + '\\' + temp_address_2[j], cv2.COLOR_BGR2GRAY)
            # print('C:\\Users\Administrator\Desktop\python code\OpenCV\plate recognition\train\chars2\chars2\\'+ middle_route[i]+ '\\' +temp_address_2[j] )
            # cv2.imshow("temp_img",temp_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            temp_img = temp_img.reshape(1, PIXEL_SIZE)
            dataArr[sample_number - 1, :] = temp_img
        label_list.extend([i] * len(temp_address_2))
    # print(label_list)
    # print(len(label_list))
    return dataArr, np.array(label_list)
import os, sys
def load_data_2(filename_1=r'C:\Users\python_gay\MySpace\Code_Space\openCVHaveATry\images\car_no\Train'):
    """
    这个函数用来加载数据集，其中filename_1是一个文件的绝对地址
    """
    # out_dir_chars = os.path.join(filename_1, 'chars')
    # out_dir_chinese = os.path.join(filename_1, 'charsChinese')
    file_count = 0

    for inner_dir in os.listdir(filename_1):
        inner_dir_full = os.path.join(filename_1, inner_dir)
        for img_dir in os.listdir(inner_dir_full):
            img_dir_fullpath = os.path.join(inner_dir_full, img_dir)
            for filename in os.listdir(img_dir_fullpath):
                prefix = filename.split('.')[0]
                suffix = filename.split('.')[1]
                if(suffix != 'jpg'):
                    continue
                file_count += 1

    print('+++++++++ train file count is: ', file_count, '++++++++++')
    pictures = np.zeros((file_count, PIXEL_SIZE))
    labels = []
    index = 0
    for inner_dir in os.listdir(filename_1):
        inner_dir_full = os.path.join(filename_1, inner_dir)
        for img_dir in os.listdir(inner_dir_full):
            img_dir_fullpath = os.path.join(inner_dir_full, img_dir)
            for filename in os.listdir(img_dir_fullpath):
                prefix = filename.split('.')[0]
                suffix = filename.split('.')[1]
                if(suffix != 'jpg'):
                    continue
                file_name = os.path.join(img_dir_fullpath, filename)
                # print('++++++ read file name: ', file_name, '++++++++++')
                # img_temp = cv2.imread(file_name.decode('gbk'))
                img_temp=cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
                img_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2GRAY)
                img_temp = img_temp.reshape(1, PIXEL_SIZE)
                pictures[index] = img_temp
                index+=1
                labels.append(img_dir)
    print('+++++++++ pictures count: ', len(labels), '++++++++++')
    return pictures, labels

# 训练SVM模型
def SVM_recognition(dataArr, label_list):
    from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
    estimator = PCA(n_components=20)  # 初始化一个可以将高维度特征向量（400维）压缩至20个维度的PCA
    # dataArr = estimator.fit_transform(dataArr)
    # new_testArr = estimator.fit_transform(testArr)

    import sklearn.svm
    svc = sklearn.svm.SVC()
    # 使用默认配置初始化SVM，对原始400维像素特征的训练数据进行建模，并在测试集上做出预测
    svc.fit(dataArr, label_list)
    from sklearn.externals import joblib  # 通过joblib的dump可以将模型保存到本地，clf是训练的分类器
    # 保存训练好的模型，通过svc = joblib.load("based_SVM_character_train_model.m")调用
    joblib.dump(svc, "based_SVM_character_train_model.m")


def SVM_recognition_character(character_list):
    character_Arr = np.zeros((len(character_list), PIXEL_SIZE))
    # print(len(character_list))
    for i in range(len(character_list)):
        character_ = cv2.resize(
            character_list[i], (20, 20), interpolation=cv2.INTER_LINEAR)
        new_character_ = character_.reshape((1, PIXEL_SIZE))[0]
        character_Arr[i, :] = new_character_

    from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
    estimator = PCA(n_components=20)  # 初始化一个可以将高维度特征向量（400维）压缩至20个维度的PCA
    # character_Arr = estimator.fit_transform(character_Arr)

    model_file ='based_SVM_character_train_model.m'
    if(not os.path.exists(model_file)):
        dataArr, label_list = load_data_2()
        SVM_recognition(dataArr, label_list)
    ############ 训练svm，获得模型并序列化到.m文件中############
    # filename_1 = r'C:\Users\Administrator\Desktop\python code\OpenCV\dizhi.txt'
    # dataArr, label_list = load_data(filename_1)
    # dataArr, label_list = load_data_2()
    # SVM_recognition(dataArr, label_list)
    ##############
    # 基于训练模型进行预测
    from sklearn.externals import joblib
    clf = joblib.load("based_SVM_character_train_model.m")
    predict_result = clf.predict(character_Arr)
    middle_route = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    print(predict_result.tolist())
    # for k in range(len(predict_result.tolist())):
    #     print('%c' % middle_route[predict_result.tolist()[k]])

if __name__ == "__main__":
    # img = imread_photo(r"C:\Users\python_gay\MySpace\Code_Space\openCVHaveATry\images\car_no\car.jpg")
    file_name = r"C:\Users\python_gay\MySpace\Code_Space\openCVHaveATry\images\car_no\Test\京E51619.jpg"
    img=cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)

    img_resize = resize_photo(img)
    gray_img_, contours = predict(img_resize)
    car_plate = chose_licence_plate(contours)
    img_name = license_segment(car_plate)
    img_name = locate_license(car_plate, img_resize)
    # image_annotated = imread_photo(img_name)
    image_upanddown = remove_plate_upanddown_border(img_name)
    
    # cv2.imshow('image_annotated', image_annotated)
    # cv2.imshow('image_upanddown', image_upanddown)
    character_list = split_licensePlate_character(image_upanddown)
    # test
    # img_test= imread_photo(r"C:\Users\python_gay\MySpace\Code_Space\openCVHaveATry\images\car_no\train_no\2.bmp")
    # img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    # character_list=[]
    # character_list.append(img_test)
    SVM_recognition_character(character_list)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
