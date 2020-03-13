# a=b'你好'
# a=b'hello'
# '你好'.encode('utf-8')
# bytes('走向前','utf-8')

import sys, os
folder_path='C:\\MySpace\\research\\machinelearning\\OpenCV\Yale_face\\yalefaces\\'
files = os.listdir(folder_path)
# [os.rename(folder_path+filename,folder_path+filename+'.gif') for filename in files if not os.path.exists(folder_path+filename+'.gif')]

for filename in files:
    man = filename.split('.')[0]
    man_folder_path = os.path.join(folder_path, man)
    if(not os.path.exists(man_folder_path)):
        os.mkdir(man_folder_path)

    file_path_src = os.path.join(folder_path, filename)
    file_path_des = os.path.join(man_folder_path, filename)
    os.rename(file_path_src, file_path_des)
