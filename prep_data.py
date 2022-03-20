import numpy as np
import pandas as pd
import os
from os import listdir
import scipy.misc
from skimage.io import imread
from PIL import Image
import imageio
import sys
import cv2

dir_path=r'F:\single_cell_segmentation-master\HK2_DIC'
main_path=dir_path+r'\training_data'
path_train_input =main_path+r'\thick\train\Img'
path_test_input=dir_path+r'\pixel_evaluation\thick\Img'
path_train_gt = main_path+r'\thick\train\BIB'
path_test_gt=dir_path+r'\pixel_evaluation\thick\BIB'
#path_output=dir_path+'/output'

def prep_data(path_img, path_gt):
    data = []
    label = []

    img_list = sorted(listdir(path_img))
    gt_list = sorted(listdir(path_gt))
    i = 0
    while (i < len(img_list)):
        img = np.array(imread(path_img + '/' + img_list[i]), dtype=np.float64)
        # img = np.array(imread(path_img +'/'+ img_list[i]))
        # print img.shape
        img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
        img = img * 1.0 / np.median(img)
        img_h = img.shape[0]
        img_w = img.shape[1]
        img = np.reshape(img, (img_h, img_w, 1))
        data.append(img)

        gt = np.array(imread(path_gt + '/' + gt_list[i]))
        # ---------------with or without normalization-----------------------
        if np.count_nonzero(gt) != 0:
            nonzero_gt = gt[gt > 0]
            gt = gt * 1.0 / np.median(nonzero_gt)

        gt = np.reshape(gt, (img_h, img_w, 1))
        label.append(gt)

        i += 1
    data = np.array(data)
    label = np.array(label)
    print (data.shape, label.shape)
    return data, label

def prep_prediction_data(path_img):
    data = []
    img_list = sorted(listdir(path_img))
    i=0
    while (i<len(img_list)):
        img = np.array(imread(path_img +'/'+ img_list[i]),dtype=np.float64)
        img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))#img*1.0 transform array to double
        img=img*1.0/np.median(img)
        img_h=img.shape[0]
        img_w=img.shape[1]
        img=np.reshape(img,(img_h,img_w,1))
        data.append(img)
        i+=1
    data=np.array(data)
    return data

train_data, train_label = prep_data(path_train_input,path_train_gt)

newpath_data=r'F:\single_cell_segmentation-master\HK2_DIC\training_data\thick\train\data'
newpath_label=r'F:\single_cell_segmentation-master\HK2_DIC\pixel_evaluation\thick\label'

for i in range (357):
    image_array = train_data[i]
    filename = newpath_data + '/data_%d.tif' % i
    cv2.imwrite(filename, image_array)



