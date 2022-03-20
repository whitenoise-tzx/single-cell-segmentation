import os
from PIL import Image
import numpy as np
import cv2
from skimage.io import imread
path = r'F:\dataset\NmuMg\test\BIB\\'
newpath = r'F:\dataset\NmuMg\test\BIB_\\'


def picture(path):
    b=0
    for i in os.listdir(path):
        img = cv2.imread(path + i)
        filename = newpath + '/label_%d.png' % b
        cv2.imwrite(filename, img)
        b+=1

        """
        img = cv2.imread(path + i, cv2.COLOR_BGR2GRAY)
        height_mask = img.shape[0]
        weight_mask = img.shape[1]
        o=0
        for row in range(height_mask):
            for col in range(weight_mask):
                if img[row, col] ==2:
                    img[row, col] = 0
                else:
                    img[row, col] = 255
                if img[row, col] == 0 or img[row, col] == 255:
                    o += 1
        image_array = img
        filename = newpath + '/label_%d.png' % b
        cv2.imwrite(filename, image_array)
        b += 1
"""
"""
        img = np.array(imread(path + i), dtype=np.float64)
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        new_img = cv2.convertScaleAbs(img)
        image_array = new_img
        filename = newpath + '/data_%d.tif' % b
        cv2.imwrite(filename, image_array)
        b+=1
"""
"""
        img = Image.open(path + i)
        img = Image.fromarray(np.uint8(img))
        t = img.convert('L')
        img = Image.fromarray(np.uint8(t))  # *255
        filename = newpath + '/label_%d.png' % b
        #cv2.imwrite(filename, img)
        img.save(filename + i)
        b += 1
"""


picture(path)
"""
def prep_data(path_img,path_gt):
        data = []
        label = []
        img_list = sorted(listdir(path_img))
        gt_list=sorted(listdir(path_gt))
        i=0
        while (i<len(img_list)):
            img = np.array(imread(path_img +'/'+ img_list[i]),dtype=np.float64)
            #img = np.array(imread(path_img +'/'+ img_list[i]))\n",
            #print img.shape\n",
            img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))#img*1.0 transform array to double\n",
            img=img*1.0/np.median(img)
            img_h=img.shape[0]
            img_w=img.shape[1]
            img=np.reshape(img,(img_h,img_w,1))
            data.append(img)
            gt =np.array(imread(path_gt + '/'+ gt_list[i]))
    #---------------with or without normalization-----------------------
            if np.count_nonzero(gt)!=0:
                nonzero_gt=gt[gt>0]
                gt=gt*1.0/np.median(nonzero_gt)

            gt=np.reshape(gt,(img_h,img_w,1))
            label.append(gt)
            i+=1
        data=np.array(data)
        label=np.array(label)

        return data, label
"""
