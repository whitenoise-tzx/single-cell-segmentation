import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
import skimage
from skimage import morphology
from skimage.morphology import disk,square,diamond
from skimage import measure
from skimage.measure import label
from skimage.color import label2rgb
from skimage.io import imread
import helpers

x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        #transforms.Normalize((0.5,), (0.5,))
    ])
y_transforms = transforms.ToTensor()
pic=Image.open("data_10.tif")
pic = np.array(pic)
pic = np.expand_dims(pic, axis=2)
pic = pic.astype('float32')
pic=x_transforms(pic)
print(pic.shape)
print(pic)