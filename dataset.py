import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import skimage
import cv2
from glob import glob
import imageio
from skimage.io import imread
import helpers

palette = [[0], [1], [2]]
num_classes = 3


class IsbiCellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'F:\isbi'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\train\images\*')
        self.mask_paths = glob(self.root + r'\train\label\*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        # self.test_img_paths = glob(self.root + r'\test\test_images\*')
        # self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class NMuMgDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = '/shared/home/v_zixin_tang/dataset/NmuMg/'
        #self.root = r'F:\single_cell_segmentation-master\NMuMg_phase_contrast'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform =transform
        self.target_transform =target_transform

    def getDataPath(self):

        self.img_paths = glob(self.root + 'train/data/*')
        self.mask_paths = glob(self.root + 'train/label/*')
        self.test_img_paths = glob(self.root + 'test/data/*')
        self.test_mask_paths = glob(self.root + 'test/label/*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(sorted(self.img_paths), sorted(self.mask_paths), test_size=0.2, random_state=42)
        # self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return sorted(self.train_img_paths),sorted(self.train_mask_paths)
        if self.state == 'val':
            return sorted(self.val_img_paths),sorted(self.val_mask_paths)
        if self.state == 'test':
            return sorted(self.test_img_paths),sorted(self.test_mask_paths)

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = Image.open(pic_path)
        pic = np.array(pic)
        pic = np.expand_dims(pic, axis=2)
        pic = pic.astype('float32')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class T47DDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'/shared/home/v_zixin_tang/dataset/T47D_fluorescence'
        #self.root = r'F:\dataset\T47D_fluorescence'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):

        self.img_paths = glob(self.root + r'/train/data/*')
        self.mask_paths = glob(self.root + r'/train/label/*')
        self.test_img_paths = glob(self.root + r'/test/data/*')
        self.test_mask_paths = glob(self.root + r'/test/label/*')

        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(sorted(self.img_paths), sorted(self.mask_paths), test_size=0.2, random_state=42)
        # self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return sorted(self.train_img_paths), sorted(self.train_mask_paths)
        if self.state == 'val':
            return sorted(self.val_img_paths), sorted(self.val_mask_paths)
        if self.state == 'test':
            return sorted(self.test_img_paths), sorted(self.test_mask_paths)

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        """
        pic = np.array(imread(pic_path), dtype=np.float64)
        cv2.normalize(pic, pic, 0, 255, cv2.NORM_MINMAX)
        pic = cv2.convertScaleAbs(pic)
        pic = np.array(pic)
        pic = pic.astype('float32')  # / 255
        mask = Image.open(mask_path)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.convert("L")
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        """
        pic = Image.open(pic_path)
        pic = np.array(pic)
        pic = np.expand_dims(pic, axis=2)
        pic = pic.astype('float32')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class HK2_DICDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = '/shared/home/v_zixin_tang/dataset/HK2_DIC/'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):

        self.img_paths = glob(self.root + 'train/data/*')
        self.mask_paths = glob(self.root + 'train/label/*')
        self.test_img_paths = glob(self.root + 'test/data/*')
        self.test_mask_paths = glob(self.root + 'test/label/*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split((self.img_paths), (self.mask_paths), test_size=0.2, random_state=42)
        # self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return (self.train_img_paths), (self.train_mask_paths)
        if self.state == 'val':
            return (self.val_img_paths), (self.val_mask_paths)
        if self.state == 'test':
            return (self.test_img_paths), (self.test_mask_paths)

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = Image.open(pic_path)
        pic = np.array(pic)
        pic = np.expand_dims(pic, axis=2)
        pic = pic.astype('float32')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        """
        pic = np.array(imread(pic_path), dtype=np.float64)
        cv2.normalize(pic, pic, 0, 255, cv2.NORM_MINMAX)
        pic = cv2.convertScaleAbs(pic)
        pic = np.array(pic)
        pic = np.expand_dims(pic, axis=2)
        pic = pic.astype('float32') #/ 255
        mask = Image.open(mask_path)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.convert("L")
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        """
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)
