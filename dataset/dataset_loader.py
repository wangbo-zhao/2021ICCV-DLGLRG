import os
from .trainsforms import Compose, RandomHorizontalFlip, Random_rotation, Random_crop_Resize
import numpy as np
import PIL.Image
import scipy.io as sio
import torch
from torch.utils import data
import cv2

class MyData(data.Dataset):  # inherit
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)

    def __init__(self, root, transform=False):
        super(MyData, self).__init__()
        self.root = root

        self.img_names = []
        self.lbl_names = []
        self.focal_names = []

        subsets = ["DUTLF-FS-Train", "HFUT-Train"]
        for subset in subsets:

            img_root = os.path.join(self.root, subset, 'train_images')
            lbl_root = os.path.join(self.root, subset, 'train_masks')
            focal_root = os.path.join(self.root, subset, 'train_original_focal')

            file_names = os.listdir(img_root)

            for i, name in enumerate(file_names):
                if not name.endswith('.jpg'):
                    continue
                self.lbl_names.append(
                    os.path.join(lbl_root, name[:-4]+'.png')
                )
                self.img_names.append(
                    os.path.join(img_root, name)
                )
                self.focal_names.append(
                    os.path.join(focal_root, name[:-4]+'.mat')
                )



    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)

        lbl_file = self.lbl_names[index]
        lbl = PIL.Image.open(lbl_file)
        lbl = lbl.resize((256, 256))
        lbl = np.array(lbl, dtype=np.int32)

        focal_file = self.focal_names[index]
        focal = sio.loadmat(focal_file)
        focal = focal['img']
        focal = np.array(focal, dtype=np.int32)

        if focal.shape[0] != 256:
            new_focal = []

            focal_num = focal.shape[2] // 3

            for i in range(focal_num):
                a = focal[:, :, i*3:i*3+3].astype(np.uint8)
                a = cv2.resize(a, (256, 256))
                new_focal.append(a)
            focal = np.concatenate(new_focal, axis=2)

        if self.transform:
            return self.transform(img, lbl, focal)
        else:
            return img, lbl, focal

    def transform(self, img, lbl, focal):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float() / 255
        lbl = lbl.unsqueeze(0)

        focal_num = focal.shape[2] // 3
        mean_focal = np.tile(self.mean_rgb, focal_num)
        std_focal = np.tile(self.std_rgb, focal_num)

        focal = focal.astype(np.float64)/255.0
        focal -= mean_focal
        focal /= std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()
        return img, lbl, focal









class MyTestData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)

    def __init__(self, root, transform=False):
        super(MyTestData, self).__init__()
        self.root = root
        self._transform = transform

        self.img_names = []
        self.names = []
        self.focal_names = []

        subsets = ["LFSD", "HFUT", "DUTLF-FS"]
        for subset in subsets:

            img_root = os.path.join(self.root, subset, "TestSet", 'test_images')
            focal_root = os.path.join(self.root, subset,  "TestSet", 'test_original_focal')
            file_names = os.listdir(img_root)

            for i, name in enumerate(file_names):
                if not name.endswith('.jpg'):
                    continue
                self.img_names.append(
                    os.path.join(img_root, name)
                )
                self.names.append(name[:-4])
                self.focal_names.append(
                    os.path.join(focal_root, name[:-4]+'.mat')
                )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img_size = img.size
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)

        # load focal
        focal_file = self.focal_names[index]
        focal = sio.loadmat(focal_file)
        focal = focal['img']
        focal = np.array(focal, dtype=np.int32)

        if focal.shape[0] != 256:
            new_focal = []

            focal_num = focal.shape[2] // 3

            for i in range(focal_num):
                a = focal[:, :, i*3:i*3+3].astype(np.uint8)
                a = cv2.resize(a, (256, 256))
                new_focal.append(a)
            focal = np.concatenate(new_focal, axis=2)


        if self._transform:
            img, focal = self.transform(img, focal)
            return img_file, img, focal
        else:
            return img_file, img, focal

    def transform(self, img, focal):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        focal_num = focal.shape[2] // 3
        mean_focal = np.tile(self.mean_rgb, focal_num)
        std_focal = np.tile(self.std_rgb, focal_num)

        focal = focal.astype(np.float64)/255.0
        focal -= mean_focal
        focal /= std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()

        return img, focal