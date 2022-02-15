import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import collections

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, data):
        img, lbl, focal_difference = data

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)


            for i in range(len(focal_difference)):
                focal_difference[i] = focal_difference[i].transpose(Image.FLIP_LEFT_RIGHT)


        data = [img, lbl, focal_difference]

        return data


class Random_rotation(object):

    def __call__(self, data):
        img, lbl, focal_difference = data
        angle = random.choice([0, 90, 180, 270])


        img = img.rotate(angle)
        lbl = lbl.rotate(angle)

        for i in range(len(focal_difference)):
            focal_difference[i] = focal_difference[i].rotate(angle)

        data = [img, lbl, focal_difference]
        return data


class Random_crop_Resize(object):
    def __init__(self, crop_size):
        self.crop_size=crop_size

    def __call__(self, data):

        img, lbl, focal_difference = data
        assert img.size == lbl.size,"img should have the same shape as label"

        width, height = img.size

        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        region = [x, y, width - x, height - y]

        img = img.crop(region)
        lbl = lbl.crop(region)

        img = img.resize((width, height), Image.BILINEAR)
        lbl = lbl.resize((width, height), Image.NEAREST)

        for i in range(len(focal_difference)):
            focal_difference[i] = focal_difference[i].crop(region)
            focal_difference[i] = focal_difference[i].resize((width, height), Image.BILINEAR)


        data = [img, lbl, focal_difference]

        return data




