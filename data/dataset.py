"""
data.py

1. load image
2. load mask
"""
import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import cv2

from utils.data import crop_shape, padding
from utils.logger import logger

EXTENSION = ['jpg', 'png', 'jpeg']


class Dataset(object):
    def __init__(self, image_path=None, mask_path=None, image_size=None, logger=None):
        self._image_path = image_path
        self._mask_path = mask_path
        self._image_size = image_size[:2]
        self._logger = logger

        self._images = list()
        self._images_names = list()

        self._masks = list()
        self._masks_names = list()

        self._image_paths = list()
        self._mask_paths = list()

        self.load()

    @property
    def image_path(self):
        return self._image_path

    @property
    def mask_path(self):
        return self._mask_path

    @property
    def images(self):
        return self._images

    @property
    def masks(self):
        return self._masks

    @property
    def images_names(self):
        return self._images_names

    @property
    def masks_names(self):
        return self._masks_names

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_paths(self):
        return self._image_paths

    @property
    def mask_paths(self):
        return self._mask_paths

    @property
    def logger(self):
        return self._logger

    def load(self):
        """
        load data

        :return:
        """
        self.logger.info("start loading path")
        self.load_path()
        self.logger.info("loading done")

        self.logger.info("start loading image and mask")
        self.load_data()
        self.logger.info("loading done")

    def load_path(self):
        """
        Load image and mask path

        :return:
        """

        for extension in EXTENSION:
            self.image_paths.extend(glob.glob(os.path.join(self.image_path, '*.{}'.format(extension))))
        self.image_paths.sort()

        for extension in EXTENSION:
            self.mask_paths.extend(glob.glob(os.path.join(self.mask_path, '*.{}'.format(extension))))
        self.mask_paths.sort()

    def load_data(self):
        """
        Load image and mask

        :return:
        """
        for image_path in self.image_paths:
            image_name = os.path.splitext((os.path.basename(image_path)))[0]
            image = cv2.imread(image_path)
            image = crop_shape(image, self.image_size)
            image = padding(image, self.image_size)
            self.images.append(image)
            self.images_names.append(image_name)

        for mask_path in self.mask_paths:
            mask_name = os.path.splitext((os.path.basename(mask_path)))[0]
            mask = cv2.imread(mask_path)
            mask = crop_shape(mask, self.image_size)
            mask = padding(mask, self.image_size)
            self.masks.append(mask)
            self.masks_names.append(mask_name)

        self._images = np.array(self.images)
        self._masks = np.array(self.masks)
