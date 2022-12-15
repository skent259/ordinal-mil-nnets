from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf


class MILImageDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence to pull in batches of bags in a MIL data set

    NOTE: currently only works with `batch_size` = 1 unless all bags have 
    the same bag_size. The numpy array can't convert different bag sizes into 
    a single array. 

    Thank you to following site for inspiration:
    https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3

    To perform data augmentation, change one of the following default arguments (range given after):
        * horizontal_flip: (True or False)
        * crop_range: (0.0, 1.0), no aug if 0
        * contrast_lower: (-inf, inf), no aug if 1
        * contrast_upper: (-inf, inf), no aug if 1
        * brightness_delta: [0, 1), no aug if 0
        * hue_delta: [-1, 1], no aug if 0
        * quality_min: [0, 100], no aug if 100
        * quality_max: [0, 100], no aug if 100
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        directory: str,
        x_col: str,
        y_col: str,
        batch_size: int,
        shuffle: bool,
        class_mode: str,
        target_size: Tuple[int],
        class_indices: Dict[str, int] = None,
        horizontal_flip: bool = False,
        crop_range: float = 0,
        contrast_lower: float = 1.0,
        contrast_upper: float = 1.0,
        brightness_delta: float = 0.0,
        hue_delta: float = 0.0,
        quality_min: int = 100,
        quality_max: int = 100,
    ):
        # TODO: maybe add `rescale`?

        self.df = dataframe.copy()
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_mode = class_mode
        self.target_size = target_size
        self.horizontal_flip = horizontal_flip
        self.crop_range = crop_range
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.quality_min = quality_min
        self.quality_max = quality_max

        self.n = len(self.df)

        if not class_indices:
            class_indices = np.unique(self.df[self.y_col].explode())
            self.class_indices = dict(zip(class_indices, np.arange(len(class_indices))))
        else:
            self.class_indices = class_indices

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, paths, target_size):
        return np.asarray([self.__get_image(i, target_size) for i in paths])

    def __get_output(self, label, class_indices):
        if type(label) is list:
            int_label = [class_indices[i] for i in label]
            int_label = np.max(int_label)
        else:
            int_label = class_indices[label]

        if self.class_mode == "sparse":
            return float(int_label)
        if self.class_mode == "categorical":
            return tf.keras.utils.to_categorical(
                int_label, num_classes=len(class_indices)
            )
        return None

    def __get_image(self, path, target_size):

        path = self.directory + path
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0

        image_arr = self.__augment_image(image_arr)  # only if non-default args used
        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()

        return image_arr

    def __getitem__(self, index):

        batches = self.df[index * self.batch_size : (index + 1) * self.batch_size]

        path_batch = batches[self.x_col]
        label_batch = batches[self.y_col]

        # Pull images and combine as array for each bag
        X_batch = np.asarray(
            [self.__get_input(i, self.target_size) for i in path_batch]
        )

        # y_batch = np.asarray([np.asarray(i) for i in batches[self.y_col]])
        # y_batch = batches[self.y_col]
        y_batch = np.asarray(
            [self.__get_output(i, self.class_indices) for i in label_batch]
        )

        return X_batch, y_batch

    def __len__(self):
        return self.n // self.batch_size

    def __augment_image(self, img: np.ndarray):

        if self.horizontal_flip:
            img = tf.image.random_flip_left_right(img)

        if self.crop_range > 0:
            crop_height = round(img.shape[0] * (1 - self.crop_range))
            crop_width = round(img.shape[1] * (1 - self.crop_range))

            img = tf.image.random_crop(img, size=[crop_height, crop_width, 3])

        if self.contrast_lower != 1.0 or self.contrast_upper != 1.0:
            img = tf.image.random_contrast(
                img, lower=self.contrast_lower, upper=self.contrast_upper
            )
        if self.brightness_delta > 0:
            img = tf.image.random_brightness(img, max_delta=self.brightness_delta)
        if self.hue_delta > 0:
            img = tf.image.random_hue(img, max_delta=self.hue_delta)

        if self.quality_min != 100 or self.quality_max != 100:
            img = tf.image.random_jpeg_quality(
                img,
                min_jpeg_quality=self.quality_min,
                max_jpeg_quality=self.quality_max,
            )

        return img


class MILTabularDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence to pull in batches of bags in a MIL data set

    NOTE: currently only works with `batch_size` = 1 unless all bags have 
    the same bag_size. The numpy array can't convert different bag sizes into 
    a single array. 

    Thank you to following site for inspiration:
    https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        x_cols: List[str],
        y_col: str,
        bag_col: str,
        batch_size: int,
        shuffle: bool,
        class_mode: str,
        class_indices: Dict[str, int] = None,
    ):
        # TODO: maybe add `rescale`?

        self.df = dataframe.copy()
        self.x_cols = x_cols
        self.y_col = y_col
        self.bag_col = bag_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_mode = class_mode

        self.bags = self.df[self.bag_col].unique()
        self.n = len(self.bags)

        if not class_indices:
            class_indices = np.unique(self.df[self.y_col].explode())
            self.class_indices = dict(zip(class_indices, np.arange(len(class_indices))))
        else:
            self.class_indices = class_indices

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, bag):
        ind = self.df[self.bag_col] == bag
        return np.asarray(self.df.loc[ind, self.x_cols])

    def __get_output(self, bag):
        ind = self.df[self.bag_col] == bag
        label = self.df[self.y_col].loc[ind,].tolist()

        if type(label) is list:
            int_label = [self.class_indices[i] for i in label]
            int_label = np.max(int_label)
        else:
            int_label = self.class_indices[label]

        if self.class_mode == "sparse":
            return float(int_label)
        if self.class_mode == "categorical":
            return tf.keras.utils.to_categorical(
                int_label, num_classes=len(self.class_indices)
            )
        return None

    def __getitem__(self, index):
        # Each row represents an instance, but each batch represents a bag
        bag_batches = self.bags[index * self.batch_size : (index + 1) * self.batch_size]

        X_batch = np.asarray([self.__get_input(i) for i in bag_batches])
        y_batch = np.asarray([self.__get_output(i) for i in bag_batches])

        return X_batch, y_batch

    def __len__(self):
        return self.n // self.batch_size
