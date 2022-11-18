from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


class DataSetType(Enum):
    FGNET = "fgnet"


DATASET_PARAM = {
    DataSetType.FGNET: {
        "dir": "datasets/fgnet/",
        "splits_dir": "splits_bag/",
        "x_col": "img_name",
        "y_col": "age_group",
        "img_size": (128, 128, 3),
        "n_classes": 6,
        "class_mode": "sparse",
        "augmentation_args": {
            "horizontal_flip": True,
            "crop_range": 0.1,
            "contrast_lower": 0.5,
            "contrast_upper": 2.0,
            "brightness_delta": 0.2,
            "hue_delta": 0.1,
            "quality_min": 50,
            "quality_max": 100,
        },
    }
}


@dataclass
class DataSet:
    """
    Data set information to be used by experiment methods

    Attributes
    ----------
    data_set_type : DataSetType
        The data set type which directly maps to information about the directory to use, x and y 
        columns, image size, and more. See DATASET_PARAM
    name : str 
        The name of the dataset.
    params : dict
        The data set parameters, including 'dir', 'x_col', 'y_col', 'img_size', 'class_mode', and 
        'augmentation_args'. 
    train : str
        The file that describes the training information. 
    test : str
        The file that describes the testing information. 
    valid : str
        The file that describes the validation information. 
    """

    data_set_type: DataSetType
    name: str

    @property
    def params(self) -> dict:
        return DATASET_PARAM[self.data_set_type]

    @property
    def train(self) -> str:
        return self.params["splits_dir"] + self.name + "_train.csv"

    @property
    def test(self) -> str:
        return self.params["splits_dir"] + self.name + "_test.csv"

    @property
    def valid(self) -> str:
        return self.params["splits_dir"] + self.name + "_valid.csv"


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
        * crop_range: (0.0, 1.0)
        * contrast_lower: (-inf, inf)
        * contrast_upper: (-inf, inf)
        * brightness_delta: [0, 1)
        * hue_delta: [-1, 1]
        * quality_min: [0, 100]
        * quality_max: [0, 100]
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

        class_indices = np.unique(self.df[self.y_col].explode())
        self.class_indices = dict(zip(class_indices, np.arange(len(class_indices))))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, paths, target_size):
        return np.asarray([self.__get_image(i, target_size) for i in paths])

    def __get_output(self, label, class_indices):
        int_label = [class_indices[i] for i in label]
        int_label = np.max(int_label)  # QUESTION: why did I do this?

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
