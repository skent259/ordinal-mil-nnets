from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass
class DataSet:
    """
    Data set information to be used by experiment methods

    Attributes
    ----------
    name : str 
        The name of the dataset.
    dir : str
        The directory of the dataset.
    train : str
        The file that describes the training information. This file should be contained in `dir`.
    test : str
        The file that describes the testing information. This file should be contained in `dir`.
    valid : str
        The file that describes the validation information. This file should be contained in `dir`.
    img_size : List[int]
        The size of the image to transform to.
    """

    name: str
    img_size: Tuple[int]
    dir: str = None
    train: str = None
    test: str = None
    valid: str = None

    def __post_init__(self):
        # can't set default values based on self.name in init, must do post initialization
        self.dir = "datasets/" + self.name + "/" if self.dir is None else self.dir
        self.train = self.name + "_train.csv" if self.train is None else self.train
        self.test = self.name + "_test.csv" if self.test is None else self.test
        self.valid = self.name + "_valid.csv" if self.valid is None else self.valid


class MILImageDataGenerator(tf.keras.utils.Sequence):
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
        dataframe,
        directory: str,
        x_col: str,
        y_col: str,
        batch_size: int,
        shuffle: bool,
        class_mode: str,
        target_size,
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
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()

        return image_arr / 255.0

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


# TODO: consider moving to the datasets folder (maybe until a utils.py file)
def convert_dataset_to_bag_level(df, bag_size, shuffle=True, seed=None):
    """
    Takes a pandas dataset and turns each column into a bag level representation.
    """

    def to_bag_column(x, bag_size):
        """
        Take a list or pandas Series and create a list of bags
        """
        row = 0
        i = 0
        bag_col = []
        bag = []
        while row < len(x):
            bag.append(x[row])
            row += 1
            i += 1

            if i >= bag_size:
                bag_col.append(bag)
                i = 0
                bag = []

        # add remaining instances to the last bag
        bag_col[-1] = bag_col[-1] + bag

        return bag_col

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    new_df = {i: to_bag_column(df[i], bag_size) for i in df.columns}

    return pd.DataFrame(new_df)
