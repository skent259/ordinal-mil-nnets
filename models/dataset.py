from dataclasses import dataclass
from enum import Enum


class DataSetType(Enum):
    FGNET = "fgnet"
    AES = "aes"
    BCNB_ALN = "bcnb"
    AMREV_TV = "amrev_tv"


DATASET_PARAM = {
    DataSetType.FGNET: {
        "dir": "datasets/fgnet/",
        "splits_dir": "splits_bag/",
        "x_col": "img_name",
        "y_col": "age_group",
        "img_size": (128, 128, 3),
        "n_classes": 6,
        "class_indices": {
            "00-03": 0,
            "04-11": 1,
            "12-16": 2,
            "17-24": 3,
            "25-40": 4,
            "41+": 5,
        },
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
    },
    DataSetType.AES: {
        "dir": "datasets/aes/",
        "splits_dir": "splits_bag/",
        "x_col": "img_name",
        "y_col": "score",
        "img_size": (128, 128, 3),  # matches Shi, Cao, and Raschka (2022)
        "n_classes": 5,
        "augmentation_args": {
            "horizontal_flip": False,
            "crop_range": 0.05,
            "contrast_lower": 0.75,
            "contrast_upper": 1.5,
            "brightness_delta": 0.1,
            "hue_delta": 0.1,
            "quality_min": 75,
            "quality_max": 100,
        },
    },
    DataSetType.BCNB_ALN: {
        "dir": "datasets/bcnb/",
        "splits_dir": "splits_bag/",
        "x_col": "img_name",
        "y_col": "aln_status",
        "img_size": (256, 256, 3),
        "n_classes": 3,
        "augmentation_args": {},
    },
    DataSetType.AMREV_TV: {
        "dir": "datasets/amrev/",
        "splits_dir": "splits_bag/",
        "x_col": "review",
        "y_col": "rating",
        "img_size": None,
        "n_classes": 5,
        "augmentation_args": {},
    },
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
        The data set parameters, including 'dir', 'x_col', 'y_col', 'img_size', and 
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

