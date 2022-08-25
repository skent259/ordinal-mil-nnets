from dataclasses import dataclass
from typing import Tuple


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
        The file that describes the training information. This file should be contained in `dir`.
    valid : str
        The file that describes the training information. This file should be contained in `dir`.
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

