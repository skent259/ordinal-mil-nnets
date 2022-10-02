from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """
    Configuration options for the experiment.

    Attributes
    ----------
    ordinal_method : str 
        The ordinal method to use. One of 'corn', 'coral' ...
    mil_method : str
        The multiple instance learning method to use. One of ...
    data_set : str 
        The data set to use. One of ... This also defines the underlying model, see XXX.py 
    batch_size : int
        The number of training examples to use in each batch.
    learning_rate : float
        The learning rate of the model.
    epochs : int
        The number of epochs to train the model with.
    """

    ordinal_method: str
    mil_method: str
    data_set: str
    batch_size: int
    learning_rate: float
    epochs: int


# TODO: maybe include features of the randomly genereated bags from ordinal images
# i.e. `n_inst` (number of instances in the bag)
