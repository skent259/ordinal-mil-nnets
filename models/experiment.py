import ast
import sys
from dataclasses import dataclass
from typing import List

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import coral_ordinal as coral
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from models.architecture import MILType, ModelArchitecture, OrdinalType
from models.dataset import DataSet, DataSetType, MILImageDataGenerator


@dataclass
class ExperimentConfig:
    """
    Configuration options for the experiment.

    Attributes
    ----------
    ordinal_method : OrdinalType 
        The ordinal method to use. One of 'corn', 'coral' ...
    mil_method : MILType
        The multiple instance learning method to use. One of 'mi-net', 'MI-net', ...
    data_set_type : DataSetType 
        The data set to use. One of ... This also defines the underlying model, see XXX.py 
    data_set_name: str
        The name of the data set to use. 
    batch_size : int
        The number of training examples to use in each batch.
    learning_rate : float
        The learning rate of the model.
    epochs : int
        The number of epochs to train the model with.
    pooling_mode : str
        The type of pooling to use for MIL, One of 'max', 'mean' which are passed to models.mil_nets.layer.MILPool
    """

    ordinal_method: OrdinalType
    mil_method: MILType
    data_set_type: DataSetType
    data_set_name: str
    batch_size: int
    learning_rate: float
    epochs: int
    pooling_mode: str = "max"

    @property
    def data_set(self) -> DataSet:
        return DataSet(data_set_type=self.data_set_type, name=self.data_set_name)

    @property
    def model_architecture(self):
        return ModelArchitecture(
            ordinal_type=self.ordinal_method,
            mil_type=self.mil_method,
            data_set_type=self.data_set.data_set_type,
            data_set_img_size=self.data_set.params["img_size"],
            n_classes=self.data_set.params["n_classes"],
            pooling_mode=self.pooling_mode,
        )

    @property
    def result_file(self) -> str:
        return (
            f"ds={self.data_set_name}__or={self.ordinal_method.value}_mil={self.mil_method.value}"
            + f"_pool={self.pooling_mode}_lr={self.learning_rate}_epochs={str(self.epochs)}.csv"
        )

    @property
    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return [
            CSVLogger(filename=self.result_file.replace(".csv", "_training.log")),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=self.result_file.replace(".csv", "_model-{epoch:02d}.hdf5"),
                save_best_only=True,
            ),
        ]


class ExperimentRunner(object):
    """
    Runs the experiment
    """

    def __init__(self, config: ExperimentConfig, output_dir: str = ""):
        self.config = config
        self.data_set = config.data_set
        self.model_architecture = config.model_architecture
        self.output_dir = output_dir

    def run(self):
        print("Running experiment...")

        train_df = self.read_data("train")
        valid_df = self.read_data("valid")
        test_df = self.read_data("test")

        train_generator = self.make_data_generator(train_df)
        valid_generator = self.make_data_generator(valid_df)
        test_generator = self.make_data_generator(test_df)
        self.class_indices = train_generator.class_indices

        self.model = self.model_architecture.build()

        history = self.train(train_generator, valid_generator)

        results = self.predict(test_generator)
        results.to_csv(self.output_dir + self.config.result_file, index=False)

        # save results on training set as back-up
        results_train = self.predict(train_generator)
        results_train.to_csv(
            self.output_dir + self.config.result_file.replace(".csv", "_train.csv"),
            index=False,
        )

        print(results)
        # n_classes = len(train_generator.class_indices)

        # TODO: want checkpointing if possible
        # TODO: consider that some data sets might not have validation data...
        return None

    def read_data(self, type: str):
        ds = self.data_set

        fname = {
            "train": ds.params["dir"] + ds.train,
            "test": ds.params["dir"] + ds.test,
            "valid": ds.params["dir"] + ds.valid,
        }

        df = pd.read_csv(fname[type], dtype=str, index_col=0)
        df = df.applymap(ast.literal_eval)
        return df

    def make_data_generator(self, dataframe: pd.DataFrame):
        ds = self.data_set

        return MILImageDataGenerator(
            dataframe=dataframe,
            directory=ds.params["dir"] + "images/",
            x_col=ds.params["x_col"],
            y_col=ds.params["y_col"],
            batch_size=1,
            shuffle=True,
            class_mode=ds.params["class_mode"],
            target_size=ds.params["img_size"],
            **ds.params["augmentation_args"],
        )

    def train(
        self,
        train_generator: MILImageDataGenerator,
        valid_generator: MILImageDataGenerator,
    ) -> tf.keras.callbacks.History:

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
        return self.model.fit(
            x=train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=valid_generator,
            validation_steps=STEP_SIZE_VALID,
            epochs=self.config.epochs,
            callbacks=self.config.callbacks,
        )

    def predict(self, test_generator: MILImageDataGenerator) -> pd.DataFrame:
        """
        Handle any post-model prediction tasks to get label output
        """

        if self.config.ordinal_method is OrdinalType.CORAL:
            ordinal_logits = self.model.predict(test_generator, verbose=1)
            cum_probs = pd.DataFrame(ordinal_logits).apply(scipy.special.expit)
            predicted_class_indices = cum_probs.apply(lambda x: x > 0.5).sum(axis=1)
            # don't add 1 because we are 0 indexing

        if self.config.ordinal_method is OrdinalType.CORN:
            ordinal_logits = self.model.predict(test_generator, verbose=1)
            cum_probs = pd.DataFrame(coral.corn_cumprobs(ordinal_logits))
            predicted_class_indices = cum_probs.apply(lambda x: x > 0.5).sum(axis=1)

        labels = dict((v, k) for k, v in self.class_indices.items())
        predictions = [labels[k] for k in predicted_class_indices]

        results = test_generator.df.copy()
        results["predictions"] = predictions
        return results

    def evaluate(self):
        """
        Evaluate the model run and record the metrics of interest
        """
        pass


if __name__ == "__main__":

    exp_config = ExperimentConfig(
        ordinal_method=OrdinalType.CORAL,
        mil_method=MILType.CAP_MI_NET,
        data_set_type=DataSetType.FGNET,
        data_set_name="fgnet_bag_wr=0.5_size=4_i=0",
        batch_size=1,
        learning_rate=0.01,
        epochs=1,
        pooling_mode="max",
    )

    print(exp_config.model_architecture.build().summary())

    exp = ExperimentRunner(exp_config)

    exp.run()
