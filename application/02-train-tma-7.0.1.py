import argparse
import sys
from typing import Dict, List

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import coral_ordinal as coral
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger

from application.config import TMAConfig
from experiment.paramtools import expand_grid
from models.architecture import MILType, OrdinalType
from models.generators import MILTabularDataGenerator
from models.mil_attention.layer import mil_attention_layer
from models.mil_nets.layer import BagWise, MILPool


def train_test_split_bagwise(
    df: pd.DataFrame, bag_col: str, train_size=0.8, shuffle=True
):

    bags = df[bag_col].unique()
    if shuffle:
        np.random.shuffle(bags)

    train, test = np.split(bags, [int(train_size * len(bags))])

    ind_train = df[bag_col].isin(train)
    ind_test = df[bag_col].isin(test)

    return df.loc[ind_train,], df.loc[ind_test,]


class TMARunner(object):
    """
    Runs the applied TMA example
    """

    def __init__(
        self, config: TMAConfig, save_results: bool = False, output_dir: str = ""
    ):
        self.config = config
        self.save_results = save_results
        self.output_dir = output_dir
        self.file: Dict[str, str] = {
            x: output_dir + y for (x, y) in config.file.items()
        }
        self.n_classes = 3
        self.n_features = 30

    @property
    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        if self.save_results:
            return [CSVLogger(filename=self.file["csv_log"])]

        return None

    def run(self, train_df, test_df, verbose=0):

        train_generator = self.make_data_generator(train_df)
        test_generator = self.make_data_generator(test_df)
        self.class_indices = train_generator.class_indices

        self.model = self.build_model()

        history = self.train(train_generator, verbose=verbose)

        if self.save_results:
            self.model.save(self.file["model"])

        return self.evaluate(test_generator)

    def make_data_generator(self, dataframe: pd.DataFrame):

        class_mode = {
            OrdinalType.CORN: "sparse",
            OrdinalType.CORAL: "sparse",
            OrdinalType.CLM_QWK_LOGIT: "categorical",
            OrdinalType.CLM_QWK_PROBIT: "categorical",
            OrdinalType.CLM_QWK_CLOGLOG: "categorical",
        }

        return MILTabularDataGenerator(
            dataframe=dataframe,
            x_cols=list(df.columns[2:]),
            y_col=Y_COL,
            bag_col=BAG_COL,
            batch_size=1,
            shuffle=True,
            class_mode=class_mode.get(self.config.ordinal_method),
        )

    def build_model(self):
        k = self.config.fc_layer_size

        ordinal_layer = {
            OrdinalType.CORAL: coral.CoralOrdinal(self.n_classes),
            OrdinalType.CORN: coral.CornOrdinal(self.n_classes),
        }
        ordinal_layer = ordinal_layer.get(self.config.ordinal_method)

        mae = {
            OrdinalType.CORAL: coral.MeanAbsoluteErrorLabels(),
            OrdinalType.CORN: coral.MeanAbsoluteErrorLabels(corn_logits=True),
        }

        loss = {
            OrdinalType.CORAL: coral.OrdinalCrossEntropy(num_classes=self.n_classes),
            OrdinalType.CORN: coral.CornOrdinalCrossEntropy(),
        }

        inputs = layers.Input(shape=(None, self.n_features))

        x = BagWise(layers.Dense(k))(inputs)
        x = BagWise(layers.LeakyReLU(alpha=0.01))(x)
        x = BagWise(layers.Dropout(0.2))(x)
        x = BagWise(layers.BatchNormalization())(x)

        if self.config.n_fc_layers >= 2:
            x = BagWise(layers.Dense(k))(x)
            x = BagWise(layers.LeakyReLU(alpha=0.01))(x)
            x = BagWise(layers.Dropout(0.2))(x)
            x = BagWise(layers.BatchNormalization())(x)

        if self.config.mil_method is MILType.CAP_MI_NET:
            outputs = MILPool(pooling_mode=self.config.pooling_mode)()(x)
            outputs = ordinal_layer(outputs)

        if self.config.mil_method is MILType.MI_ATTENTION:
            n_att_weights = 128
            outputs = mil_attention_layer(x, n_att_weights, use_gated=False)
            outputs = ordinal_layer(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.config.learning_rate),
            loss=loss.get(self.config.ordinal_method),
            metrics=["accuracy", mae.get(self.config.ordinal_method)],
        )

        return model

    def train(
        self, train_generator: MILTabularDataGenerator, **kwargs,
    ) -> tf.keras.callbacks.History:

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        return self.model.fit(
            x=train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            epochs=self.config.epochs,
            callbacks=self.callbacks,
            **kwargs,
        )

    def evaluate(
        self, test_generator: MILTabularDataGenerator
    ) -> tf.keras.callbacks.History:
        """
        Evaluate the model run and record the metrics of interest
        """
        STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
        return self.model.evaluate(x=test_generator, steps=STEP_SIZE_TEST)


# Command line arguments -----------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--ex_file", required=True)
parser.add_argument("-i", "--i", default=1, required=False, type=int)
parser.add_argument("--output_dir", default="", required=False)
args = parser.parse_args()

# Read in example file (parameters) ------------------------------------------#
ex = pd.read_csv(args.ex_file)
ex["mil_method"] = [MILType[x] for x in ex["mil_method"]]
del ex["file"]

mod_args = ex.iloc[args.i]
print(mod_args)

Y_COL = "grade_differentiation"
BAG_COL = "case_number"

## Load data set -------------------------------------------------------------#

data_dir = "datasets/tma"
df = pd.read_csv(f"{data_dir}/tma_stage_imputations_1.0.csv")

df_train, df_test = train_test_split_bagwise(df, bag_col=BAG_COL, train_size=0.8)

## List hyperparameters ------------------------------------------------------#
hyperparameters = {
    "ordinal_method": [OrdinalType.CORAL, OrdinalType.CORN],
    "learning_rate": [0.01, 0.0005, 0.00001],
    "n_fc_layers": [1, 2],
    "fc_layer_size": [200, 400, 800],
}

hp = expand_grid(hyperparameters)

## 5-fold GridSearch ----------------------------------------------------------#

CV_FOLDS = 5
bag_info = df_train[[BAG_COL, Y_COL]].drop_duplicates().reset_index(drop=True)
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)

gs_metrics = np.zeros((hp.shape[0], CV_FOLDS))
j = 0
for train, val in skf.split(bag_info[BAG_COL], bag_info[Y_COL]):

    def get_rows_from_bag_ind(df: pd.DataFrame, bag_info, bag_ind):
        bags = bag_info[BAG_COL][bag_ind]
        df_ind = df[BAG_COL].isin(bags)

        return df.loc[
            df_ind,
        ]

    df_train_j = get_rows_from_bag_ind(df_train, bag_info, train)
    df_val_j = get_rows_from_bag_ind(df_train, bag_info, val)

    for i in range(hp.shape[0]):

        # Run model
        hp_args = dict(hp.iloc[i])
        config = TMAConfig(**mod_args, **hp_args)
        tma = TMARunner(config, save_results=False)

        loss, acc, mae = tma.run(df_train_j, df_val_j, verbose=2)

        # Save model performance
        gs_metrics[i, j] = mae if mod_args["metric"] == "mae" else acc

    j += 1


## Determine best performing parameters -----------------------------------------#

# reframe so that always maximizing metrics
mult_max = -1 if mod_args["metric"] == "mae" else 1
metric_avg = mult_max * gs_metrics.mean(axis=1)

best_hp = hp.iloc[np.argmax(metric_avg)]
print(best_hp)


## Re-fit on full training set --------------------------------------------------#

config = TMAConfig(**mod_args, **best_hp)
tma = TMARunner(config, save_results=True, output_dir=args.output_dir)
metrics = tma.run(df_train, df_test, verbose=2)

# Save metrics to single-row .csv
metrics2 = {i: j for (i, j) in zip(["loss", "accuracy", "mae"], metrics)}
metrics_df = pd.DataFrame({**mod_args, **metrics2}, index=[0])
metrics_df["mil_method"] = metrics_df["mil_method"].apply(lambda x: x.name)
metrics_df.to_csv(args.output_dir + config.file["metrics"])

# Save CV gridsearch summary to .csv
hp["metric_avg"] = metric_avg
hp.to_csv(args.output_dir + config.file["gridsearch"])
