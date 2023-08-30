import sys

[sys.path.append(i) for i in [".", ".."]]  # need to access datasets and models module

import coral_ordinal as coral
import numpy as np
import pandas as pd
import tensorflow as tf

from application.config import TMAConfig
from models.architecture import MILType, OrdinalType
from models.generators import MILTabularDataGenerator
from models.mil_attention.layer import mil_attention_layer
from models.mil_nets.layer import BagWise, MILPool

## Make class to load attention weights --------------------------------------#

Y_COL = "grade_differentiation"
BAG_COL = "case_number"


class AttentionWeightsLoader(object):
    def __init__(self, df: pd.DataFrame, config: TMAConfig, class_mode: str = "sparse"):
        self.df = df
        self.config = config
        self.x_cols = list(df.columns[2:])
        self.class_mode = class_mode

    def run(self) -> np.array:
        self.make_generator()
        self.load_model()
        self.construct_sub_models()
        return self.compute_attention_weights()

    def make_generator(self):
        self.df_generator = MILTabularDataGenerator(
            dataframe=self.df,
            x_cols=self.x_cols,
            y_col=Y_COL,
            bag_col=BAG_COL,
            batch_size=1,
            shuffle=False,
            class_mode=self.class_mode,
        )

    def load_model(self):
        model_path = "results/tma/" + self.config.file["model"]
        self.model = tf.keras.models.load_model(model_path)

    def construct_sub_models(self):
        self.pre_attention_model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer(index=-3).output
        )

        self.attention_layer = self.model.get_layer(index=-2)

        self.attention_weights_model = tf.keras.Model(
            inputs=self.attention_layer.input,
            outputs=self.attention_layer.get_layer(index=-2).output,
        )

    def compute_attention_weights(self):
        x = self.pre_attention_model.predict(self.df_generator)
        return self.attention_weights_model.predict(x)


## Load data set -------------------------------------------------------------#

data_dir = "datasets/tma"
df = pd.read_csv(f"{data_dir}/tma_stage_imputations_1.0.csv")

## Load experiment file ------------------------------------------------------#

ex = pd.read_csv("application/params/example-tma-7.0.1.csv")
ex["mil_method"] = [MILType[x] for x in ex["mil_method"]]
del ex["file"]

ex = ex[ex["mil_method"] == MILType.MI_ATTENTION]

## Compute attention weights -------------------------------------------------#

for i in range(len(ex)):
    mod_args = ex.iloc[i]
    config = TMAConfig(**mod_args)
    att_loader = AttentionWeightsLoader(df, config, class_mode="sparse")
    att = att_loader.run()

    out_file = config.file["metrics"].replace("_metrics.csv", "_att-weights.csv")
    np.savetxt("results/tma/" + out_file, att, delimiter=",")
