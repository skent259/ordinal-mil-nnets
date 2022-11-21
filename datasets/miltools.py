import numpy as np
import pandas as pd


class MILDataSetConverter(object):
    """
    An object that can convert a data set to a MIL data set. The several options 
    are wrapped into a single object that can be used. 

    TODO: investigate why so slow on the aes data set. 
    """

    def __init__(
        self, df: pd.DataFrame, y_col: str, convert_type: str, shuffle: bool = True
    ):
        self.df = df.copy()
        self.y_col = y_col
        self.convert_type = convert_type
        self.shuffle = shuffle

    @property
    def class_indices(self):
        classes = np.unique(self.df[self.y_col])
        return dict(zip(classes, np.arange(len(classes))))

    def convert(self, bag_size: int, wr: float = 1.0, seed: int = 0) -> pd.DataFrame:
        """
        Main method that converts the given data set based on specified options. 
        Mostly just calls the correct underlying method. 
        """

        if self.convert_type == "random":
            return self.ds_to_bag_level_random(bag_size, seed)
        if self.convert_type == "wr":
            return self.ds_to_bag_level_by_wr(bag_size, wr, seed)
        return None

    def ds_to_bag_level_random(self, bag_size: int, seed: int = 0) -> pd.DataFrame:
        """
        Simplest implementation. Generate bags based on the order of a (potentially 
        shuffled) data set. 
        """

        if self.shuffle:
            df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)

        new_df = {i: self.to_bag_column_random(df[i], bag_size) for i in df.columns}
        return pd.DataFrame(new_df)

    def ds_to_bag_level_by_wr(
        self, bag_size: int, wr: float, seed: int = 0
    ) -> pd.DataFrame:
        """
        Implementation that uses the witness rate to randomly pull the needed instances
        first, then fill in with the remaining. Uses the assumption that the "witness" 
        instance in the maximal class
        """

        if self.shuffle:
            df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            df = self.df.reset_index(drop=True)

        bag_indices = self.bag_column_from_wr(df, bag_size, wr)

        new_df = {i: [list(df[i][ind]) for ind in bag_indices] for i in df.columns}
        return pd.DataFrame(new_df)

    def bag_column_from_wr(self, df: pd.DataFrame, bag_size: int, wr: float) -> list:
        """
        Algorithm that splits instances into bags that satisfy the given witness rate.
        Briefly, it will fill bags with the requied witness and non-witness instances randomly, 
        provided that there are enough instances. This process is done for each class equally until 
        that class can't be used. It will end when no more bags can be filled. 
        """
        n_witness_inst = round(wr * bag_size)
        n_nonwitness_inst = bag_size - n_witness_inst

        y_rem = df[self.y_col].apply(lambda x: self.class_indices[x])
        classes_rem = set(self.class_indices.values())
        bags = {}
        i = 0

        while len(classes_rem) > 0:

            for _class in reversed(list(classes_rem)):

                enough_witness_instances = len(y_rem[y_rem == _class]) > n_witness_inst
                enough_nonwitness_instances = (
                    len(y_rem[y_rem <= _class]) - n_witness_inst > n_nonwitness_inst
                )

                if not enough_witness_instances or not enough_nonwitness_instances:
                    classes_rem -= set([_class])
                if enough_witness_instances and enough_nonwitness_instances:
                    # add witness instances
                    witness_inst = y_rem[y_rem == _class].sample(n=n_witness_inst)
                    y_rem = y_rem.drop(index=witness_inst.index)

                    # add nonwitness instances
                    nonwitness_inst = y_rem[y_rem <= _class].sample(n=n_nonwitness_inst)
                    y_rem = y_rem.drop(index=nonwitness_inst.index)

                    bags[i] = pd.concat([witness_inst, nonwitness_inst])
                    i += 1

        return [bags[i].index for i in bags]

    def to_bag_column_random(self, x: pd.Series, bag_size: int) -> pd.Series:
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

        return pd.Series(bag_col)

