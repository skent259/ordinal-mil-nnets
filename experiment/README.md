# Experiment

The experimental design considers the following factors:

* Ordinal method
* MIL method and pooling function 
* Data set
* Learning rate
* Data splitting replication (random)

The individual options for each design factor are provided in the `{dataset}-params.py` files. In general, all combinations of these options are used, creating many individual models that require running. In total, 7,200 independent models were trained (1,200 for AFAD, 3,000 for FGNET, 600 for BCNB, 1,200 for AMREV, and 1,200 for IMDB). 