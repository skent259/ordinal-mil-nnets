# Models

This module contains the main code for implementing the MIL and ordinal neural networks, in addition to some helper code for running the experiment.

The following code is useful for general applications:

* The `clm_qwk/` directory contains `CLM` keras activation layer and losses needed, plus a resnet block file
* The `mil_attention/` directory contains the `MILAttentionLayer` keras layer
* The `mil_nets/` directory contains the keras layers needed for mi-net and MI-net, using existing layers. In comparison to other approaches, this doesn't require equal bag size.

The following code is relevant to the simulations run in the experiment:

* The `architecture.py` file sets up model architectures programmatically
* The `dataset.py` file contains classes to handle data set parameters and logic
* The `experiment.py` file sets up the `ExperimentRunner` object based on a `ExperimentConfig` data class. This has the logic for the whole experiment.
* The `generators.py` file build generators that can be used in training. Relevant to others is that these generators are set up to not need equal bag size. See details within


