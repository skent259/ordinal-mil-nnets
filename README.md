# Ordinal, Multiple Instance Deep Learning

Many approaches have been proposed for deep learning for ordinal labels or multiple-instance learning (MIL) structure separately. However, few works have proposed a deep learning framework for ordinal labels and MIL structure together. This repository contains tensorflow-based code for implementing a deep learning framework for ordinal, MIL data, and includes experiments from the manuscript "Ordinal, Multiple Instance Deep Learning" by Sean Kent and Menggang Yu.

As a quick reference for getting started:

* The experiments can be run via the `run.sh` file, however this will take considerable CPU time. We recommend that you run in a high-throughput environment in batches (see `run.sh` for inspiration)
* Raw model code is contained in the `models/` directory, in addition to helper code that allows for running the experiment and application. 
* If you are looking for a quick way to re-use the underlying code for your own deep learning, the examples in the `test/` directory are a useful place to start. 
* Analysis, including code to replicate the figures and tables, is in the `analysis/` directory for the main experiment and the `application/` directory for an application to TMA data. 
* Data (open-source) can be downloaded and processed in individual folders under the `datasets/` directory. 
* The `condor/` directory can be ignored. It was used to run simulations in the HTCondor (high-throughput) environment.


## Methods compared

A full description of the methods is given in the manuscript. Where code was not available in a package, but the implementation was present elsewhere, we have copied the code into a directory with credit given below.

| Method name              | Type    | Directory                                                 | Reference     |
| ------------------------ | ------- | --------------------------------------------------------- | ------------- |
| mi-net                   | MIL     | `mil_nets/`                                               | [1], [2], [3] |
| MI-net                   | MIL     | `mil_nets/`                                               | [2]           |
| MI-net (DS)              | MIL     | `mil_nets/`                                               | [2]           |
| MI-net (Attention)       | MIL     | `mil_attention/`                                          | [3]           |
| MI-net (Gated-attention) | MIL     | `mil_attention/`                                          | [3]           |
| CORAL                    | Ordinal | NA, see "coral-ordinal",  "coral-pytorch" python packages | [4]           |
| CORN                     | Ordinal | NA, see "coral-ordinal",  "coral-pytorch" python packages | [5]           |
| CLM QWK                  | Ordinal | `clm_qwk/`                                                | [6]           |

* `mil_nets/`: Originally used <https://github.com/yanyongluan/MINNs> for inspiration/testing, but final code uses layers from tensorflow.
* `mil_attention/`: <https://keras.io/examples/vision/attention_mil_classification/>
* `clm_qwk/` <https://github.com/ayrna/deep-ordinal-clm>
* "coral-ordinal": <https://github.com/ck37/coral-ordinal>
* "coral-pytorch": <https://github.com/Raschka-research-group/coral-pytorch>

## References

[1] Ramon, J., & De Raedt, L. (2000). Multi instance neural networks. Proceedings of the ICML-2000 Workshop on Attribute-Value and Relational Learning, 53–60.

[2] Wang, X., Yan, Y., Tang, P., Bai, X., & Liu, W. (2018). Revisiting multiple instance neural networks. Pattern Recognition, 74, 15–24. https://doi.org/10.1016/j.patcog.2017.08.026

[3] Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple instance learning. Proceedings of the 35th International Conference on Machine Learning, 2127–2136. https://proceedings.mlr.press/v80/ilse18a.html

[4] Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. Pattern Recognition Letters, 140, 325–331. https://doi.org/10.1016/j.patrec.2020.11.008

[5] Shi, X., Cao, W., & Raschka, S. (2022). Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. ArXiv Preprint ArXiv:2111.08851. http://arxiv.org/abs/2111.08851

[6] Vargas, V. M., Gutiérrez, P. A., & Hervás-Martínez, C. (2020). Cumulative link models for deep ordinal classification. Neurocomputing, 401, 48–58. https://doi.org/10.1016/j.neucom.2020.03.034
