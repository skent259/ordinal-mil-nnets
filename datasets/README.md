# Datasets

In this directory, each subfolder contains the code for an different data set used. This includes scripts for downloading, processing, conversion to MIL structure (if necessary), and train/test splitting for reproducibility.

## Data set information

[**AFAD**](https://afad-dataset.github.io): The Asian Face Age Data set contains 165,501 images of Asian faces with an age label ranging 15–40 years (Niu et al., 2016). Following Shi et al. (2022), we use 13 age labels ranging 18–30 years. We split the data two times randomly into 80% training, 15% testing, and 5% validation. To provide MIL structure, we group images (within the train, test, and validation sets) five times randomly into bags of size 4, with a witness rate of 50%; non-witness images have an instance label lower than the bag label. The training data set is then balanced by the bag label via random under-sampling. This process results in 10 distinct data set replications, having about 10700, 4500, and 1500 bags for training, testing, and validation, respectively.

[**FGNET**](https://yanweifu.github.io/FG_NET_data/index.html): The FG-NET data set is a small benchmark age label data set with 1002 color face images from 82 subjects (Lanitis et al., 2002). We grouped these ages into six categories based on age ranges [0–3, 4–11, 12– 16, 17–24, 25–40, 40+], as in Vargas et al. (2020). The data is split into training, testing, and validation, then grouped into bags according to the same process as the AFAD data set described above; the only difference is that five random splits are completed, and random over-sampling (via bootstrap) is done on the training data set after grouping into bags. This process results in 25 distinct data set replications, having about 250, 35, and 12 bags for training, testing, and validation, respectively.

[**BCNB**](https://bupt-ai-cz.github.io/BCNB/): The Breast Cancer Core-needle Biopsy data set contains 1,058 whole slide images from patient biopsies with tumor annotations (Xu et al., 2021). The ordinal outcome used is the metastatic status of auxiliary lymph nodes (ALN), grouped as [0, 1–2, 2+]. The MIL structure comes from using 256 × 256 pixel, non-overlapping patches (instances) from each WSI (bag), generated from the annotated tumor regions (Xu et al., 2021). Since some WSIs had as many as 300 patches, a maximum of 30 patches are used in each bag. We split the data five times randomly into 60% training, 20% testing, and 20% validation. The training data set is then balanced by the bag label via random under-sampling, and the validation set is limited to 50 bags in total. This process results in 5 distinct data set replications, having about 500, 210, and 50 bags for training, testing, and validation, respectively.

[**AMREV**](http://sifaka.cs.uiuc.edu/~wang296/Data/): The Amazon sentiment data set used here contain product reviews and a score rating of [1, 2, 3, 4, 5] from Amazon.com in the TV category only (Wang et al., 2011). Each review is a bag and each sentence is an instance with the bag. The only pre-processing done on the raw review text is removing HTML tags and excess new-line characters. We split the data 10 times randomly into 1,000 training, 2,000 testing, and 200 validation bags from the 114,763 reviews in the TV category.

[**IMDB**](https://ai.stanford.edu/~amaas/data/sentiment/): The IMDB movie review data set contains 50,000 movie reviews and a score rating downloaded from IMDB (imdb.com) (Maas et al., 2011). Following Section 2.5, each review is a bag and each sentence is an instance with the bag. The IMDB data set is processed and split in the same manner as AMREV, leaving 10 distinct replications with 1,000 training, 2,000 testing, and 200 validation reviews in each.

See also: 
* `models/dataset.py` contains some code to capture data set structure used in experiments and test scripts
* `experiment/` contains specific parameters for each individual data set used in the experiment. 

## References

[1] Lanitis, A., Taylor, C. J., & Cootes, T. F. (2002). Toward automatic simulation of aging effects on face images. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(4), 442–455.

[2] Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142–150.

[3] Niu, Z., Zhou, M., Wang, L., Gao, X., & Hua, G. (2016). Ordinal regression with multiple output CNN for age estimation. *2016 IEEE Conference on Computer Vision and Pattern Recognition*, 4920–4928. https://doi.org/10.1109/CVPR.2016.532

[4] Shi, X., Cao, W., & Raschka, S. (2022). Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. *ArXiv Preprint ArXiv:2111.08851*. http://arxiv.org/abs/2111.08851

[5] Vargas, V. M., Gutiérrez, P. A., & Hervás-Martínez, C. (2020). Cumulative link models for deep ordinal classification. *Neurocomputing*, 401, 48–58. https://doi.org/10.1016/j.neucom.2020.03.034

[6] Wang, H., Wang, C., Zhai, C., & Han, J. (2011). Learning online discussion structures by conditional random fields. *Proceedings of the 34th International Acm Sigir Conference on Research and Development in Information Retrieval*, 435–444. https://doi.org/10.1145/2009916.2009976

[7] Xu, F., Zhu, C., Tang, W., Wang, Y., Zhang, Y., Li, J., Jiang, H., Shi, Z., Liu, J., & Jin, M. (2021). Predicting axillary lymph node metastasis in early breast cancer using deep learning on primary tumor biopsy slides. *Frontiers in Oncology*, 11, 759007. https://doi.org/10.3389/fonc.2021.759007
