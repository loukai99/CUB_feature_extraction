# CUB_feature_extraction
I extract the multiview features of CUB dataset. It contains two views, one of which is from image features extracted by GoogleNet(2048-dimension) and the other is from text features using doc2vec(300-dimension).

The feature extraction method refer to the paper [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916) by Han Zhang*, Tao Xu*, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas.

## Access to processed data

You can download the [processed data](https://drive.google.com/file/d/1QdrOd_eVdAM_WZB5QAgb-tUWWQO69VgC/view?usp=sharing)(has extract two views feature and save the feature in .mat file) or download the raw data from [CUB's official website](https://data.caltech.edu/records/20098) then run the `CUB-feature_extraction.py`.
