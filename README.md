# <a href="https://openreview.net/forum?id=GwXrGy_vc8m" target="_blank"> Estimating Noise Transition Matrix with Label Correlations for Noisy Multi-Label Learning </a> - Official PyTorch Code (NeurIPS 2022)

### Abstract:
In label-noise learning, the noise transition matrix, bridging the class posterior for noisy and clean data, has been widely exploited to learn statistically consistent classifiers. The effectiveness of these algorithms relies heavily on estimating the transition matrix. Recently, the problem of label-noise learning in multi-label classification has received increasing attention, and these consistent algorithms can be applied in multi-label cases. However, the estimation of transition matrices in noisy multi-label learning has not been studied and remains challenging, since most of the existing estimators in noisy multi-class learning depend on the existence of anchor points and the accurate fitting of noisy class posterior. To address this problem, in this paper, we first study the identifiability problem of the class-dependent transition matrix in noisy multi-label learning, and then inspired by the identifiability results, we propose a new estimator by exploiting label correlations without neither anchor points nor accurate fitting of noisy class posterior. Specifically, we estimate the occurrence probability of two noisy labels to get noisy label correlations. Then, we perform sample selection to further extract information that implies clean label correlations, which is used to estimate the occurrence probability of one noisy label when a certain clean label appears. By utilizing the mismatch of label correlations implied in these occurrence probabilities, the transition matrix is identifiable, and can then be acquired by solving a simple bilinear decomposition problem. Empirical results demonstrate the effectiveness of our estimator to estimate the transition matrix with label correlations, leading to better classification performance.

### Requirements:
* Python 3.8.10
* Pytorch 1.8.0 (torchvision 0.9.0)
* Numpy 1.19.5
* Scikit-learn 1.0.1


### Running the code:
To run the code use the provided scripts in the script folder. The dataset has to be placed in the data folder (should be done automatically). 
The arguments for running the code are as follows:

```
* batch_size, the batch size for learning
* num_classes, the number of classes in the dataset
* warmup_epoch, the number of warmup epoch for standard training
* nepochs, the number of training epoch for reweighting
* sample_epoch, the number of epoch for sample selection during warmup training
* sample_th, the threshold of sample selection
* nworkers, the number of workers in dataloder
* dataset, the name of the used dataset, i.e., 'voc2007', 'voc2012', or 'coco'
* seed, the random seed for label noise simulation
* root, the path for the data folder
* out, the path for the output folder
* noise_rate_p, the noise rate for positive class value
* noise_rate_n, the noise rate for negative class value
* lr, learning rate for learning
* weight-decay, the weight decay for learning,
* estimator, the method for estimating transition matrix, i.e.,'T', 'dualT', or 'ours'
* filter_outlier, the  parameter for T estimator and dualT estimator
```

### Citation:
If you find the code useful in your research, please consider citing our paper:

```
 @InProceedings{Li2022MLT,
  title = {Estimating Noise Transition Matrix with Label Correlations for Noisy Multi-Label Learning},
  authors = {Shikun Li and Xiaobo Xia and Hansong Zhang and Yibing Zhan and Shiming Ge and Tongliang Liu},
  year={2022},
  booktitle ={36th Conference on Neural Information Processing Systems (NeurIPS 2022)},
 } 
```

Note: Our implementation uses parts of some public codes [1-2].

[1] "Dual T: Reducing Estimation Error for Transition Matrix in Label-noise Learning" https://github.com/a5507203/dual-t-reducing-estimation-error-for-transition-matrix-in-label-noise-learning

[2] "Classification with Noisy Labels by Importance Reweighting" https://github.com/xiaoboxia/Classification-with-noisy-labels-by-importance-reweighting
