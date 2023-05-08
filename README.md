# Algorithmc Fairness Improvement of Bias-Contrastive Learning

The code mainly builds on [the pytorch implementation](https://github.com/mqraitem/Bias-Mimicking) of [Bias Mimicking: A simple sampling approach for Bias Mitigation](https://arxiv.org/pdf/2209.15605.pdf) with some small changes.

## Setup

### Set up conda environment  
```
conda create -n xx python=3.8
conda activate xx
```

### Install packages

* pytorch=1.10.1 
* scipy
* tqdm 
* scikit-learn

### Prepare dataset.

- CelebA  
Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset under `data/celeba`

- UTKFace  
Download [UTKFace](https://susanqq.github.io/UTKFace/) dataset under `data/utk_face`

- CIFAR10  
Download [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset under `data/cifar10`


## Train.

From the main directory, run: 

```
python train_[DATASET]/train_[DATASET]_[METHOD].py --exp_name [EXP_NAME] --task [TASK] --seed [SEED] 
```


where mode refers to whether the distribution is left as is/undersampled/upweighted/oversampled when training the predictive linear layer. 

## Acknowledgements

The code for non sampling methods builds on [this work](https://github.com/grayhong/bias-contrastive-learning). Furthermore, the code for GroupDRO is obtained from [this work](https://github.com/kohpangwei/group_DRO)
