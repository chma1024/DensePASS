# DensePASS

## Dense Panoramic Semantic Segmentation via Unsupervised Domain Adaptation

![Example segmentation](DensePASS_3D_lane.png?raw=true "Example segmentation")

### Installation

### Dataset

For the basic setting of this work, please prepare datasets of Cityscapes, Stanford2D3D, and DensePASS.

Our proposed DensePASS is available at [Google Drive](https://drive.google.com/file/d/1deXWKCKmo6ecsVcqxaCdESCSSclTlfze/view?usp=sharing).

The DensePASS dataset has 100 panoramic images and annotations for evaluation.

The other unlabeled images could be found at [WildPASS](https://github.com/elnino9ykl/WildPASS).

### Training

'train.py' in S_R, A_S and S_A_R are used for training on outdoor dataset with different modules settings. 

The training configurations can be adjusted at 'train.py' or by parameters like --model DANet.

An example of training is 'python train.py'.

### Acknowledgements
This code is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet).


