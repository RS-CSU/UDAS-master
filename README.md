# Unsupervised Domain Adaptation for Semantic Segmentation of High-Resolution Remote Sensing Imagery Driven by Category-Certainty Attention

Pytorch implementation of our method for cross-domain semantic segmentation of the high-resolution remote sensing imagery. 

Contact: Jingru Zhu (zhujingru1012@163.com) and Jie Chen (cj2011@csu.edu.cn)

## Paper
[Unsupervised Domain Adaptation for Semantic Segmentation of High-Resolution Remote Sensing Imagery Driven by Category-Certainty Attention](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9667523) <br />
Jie Chen , Member, IEEE, Jingru Zhu , Ya Guo , Geng Sun, Yi Zhang, and Min Deng <br />
IEEE Transactions on Geoscience and Remote Sensing, 2022.

Please cite our paper if you find it useful for your research.

```
@inproceedings{UDAS_2022,
  author = {Jie Chen and Jingru Zhu and Ya Guo and Geng Sun and Yi Zhang and and Min Deng},
  booktitle = {IEEE Transactions on Geoscience and Remote Sensing},
  title = {Unsupervised Domain Adaptation for Semantic Segmentation of High-Resolution Remote Sensing Imagery Driven by Category-Certainty Attention},
  year = {2022}
}
```

## Example Results


## Quantitative Reuslts


## Installation
* Install PyTorch from http://pytorch.org with Python 3.6 and PyTorch 1.8.0

* Clone this repo
```
git clone https://github.com/RS-CSU/UDAS-master
cd UDAS-master
```
## Dataset
* Download the [Potsdam Dataset](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx/) as the source domain, and put it in the `dataset/Potsdam` folder

* Download the [Vaihingen Dataset](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx/) as the target domain, and put it in the `data/Vaihingen` folder

## Testing
* Download the pre-trained [Potsdam_best model](http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth) and put it in the `checkpoints_potsdam` folder
* Download the pre-trained [pot2vai model](http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth) and put it in the `checkpoints_pot2vai` folder

* Test the model and results will be saved in the `results` folder

```
python test.py
```

## Training Examples
* Train the Potsdam-to-Vaihingen model

```
python train_pot2vai_9_5.py
```

## Related Implementation and Dataset
* 

## Acknowledgment
This code is heavily borrowed from [Pytorch-AdaptSegNet](https://github.com/wasidennis/AdaptSegNet).

## Note
The model and code are available for non-commercial research purposes only.

* 07/2022: code released




