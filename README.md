# [ECCV20] Content-Consistent Matching for Domain Adaptive Semantic Segmentation

This is a PyTorch implementation of [CCM](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590426.pdf).  


### News: [GTA-4K](https://github.com/Solacex/CCM/tree/master/GTA5-4K) list is available! 
A smaller subset of GTA5 dataset that shares higher layout similarites with Cityscapes. 


### Prerequisites

To install requirements:

```setup
pip install -r requirements.txt
```

- Python 3.6
- GPU Memory: 24GB for the first stage(Source-only Model), and 12GB for the second stage
- Pytorch 1.4.0



## Getting Started

1. Download the dataset [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Cityscapes](https://www.cityscapes-dataset.com/).
2. Download the ImageNet-pretrained Model [[Link](https://drive.google.com/open?id=13kjtX481LdtgJcpqD3oROabZyhGLSBm2)].
3. Download the Source-only Model [Link](https://drive.google.com/file/d/1-52RggreImwr_BVcGzm41j0mchxclwwu/view?usp=sharing). 

## Training

To train the source-only model:

```train
CUDA_VISIBLE_DEVICES=0 python so_run.py
```

To train the adaptation model:

```train
CUDA_VISIBLE_DEVICES=0 python run.py
```

## Evaluation

To perform evaluation on a multiple models under a directory:

```eval
python eval.py --frm your_dir 
```

To perform evaluation on single model:

```eval
python eval.py --frm model.pth --single
```



## Citation 

If you find it helpful, please consider citing: 

```
@inproceedings{li2020content,
  title={Content-consistent matching for domain adaptive semantic segmentation},
  author={Li, Guangrui and Kang, Guoliang and Liu, Wu and Wei, Yunchao and Yang, Yi},
  booktitle={European Conference on Computer Vision},
  pages={440--456},
  year={2020},
  organization={Springer}
}
```

