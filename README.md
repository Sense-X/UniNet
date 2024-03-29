## Pytorch implementation of [UniNet (ECCV 2022)](https://arxiv.org/abs/2207.05420)

![tenser](assets/backbone.png)
![performance](assets/acc.png)

This repo is the offcial implementation of the paper [UniNet: Unified Architecture Search with Convolution, Transformer, and MLP](https://arxiv.org/abs/2207.05420)

```
@article{UniNet,
  author  = {Jihao Liu, Xin Huang, Guanglu Song, Yu Liu, Hongsheng Li},
  journal = {arXiv:2207.05420},
  title   = {UniNet: Unified Architecture Search with Convolution, Transformer, and MLP},
  year    = {2022},
}
```

### Update
20/12/2022 Update pretrained models.

25/10/2022 Update the source code.

#### Environment
The code is tested with ```torch==1.11``` and ```timm==0.5.4```.


### Availble models
|Models | Params (M) | FLOPs (G) | Pretrain Epochs | Top-1 Acc. | ckpt |
| :---: | :---: | :---: | :---: | :---: | :---: |
| UniNet-B1 | 11.5 | 1.1 | 300 | 81.0 | [ckpt](https://drive.google.com/drive/folders/14gp-Vtmtd3MNNlrmYtF5FcUi0rm4CaGi?usp=share_link)|

### Run experiments

Currently, we supporting running experiments with slurm.
You can reproduce the results of UniNet-B1 as follows: 

```sh exp/b1/run.sh partition 8```
