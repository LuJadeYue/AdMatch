# AdMatch
Code for the paper "An adaptive semi-supervised method for remote sensing scene classification based on class-related complexity index"

### Set up datasets

To launch the training on EuroSAT (rgb or MS), it is necessary to download the corresponding datasets. The `root_dir` variable in the corresponding `datasets/eurosat_dataset.py` and `datasets/eurosat_rgb_dataset.py` files shall be adjusted according to the dataset path. 

### Train a model

To train a model on EuroSAT RGB by using EfficientNet B0 from scratch,  you can use: 

```
python train.py --dataset ucm --net efficientnet-b0
```

`--net ` can be used to specify the EfficientNet model, whilst `--dataset` can be used to specify the dataset. Use `eurosat_rgb` for EuroSAT RGB, `eurosat_ms` for EuroSAT MS, and `ucm` for UCM dataset.

Instead of starting the training from scratch, it is possible exploit a model pretrained on ImageNet. To do it,  you can use: 

```
python train.py --dataset eurosat_rgb --net efficientnet-b0 --pretrained
```

Information on additional flags can be obtained by typing:

```
python train.py --help
```

For additional information on training, including the use of single/multiple GPUs, please refer to [FixMatch-pytorch](https://github.com/LeeDoYup/FixMatch-pytorch).

