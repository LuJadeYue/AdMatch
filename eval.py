from __future__ import print_function, division
import os

import torch
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
from operator import truediv

from utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader


def reports(y_test, y_pred, name):
    if name == 'aid':
        target_names = ['Airport', 'BareLand', 'BaseballField', 'Beach',
                        'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
                        'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain',
                        'Church',
                        'Park', 'Parking', 'Playground', 'Pond', 'Port',
                        'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential',
                        'Square', 'Stadium', 'StorageTanks', 'Viaduct']
    elif name == 'ucm':
        target_names = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral",
                        "denseresidential", "forest", "freeway", "golfcourse", "harbor"]
    elif name == 'eurosat_ms':
        target_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                        'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River',
                        'SeaLake']

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names, digits=6)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)

    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path", type=str, default="./saved_models/fixmatch/model_best.pth"
    )
    parser.add_argument("--use_train_model", action="store_true")

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="WideResNet")
    parser.add_argument("--net_from_name", type=bool, default=False)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    """
    Data Configurations
    """
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training. "
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = (
        checkpoint["train_model"] if args.use_train_model else checkpoint["eval_model"]
    )

    _net_builder = net_builder(
        args.net,
        args.net_from_name,
        {
            "depth": args.depth,
            "widen_factor": args.widen_factor,
            "leaky_slope": args.leaky_slope,
            "dropRate": args.dropout,
        },
    )

    _eval_dset = SSL_Dataset(
        name=args.dataset, train=False, data_dir=args.data_dir, seed=args.seed
    )

    eval_dset_basic = _eval_dset.get_dset()
    args.num_classes = _eval_dset.num_classes
    args.num_channels = _eval_dset.num_channels

    net = _net_builder(num_classes=args.num_classes, in_channels=args.channels)
    net.load_state_dict(load_model)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    eval_loader = get_data_loader(eval_dset_basic, args.batch_size, num_workers=1)

    pred_test = []
    truth_test = []

    acc = 0.0
    with torch.no_grad():
        for _, image, target in eval_loader:
            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)
            pred_test.extend(np.array(logit))
            truth_test.extend(np.array(target))
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()

    print(f"Test Accuracy: {acc / len(eval_dset_basic)}")
    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(truth_test, pred_test,
                                                                                           args.dataset)

import seaborn as sn
import pandas as pd
import matplotlib as plt
if datasets == 'aid':
    target_names = ['Airport', 'BareLand', 'BaseballField', 'Beach',
                    'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
                    'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain',
                    'Church',
                    'Park', 'Parking', 'Playground', 'Pond', 'Port',
                    'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential',
                    'Square', 'Stadium', 'StorageTanks', 'Viaduct']
elif datasets == 'ucm':
    target_names = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral",
                    "denseresidential", "forest", "freeway", "golfcourse", "harbor"]
elif datasets == 'eurosat_ms':
    target_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River',
                    'SeaLake']
df_cm = pd.DataFrame(confusion.numpy(),
                     index = [i for i in list(target_names.keys())],
                     columns = [i for i in list(target_names.keys())])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap="BuPu")

import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion, classes=target_names, normalize=False, title='Normalized confusion matrix')
