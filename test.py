import torch
from torch import nn
import logging
from torch import optim
from Model import CombinedModel
from utils import Data_Loader,mkfolder
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter
import timm
import datetime


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def plot_confusion_matrix(targets, predictions,info):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'./experiment/{info["target"]}/{info["fold"]}/{info["exp_time"]}/confusion_matrix/test.png')
    plt.close()

def test_epoch(device, info):

    Testset = Data_Loader(info['root_path'], 'test', info['fold'], info['target'])
    test_loader = DataLoader(Testset, shuffle=True, **info['loader_args'], drop_last=True)
    best_model = CombinedModel(1, 2, 64, 14, 64)
    best_model.load_state_dict(torch.load(f'{info["exp_path"]}/checkpoints/epoch/epoch{info["checkpoint"]}.pth'))
    best_model.to(device)
    best_model.eval()

    all_targets = []
    all_predictions_proba = []
    for batch in test_loader:
        with torch.no_grad():
            images, target, ID, tabular = batch['image'], batch['target'], batch['ID'], batch['tabular']
            images = images.to(device=device, dtype=torch.float32)
            images = images.unsqueeze(1)
            target = target.to(device=device, dtype=torch.float32).view(-1, 1)
            tabular = tabular.to(device=device, dtype=torch.float32)
            pred = best_model(images, tabular)
            logits = torch.sigmoid(pred)
            all_targets.extend(target.cpu().numpy())
            all_predictions_proba.extend(logits.cpu().numpy())

    # 转换为 NumPy 数组
    all_targets = np.array(all_targets)
    all_predictions_proba = np.array(all_predictions_proba)

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(all_targets, all_predictions_proba)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{info["exp_path"]}/ROC/Test_ROC.png')
    plt.close()


    # 使用先前确定的最佳阈值
    with open(f'{info["exp_path"]}/checkpoints/epoch/threshold.txt', 'r') as file:

        for line in file:
            line_epoch=line.split(',')
            if line_epoch[0] == f'epoch:{info["checkpoint"]}':
                optimal_threshold  = float(line_epoch[1])

    

    all_predictions = (all_predictions_proba >= optimal_threshold).astype(int)

    # 计算性能指标
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets, all_predictions,info)

    print(f'Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}')


if __name__ == '__main__':

    exp_path='./experiment/pattern ARE/fold0/2024-01-21 21:09:59.028202'

    exp_split=exp_path.split('/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = {
        'exp_path':exp_path,
        'root_path': '/home/wei/AVM236',
        'checkpoint':5,
        'fold':exp_split[3],
        'target':exp_split[2],
        'loader_args': {
            'batch_size': 2,
            'num_workers': 8,
            'pin_memory': True
        },
        'exp_time':exp_split[4],
    }
        
    test_epoch(device, info)