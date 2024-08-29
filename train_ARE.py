import torch
from torch import nn
import logging
from torch import optim
from Model import CombinedModel
from utils import Data_Loader,mkfolder,FocalLoss
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



def plot_confusion_matrix(targets, predictions,info,epoch):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/confusion_matrix/epoch{epoch}.png')
    plt.close()

def plot_ROC(all_targets,all_predictions_proba,info,epoch):
    fpr, tpr, _ = roc_curve(all_targets, all_predictions_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr,
            tpr,
            color='blue',
            lw=2,
            label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/ROC/Epoch{epoch}.png')
    plt.close()


def start_train(model, device, info):
    best_f1 = 0.0
    Trainset = Data_Loader(info['root_path'], 'train', info['fold'],
                           info['target'])
    Valset = Data_Loader(info['root_path'], 'val', info['fold'],
                         info['target'])
    train_loader = DataLoader(Trainset,
                              shuffle=True,
                              **info['loader_args'],
                              drop_last=True)
    val_loader = DataLoader(Valset,
                            shuffle=True,
                            **info['loader_args'],
                            drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=100)
    criterion = FocalLoss()
    # weight=torch.tensor([3]).to(device)
    # criterion=nn.BCEWithLogitsLoss(pos_weight=weight)
    # # criterion=nn.BCEWithLogitsLoss()
    writer = SummaryWriter(log_dir='logs')

    for epoch in range(1, info['epochs'] + 1):
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        progress_bar = tqdm(enumerate(train_loader), total=num_batches)

        for batch_idx, batch in progress_bar:
            images, target, ID, tabular = batch['image'], batch[
                'target'], batch['ID'], batch['tabular']
            images = images.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32).view(-1, 1)
            tabular = tabular.to(device=device, dtype=torch.float32)
            images = images.unsqueeze(1)
            pred = model(images, tabular)

            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_description(
                f'Epoch {epoch}/{info["epochs"]}, Batch {batch_idx}/{num_batches}'
            )

        average_loss = total_loss / num_batches
        print(f'Epoch {epoch} - Average Loss: {average_loss:.4f}')
        writer.add_scalar('Training Loss', average_loss, epoch)
        if epoch % 1 == 0:
            model.eval()
            all_targets = []
            all_predictions_proba = []
            for batch in val_loader:
                images, target, ID, tabular = batch['image'], batch[
                    'target'], batch['ID'], batch['tabular']
                images = images.to(device=device, dtype=torch.float32)
                images = images.unsqueeze(1)
                target = target.to(device=device,
                                   dtype=torch.float32).view(-1, 1)
                tabular = tabular.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    pred = model(images, tabular)
                pred = torch.sigmoid(pred)
                all_targets.extend(target.cpu().numpy())
                all_predictions_proba.extend(pred.cpu().numpy())

            all_predictions_proba = np.array(all_predictions_proba)
            all_targets = np.array(all_targets)

            precision, recall, thresholds = precision_recall_curve(
                all_targets, all_predictions_proba)
            f1_scores = 2 * precision * recall / (precision + recall)
          
            optimal_f1 = np.nanmax(f1_scores)
            optimal_idx=np.where(f1_scores==optimal_f1)[0][0]
            optimal_threshold = thresholds[optimal_idx]
            optimal_predictions = (all_predictions_proba
                                   >= optimal_threshold).astype(int)

            accuracy = accuracy_score(all_targets, optimal_predictions)
            f1_optimal = f1_scores[optimal_idx]
            print(optimal_predictions)
            plot_confusion_matrix(all_targets,optimal_predictions,info,epoch)

            writer.add_scalar('Validation Accuracy', accuracy, epoch)
            writer.add_scalar('Validation F1 Score', f1_optimal, epoch)
            print(
                f'Epoch {epoch} - Validation Accuracy: {accuracy:.4f}, Validation F1 Score: {f1_optimal:.4f},optimal threshold:{optimal_threshold}'
            )

            plot_ROC(all_targets,all_predictions_proba,info,epoch)
 
            if f1_optimal >= best_f1:
                best_f1 = f1_optimal
                torch.save(model.state_dict(), f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/checkpoints/best_model.pth')
                with open(f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/checkpoints/threshold.txt', 'w') as file:
                    file.write(f'{optimal_threshold}\n')
            
            torch.save(model.state_dict(), f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/checkpoints/epoch/epoch{epoch}.pth')
            with open(f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/checkpoints/epoch/threshold.txt', 'a') as file:
                file.write(f'epoch:{epoch},{optimal_threshold}\n')

        scheduler.step()

    writer.close()
    print(f'Best Val F1 score:{best_f1}')
    

def start_test(device, info):
    Testset = Data_Loader(info['root_path'], 'test', info['fold'], info['target'])
    test_loader = DataLoader(Testset, shuffle=True, **info['loader_args'], drop_last=True)
    best_model = CombinedModel(1, 2, 64, 14, 64)
    best_model.load_state_dict(torch.load(f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/checkpoints/best_model.pth'))
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
    plt.savefig(f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/ROC/Test_ROC.png')
    plt.close()

    # 使用先前确定的最佳阈值
    with open(f'./experiment/{info["target"]}/fold{info["fold"]}/{info["exp_time"]}/checkpoints/threshold.txt', 'r') as file:
        line = file.readline()  # 读取文件的第一行
    try:
        optimal_threshold  = float(line.strip())  # 转换文本行为浮点数，并去掉可能的空白字符
    except ValueError:
        print(f"无法转换行 '{line.strip()}' 为浮点数。")
    all_predictions = (all_predictions_proba >= optimal_threshold).astype(int)

    # 计算性能指标
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets, all_predictions, info,'test')

    print(f'Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}')


if __name__ == '__main__':

    info = {
        'root_path': '/home/wei/AVM236',
        'threshold': 0.5,
        'epochs': 10,
        'fold':0,
        'target':'(ARE)',
        'loader_args': {
            'batch_size': 2,
            'num_workers': 8,
            'pin_memory': True
        },
        'exp_time':datetime.datetime.now()
    }

    mkfolder(info['fold'],info['target'],info['exp_time'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = CombinedModel(1, 2, 64, 14, 64)
    Model.to(device)
    start_train(Model, device,info)
    start_test(device,info)
