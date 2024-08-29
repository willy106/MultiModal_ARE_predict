import os


def mkfolder(fold,target,time):
    
    checkpoint_path=f'./experiment/{target}/fold{fold}/{time}/checkpoints'
    epoch_path=f'./experiment/{target}/fold{fold}/{time}/checkpoints/epoch'
    confusion_path=f'./experiment/{target}/fold{fold}/{time}/confusion_matrix'
    ROC_path=f'./experiment/{target}/fold{fold}/{time}/ROC'
    
    os.makedirs(checkpoint_path,exist_ok=True)
    os.makedirs(epoch_path,exist_ok=True)
    os.makedirs(confusion_path,exist_ok=True)
    os.makedirs(ROC_path,exist_ok=True)