import numpy as np
import torch

import SimpleITK as sitk
from skimage.util import montage

from torch.utils.data import Dataset
import torch.utils.data
import pandas as pd
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
# from preprocess_tabular import pre_tabular
from sklearn.preprocessing import MinMaxScaler

class Data_Loader(Dataset):

    def __init__(self, root, mode, fold, targets):

        self.targets = targets

        if mode == 'test':
            file_name = f"{mode}.xlsx"
        else:
            file_name = f"fold{fold}_{mode}.xlsx"

        df = pd.read_excel(
            os.path.join('./Dataset/TabularData_pattern', file_name))
        df['Hx no. (VGH)']=df['Hx no. (VGH)'].str.strip()
        df=df[df['Hx no. (VGH)'] != '2754059-6']
        df.reset_index(drop=True, inplace=True)
        self.tabular=pre_tabular(df,self.targets,mode)
       
    #     if targets == '(ARE)':
    #         if mode=='test':
    #             self.tabular = self.tabular.drop([
    #                 'pattern ARE', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
    #                 'Basal ganglia', 'Brain stem', 'Cerebellum', 'Brain stem','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
    #     'GKS'], axis=1)
    #         else:
    #             self.tabular = self.tabular.drop([
    #                 'pattern ARE', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
    #                 'Brain stem', 'Cerebellum', 'Brain stem',
    #                 'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
    #     'GKS','index'], axis=1)

    #     elif targets == 'pre CO  H':
    #         self.tabular = self.tabular.drop(['(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
    #             'Brain stem', 'Cerebellum', 'Brain stem',
    #             'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
    #    'GKS'], axis=1)
    #     else:
    #         if mode=='test':
    #             self.tabular = self.tabular.drop([
    #                 '(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
    #                 'Basal ganglia', 'Brain stem', 'Cerebellum', 'Brain stem','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
    #     'GKS'], axis=1)
    #         else:
    #             self.tabular = self.tabular.drop([
    #                 '(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
    #                 'Brain stem', 'Cerebellum', 'Brain stem',
    #                 'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
    #     'GKS','index'], axis=1)
                
    #     self.tabular = self.tabular.dropna(axis=0, how='any')
        print(self.tabular.columns)
        print(f'Targets:{self.targets}')
        self.patientIDs = df['Hx no. (VGH)'].str.strip()
        # print(len(self.patientIDs))
        # self.patientIDs=self.patientIDs[self.patientIDs['Hx no. (VGH)'] != '2754059-6']
        self.T2_image = []

        for ID in enumerate(self.patientIDs):
            self.T2_image.append(os.path.join(root, str(ID[1]), 'T2_ROI.nii'))

    def __getitem__(self, index):

        T2 = sitk.ReadImage(self.T2_image[index])
        T2_img = sitk.GetArrayFromImage(T2)
        pateint_tabular = self.tabular.loc[self.tabular['Hx no. (VGH)'] ==
                                           self.patientIDs[index]].drop(
                                               ['Hx no. (VGH)'], axis=1)

        target = pateint_tabular[self.targets]

        pateint_tabular = pateint_tabular.drop([self.targets], axis=1)
        # print(pateint_tabular.columns)
        pateint_tabular = torch.Tensor(np.array(pateint_tabular))
        return {
            'image': self.normalize(T2_img),
            'tabular': pateint_tabular,
            'ID': self.patientIDs[index],
            'target': int(target.values)
        }

    def __len__(self):
        return len(self.T2_image)

    def normalize(self, img):

        s, w, h = img.shape
        assert s <= 64, f"slice > 64,{s}"
        ZeroPadding = np.zeros((64, w, h))
        ZeroPadding[:s, :, :] = img
        ZeroPadding = (ZeroPadding - np.min(ZeroPadding)) / (
            np.max(ZeroPadding) - np.min(ZeroPadding))
        ZeroPadding = np.resize(ZeroPadding, (64, 128, 128))
        ZeroPadding = torch.from_numpy(ZeroPadding)
        return ZeroPadding


def pre_tabular(df,targets,mode):

    df=df.dropna(axis=0, how='any')
    
    scaler = MinMaxScaler()
    if targets == '(ARE)':
        if mode=='test':
                df= df.drop([
                    'pattern ARE', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                    'Basal ganglia', 'Brain stem', 'Cerebellum', 'Brain stem','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
        'GKS'], axis=1)
        else:
                df = df.drop([
                    'pattern ARE', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                    'Brain stem', 'Cerebellum', 'Brain stem',
                    'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
        'GKS','index'], axis=1)

    elif targets == 'pre CO  H':
            df = df.drop(['(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                'Brain stem', 'Cerebellum', 'Brain stem',
                'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
       'GKS'], axis=1)
    else:
            if mode=='test':
                df = df.drop([
                    '(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                    'Basal ganglia', 'Brain stem', 'Cerebellum', 'Brain stem','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
        'GKS'], axis=1)
            else:
                df = df.drop([
                    '(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                    'Brain stem', 'Cerebellum', 'Brain stem',
                    'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
        'GKS','index'], axis=1)
                
    df['Age at GK'] = scaler.fit_transform(df[['Age at GK']])
    df['CSF體積(ml)'] = scaler.fit_transform(df[['CSF體積(ml)']])
    df['腦組織體積(ml)'] = scaler.fit_transform(df[['腦組織體積(ml)']])
    df['血管體積(ml)'] = scaler.fit_transform(df[['血管體積(ml)']])
    df['TC(Gy)'] = scaler.fit_transform(df[['TC(Gy)']])
    df['TP(Gy)'] = scaler.fit_transform(df[['TP(Gy)']])
    df['RV'] = scaler.fit_transform(df[['RV']])
                
    return df 

# if __name__=='__main__':

# root_path='/media/wei/08EE91D9EE91BF7E/AVM projects_T2_236_subjects_final result/AVM projects_T2_236_subjects_final result/'
# testLoader=Data_Loader(root_path,'train')
# # for i in range (50):
# #     T2_img,ID,PatientTabular=testLoader.__getitem__(i)
# T2_img=testLoader.__getitem__(5)
# # print(T2_img)
# for i in range(50):
#     plt.imshow(T2_img[i])
#     print(i)
#     plt.show()
