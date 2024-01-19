# -*- coding: utf-8 -*-
"""
@Version: 0.3
@Author: CarpeDiem
@Date: 2023/3/31
@Description: load SNP_data、brain_data、SNP_ID
# 导入影像数据、分类和评分
"""
from torch.utils.data import Dataset
import os
import scipy.io as scio
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data samples .npy fiels
def load_dataset(task, folder, data_name, in_start, in_end):
    my_task = []
    for t in task:
        if t == 'AD':
            my_task.append(0)
        if t == 'MCI':
            my_task.append(1)
        if t == 'HC':
            my_task.append(2)
    my_task = np.array(my_task)

    # path = "./data/744_Data_230418/"          # VBM
    path = "./data/749_Data_230512/"            # FS
    Y_dis = np.argmax(np.load(path+"HC_MCI_AD_230511.npy"), axis=-1)    # Disease labels, AD: 0, MCI: 1, HC: 2

    # Extract appropriate indicies for the given task
    task_idx = np.zeros(shape=Y_dis.shape)

    for t in range(len(my_task)):
        task_idx += np.array(Y_dis == my_task[t])
    task_idx = task_idx.astype(bool)

    Y_dis = Y_dis[task_idx]

    # X_MRI = np.load(path + "744_VBM_230418.npy")[task_idx, :]
    # S_cog = np.load(path + "744_MMSE_230418.npy")[task_idx]
    X_MRI = np.load(path + "749_FS_230511.npy")[task_idx, :]
    S_cog = np.load(path + "MMSE.npy")[task_idx]


    # One-hot encoding for the disease label
    for i in range(np.unique(Y_dis).shape[0]):
        Y_dis[Y_dis == np.unique(Y_dis)[i]] = i

    # Normalization
    S_cog = S_cog.astype(np.float32)
    S_cog /= 30.0

    data_file = os.path.join(folder, data_name)
    data = scio.loadmat(data_file)
    X_SNP = data['Gene_Data'][task_idx, in_start:in_end]
    scaler_SNP = StandardScaler()
    X_SNP_tr = scaler_SNP.fit_transform(X_SNP)
    name_X_SNP = data['SNP_ID'][in_start:in_end]    

    # MRI normalization
    scaler_MRI = MinMaxScaler()
    X_MRI_tr = scaler_MRI.fit_transform(X_MRI)

    return X_MRI_tr, X_SNP_tr, S_cog, Y_dis, name_X_SNP


def load_features(folder, dataname):
    datafile = os.path.join(folder, dataname)
    data = pd.read_excel(datafile)
    data = np.array(data)
    features = data.flatten(order='C')
    return features