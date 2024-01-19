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

# Load data samples .npy文件
def load_dataset(fold, task, folder, data_name, in_start, in_end):
    my_task = []
    for t in task:
        if t == 'AD':
            my_task.append(0)
        if t == 'MCI':
            my_task.append(1)
        if t == 'HC':
            my_task.append(2)
    my_task = np.array(my_task)

    path = "./data/744_Data_230418/"
    Y_dis = np.argmax(np.load(path+"HC_MCI_AD_230418.npy"), axis=-1)    # Disease labels, AD: 0, MCI: 1, HC: 2

    # Extract appropriate indicies for the given task
    task_idx = np.zeros(shape=Y_dis.shape)

    for t in range(len(my_task)):
        task_idx += np.array(Y_dis == my_task[t])
    task_idx = task_idx.astype(bool)

    Y_dis = Y_dis[task_idx]

    X_MRI = np.load(path + "744_VBM_230418.npy")[task_idx, :]               # VBM Modality
    S_cog = np.load(path + "744_MMSE_230418.npy")[task_idx]
    # X_MRI = np.load(path + "749_FS_230511.npy")[task_idx, :]              # FS Modality
    # S_cog = np.load(path + "MMSE.npy")[task_idx]
    
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

    # Data randomizing
    rand_idx = np.random.RandomState(seed=951014).permutation(Y_dis.shape[0])
    X_MRI = X_MRI[rand_idx, ...]
    X_SNP = X_SNP[rand_idx, ...]
    Y_dis = Y_dis[rand_idx, ...]
    S_cog = S_cog[rand_idx, ...]

    # Fold dividing
    rand_idx = np.random.RandomState(seed=5930).permutation(Y_dis.shape[0])
    num_samples = int(Y_dis.shape[0]/5)
    ts_idx = rand_idx[num_samples * (fold - 1):num_samples * fold]
    tr_idx = np.setdiff1d(rand_idx, ts_idx)

    X_MRI_tr, X_MRI_ts = X_MRI[tr_idx, :], X_MRI[ts_idx, :]
    X_SNP_tr, X_SNP_ts = X_SNP[tr_idx, :], X_SNP[ts_idx, :]
    Y_dis_tr, Y_dis_ts = Y_dis[tr_idx], Y_dis[ts_idx]
    S_cog_tr, S_cog_ts = S_cog[tr_idx], S_cog[ts_idx]

    # MRI normalization
    scaler = MinMaxScaler()
    X_MRI_tr = scaler.fit_transform(X_MRI_tr)
    X_MRI_ts = scaler.transform(X_MRI_ts)

    return X_MRI_tr, X_MRI_ts, X_SNP_tr, X_SNP_ts, S_cog_tr, S_cog_ts, Y_dis_tr, Y_dis_ts, name_X_SNP


def load_features(folder, dataname):                        # load features names
    datafile = os.path.join(folder, dataname)
    data = pd.read_excel(datafile)
    data = np.array(data)
    features = data.flatten(order='C')
    return features