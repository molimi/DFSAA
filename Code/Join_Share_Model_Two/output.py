# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2023/4/24
@Description: 使用 LassoNet 框架，融合回归和分类，分为共享层和输出层
@Improvement:  1. 网络模型   [2000, 1024, 512, 128]
               2. 五折交叉验证
"""
import torch
from sklearn.model_selection import train_test_split
from interfaces import LassoNetRegressorClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
import json

from loaddata import *
from utils import *
from plot import *

# device configuration
my_seed = 42069
set_random(my_seed)
device = get_device(0.5, my_seed)

in_start = 111500
in_end = 113500

top_k = 10

task = ['AD', 'MCI', 'HC']          # ['HC', 'AD'], ['AD', 'MCI'], ['HC', 'MCI'], ['AD', 'MCI', 'HC']
for fold in range(5):               # five-fold cross-validation
    # X_MRI, X_SNP, y_reg, y_cla, SNP_names = load_dataset(['HC', 'AD'], './data/744_Data_230418', 'Gene_VBM.mat', in_start, in_end)
    # MRI_names = load_features("./data/744_Data_230418", "VBM_ROI_Name_230418.xlsx")
    X_MRI, X_SNP, y_reg, y_cla, SNP_names = load_dataset(['AD', 'HC'], './data/749_Data_230512', 'Gene_FS.mat', in_start, in_end)
    MRI_names = load_features("./data//749_Data_230512", "FS_ROI_Name_230418.xlsx")
    X_MRI_train, X_MRI_test, X_SNP_train, X_SNP_test, y_reg_train, y_reg_test, y_cla_train, y_cla_test = train_test_split(X_MRI, X_SNP, y_reg, y_cla)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"Result_{current_time}"
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    model = LassoNetRegressorClassifier(hidden_dims_D=(64, 16), hidden_dims_A=(128, 64), verbose=2, patience=(100, 5), torch_seed=my_seed, backtrack=True, device=device)
    path_D, path_A = model.path(X_MRI_train, X_SNP_train, y_reg_train, y_cla_train)

    n_selected_D = []
    mse_D = []
    mae_D = []
    accuracy = []
    roc_score = []
    bca = []
    lambda_D = []
    objective_D = []
    skip_reg_weight_D = []
    skip_cla_weight_D = []

    n_selected_A = []
    lambda_A = []
    objective_A = []
    skip_reg_weight_A = []


    for save_D, save_A in zip(path_D, path_A):
        model.load(save_D.state_dict, save_A.state_dict)
        y_reg_pred, y_cla_pred = model.predict_D(X_MRI_test)
        n_selected_D.append(save_D.selected.sum().cpu().numpy())
        mse_D.append(np.sqrt(mean_squared_error(y_reg_test*30., y_reg_pred*30.)))
        mae_D.append(mean_absolute_error(y_reg_test*30., y_reg_pred*30.))
        accuracy.append(accuracy_score(y_cla_test, y_cla_pred))
        roc_score.append(roc_auc_score(y_cla_test, y_cla_pred))
        cm = confusion_matrix(y_cla_test, y_cla_pred)
        bca.append(np.mean([cm[i][i]/sum(cm[i]) for i in range(len(cm))]))
        objective_D.append(save_D.val_objective)
        lambda_D.append(save_D.lambda_)
        skip_reg_weight_D.append(save_D.state_dict["skip_reg.weight"].detach().cpu().numpy())       
        skip_cla_weight_D.append(save_D.state_dict["skip_cla.weight"].detach().cpu().numpy())
        n_selected_A.append(save_A.selected.sum().cpu().numpy())
        objective_A.append(save_A.val_objective)
        lambda_A.append(save_A.lambda_)
        skip_reg_weight_A.append(save_A.state_dict["skip.weight"].detach().cpu().numpy())

    df1 = pd.DataFrame({'n_selected_D': n_selected_D, 'lambda_D': lambda_D, 'mse_D': mse_D,\
                        'mae_D': mae_D, 'accuracy': accuracy, 'roc_score': roc_score, 'bca': bca})       
    df2 = pd.DataFrame({'n_selected_A': n_selected_A, 'lambda_A': lambda_A})     


    file_name1 = str(fold) + "_params_vbm_231220.xlsx"
    path_name = os.path.join(folder_path, file_name1)
    with pd.ExcelWriter(path_name, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='MRI', index=False)
        df2.to_excel(writer, sheet_name='SNP', index=False)

    plt.switch_backend('agg')
    fig = plt.figure(figsize=(12, 12))

    plt.subplot(411)
    plt.grid(True)
    plt.plot(n_selected_D, mse_D, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("MSE")

    plt.subplot(412)
    plt.grid(True)
    plt.plot(lambda_D, mse_D, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("MSE")

    plt.subplot(413)
    plt.grid(True)
    plt.plot(lambda_D, n_selected_D, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.subplot(414)
    plt.grid(True)
    plt.plot(lambda_A, n_selected_A, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")
    file_name1 = str(fold) + "Regressor" + "_vbm_loss_231220.png"
    loss_path_name = os.path.join(folder_path, file_name1)
    plt.savefig(loss_path_name)
    plt.clf()

    plt.switch_backend('agg')
    fig = plt.figure(figsize=(12, 12))

    plt.subplot(411)
    plt.grid(True)
    plt.plot(n_selected_D, accuracy, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("classification accuracy")

    plt.subplot(412)
    plt.grid(True)
    plt.plot(lambda_D, accuracy, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("classification accuracy")

    plt.subplot(413)
    plt.grid(True)
    plt.plot(lambda_D, n_selected_D, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.subplot(414)
    plt.grid(True)
    plt.plot(lambda_A, n_selected_A, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected SNP_features")

    file_name1 = str(fold) + '_classify' + "_vbm_loss-231220.png" 
    loss_path_name = os.path.join(folder_path, file_name1)
    plt.savefig(loss_path_name)
    plt.clf()

    # ind = objective.index(min(objective))
    ind_D = lambda_D.index(min(lambda_D, key=lambda x : abs(x-160)))        
    w_reg_data = skip_reg_weight_D[ind_D]
    # print(w_reg_data.shape)
    fig_snp(w_reg_data, top_k, MRI_names, "regressor", folder_path)
    # print(mse_D[ind])

    w_cla_data = skip_cla_weight_D[ind_D][0, :].reshape(1, -1)         
    
    # print(w_cla_data.shape)
    fig_snp(w_cla_data, top_k, MRI_names, "classify", folder_path)
    # print(accuracy[ind])

    # ind = objective.index(min(objective))
    ind_A = lambda_A.index(min(lambda_A, key=lambda x : abs(x-2800)))       # AD&HC 2800 AD&MCI 180
    w_reg_data = skip_reg_weight_A[ind_A]
    # print(w_reg_data.shape)
    fig_snp(w_reg_data, top_k, SNP_names, "regressor", folder_path)

    df3 = pd.DataFrame({'snp_1_reg_weights': skip_reg_weight_A[ind_A][0, :].ravel(),  'snp_2_reg_weights': skip_reg_weight_A[ind_A][1, :].ravel(),
                        'snp_3_reg_weights': skip_reg_weight_A[ind_A][2, :].ravel()})
    df4 = pd.DataFrame({'vbm_reg_weights': skip_reg_weight_D[ind_D].ravel(), \
                        'vbm_cla_weights_AD': skip_cla_weight_D[ind_D][0, :].ravel(), 'vbm_cla_weights_HC': skip_cla_weight_D[ind_D][1, :].ravel()})
    df5 = pd.DataFrame({'ind_D': [ind_D], 'mse_D': [mse_D[ind_D]], 'mae_D': [mae_D[ind_D]], 'accuracy': [accuracy[ind_D]], \
                        'roc_score': [roc_score[ind_D]], 'bca': [bca[ind_D]], 'ind_A': [ind_A]})

    file_name2 = str(fold) + "_weights_vbm_231220.xlsx"
    path_name = os.path.join(folder_path, file_name2)
    with pd.ExcelWriter(path_name, engine='openpyxl') as writer:
        df3.to_excel(writer, sheet_name='weights_snp', index=False)
        df4.to_excel(writer, sheet_name='weights_vbm', index=False)
        df5.to_excel(writer, sheet_name='evaluation', index=False)