# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2023/4/24
@Description: 使用 LassoNet 框架，融合回归和分类，分为共享层和输出层
@Improvement:  1. 网络模型   [2000, 1024, 512, 128]
"""

import torch
from sklearn.model_selection import train_test_split
from interfaces import LassoNetRegressorClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import datetime
from loaddata_fold import *
from utils import *
from plot import *
import pandas as pd

# device configuration
my_seed = 42069
set_random(my_seed)
device = get_device(0.3, my_seed)
in_start = 111500
in_end = 113500
top_k = 10

task = ['AD', 'MCI', 'HC']          # ['HC', 'AD'], ['AD', 'MCI'], ['HC', 'MCI'], ['AD', 'MCI', 'HC']
for fold in range(5):               # five-fold cross-validation
    # Get the current time and format it as a string（"2023-12-21_15-30-00"）
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a folder name, for example: "Result_2023-12-21_15-30-00"
    folder_name = f"Result_{current_time}"
    # Create folder path
    folder_path = os.path.join(os.getcwd(), folder_name)

    # Create folder
    os.makedirs(folder_path, exist_ok=True)

    X_train, X_test, X_SNP_train, X_SNP_test, y_reg_train, y_reg_test, y_cla_train, y_cla_test, SNP_names = load_dataset(fold+1, ['AD', 'HC'], './data/744_Data_230418', 'Gene_VBM.mat', in_start, in_end)
    feature_names = load_features("./data//744_Data_230418", "VBM_ROI_Name_230418.xlsx")

    model = LassoNetRegressorClassifier(hidden_dims=(64, 16), verbose=2, patience=(100, 5), torch_seed=my_seed, backtrack=True, device=device)
    path = model.path(X_train, y_reg_train, y_cla_train)

    n_selected = []
    mse = []
    mae = []
    accuracy = []
    roc_score = []
    bca = []
    lambda_ = []
    objective = []
    skip_reg_weight = []
    skip_cla_weight = []

    for save in path:
        model.load(save.state_dict)
        y_reg_pred, y_cla_pred = model.predict(X_test)
        n_selected.append(save.selected.sum().cpu().numpy())
        mse.append(np.sqrt(mean_squared_error(y_reg_test*30., y_reg_pred*30.)))
        mae.append(mean_absolute_error(y_reg_test*30., y_reg_pred*30.))
        accuracy.append(accuracy_score(y_cla_test, y_cla_pred))
        roc_score.append(roc_auc_score(y_cla_test, y_cla_pred))
        cm = confusion_matrix(y_cla_test, y_cla_pred)
        bca.append(np.mean([cm[i][i]/sum(cm[i]) for i in range(len(cm))]))
        objective.append(save.val_objective)
        lambda_.append(save.lambda_)
        skip_reg_weight.append(save.state_dict["skip_reg.weight"].detach().cpu().numpy())
        skip_cla_weight.append(save.state_dict["skip_cla.weight"].detach().cpu().numpy())


    df1 = pd.DataFrame({'n_selected': n_selected, 'lambda': lambda_, 'mse': mse,\
                        'mae_D': mae, 'accuracy': accuracy, 'roc_score': roc_score, 'bca': bca})        # regression
    
    # Create Excel file and write data
    file_name1 = str(fold) + "_params_vbm_231228.xlsx"
    path_name = os.path.join(folder_path, file_name1)
    with pd.ExcelWriter(path_name, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='MRI', index=False)
    
    # regression
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(12, 12))

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, mse, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("MSE")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, mse, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("MSE")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")
    file_name1 = str(fold) + "Regressor" + "_vbm_loss_231228.png"
    loss_path_name = os.path.join(folder_path, file_name1)
    plt.savefig(loss_path_name)
    plt.clf()

    # classification
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(12, 12))

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, accuracy, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("classification accuracy")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, accuracy, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("classification accuracy")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    file_name1 = str(fold) + '_classify' + "_vbm_loss-231228.png" 
    loss_path_name = os.path.join(folder_path, file_name1)
    plt.savefig(loss_path_name)
    plt.clf()

    ind = lambda_.index(min(lambda_, key=lambda x : abs(x-66)))         # FS 100 AD 66
    w_reg_data = skip_reg_weight[ind]
    # print(w_reg_data.shape)
    fig_snp(w_reg_data, top_k, feature_names, "regressor", folder_path)
    # print(mse[ind])

    w_cla_data = skip_cla_weight[ind][0, :].reshape(1, -1)          # AD and HC are discussed separately.
    # print(w_cla_data.shape)
    fig_snp(w_cla_data, top_k, feature_names, "classify", folder_path)
    print(accuracy[ind])

    df3 = pd.DataFrame({'vbm_reg_weights': skip_reg_weight[ind].ravel(), \
                        'vbm_cla_weights_AD': skip_cla_weight[ind][0, :].ravel(), 'vbm_cla_weights_HC': skip_cla_weight[ind][1, :].ravel()})
    df4 = pd.DataFrame({'ind_D': [ind], 'mse_D': [mse[ind]], 'mae_D': [mae[ind]], 'accuracy': [accuracy[ind]], \
                        'roc_score': [roc_score[ind]], 'bca': [bca[ind]]})

    file_name2 = str(fold) + "_weights_vbm_231228.xlsx"
    path_name = os.path.join(folder_path, file_name2)
    with pd.ExcelWriter(path_name, engine='openpyxl') as writer:
        df3.to_excel(writer, sheet_name='weights_vbm', index=False)
        df4.to_excel(writer, sheet_name='evaluation', index=False)