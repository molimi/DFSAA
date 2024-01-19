# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2023/3/9
@Description: 使用 LassoNet 框架，做多输出
@Improvement:  1. 网络模型   [2000, 1024, 512, 128]
"""
import torch
from sklearn.model_selection import train_test_split
from interfaces import LassoNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
from loaddata_fold import *
from utils import *
from plot import *

# device configuration
my_seed = 42069
set_random(my_seed)

device = get_device(0.5, my_seed)
in_start = 111500                                               # Select the location of the gene
in_end = 113500
set_index = [38, 0, 39, 54, 79, 5, 51, 23, 3, 45]             # VBM, brain regions selected using regression and classification experiments
# set_index = [20, 35, 26, 34, 22, 39, 21, 54, 71, 23]        # FS，brain regions selected using regression and classification experiments

top_k = 10

task = ['AD', 'MCI', 'HC']          # ['HC', 'AD'], ['AD', 'MCI'], ['HC', 'MCI'], ['HC', 'MCI', 'AD']
for fold in range(5):               # five-fold cross-validation
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"Result_{current_time}"
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    y_train, y_test, X_train, X_test, y_reg_train, y_reg_test, y_cla_train, y_cla_test, SNP_names = load_dataset(fold+1, ['MCI', 'HC'], './data/744_Data_230418', 'Gene_VBM.mat', in_start, in_end, set_index)
    model = LassoNetRegressor(hidden_dims=(128, 64), verbose=2, patience=(100, 10), torch_seed=my_seed, device=device)
    path = model.path(X_train, y_train)

    n_selected = []
    mse = []
    mae = []
    lambda_ = []
    skip_weight = []

    for save in path:
        model.load(save.state_dict)
        y_pred = model.predict(X_test)
        n_selected.append(save.selected.sum().cpu().numpy())
        mse.append(mean_squared_error(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
        lambda_.append(save.lambda_)
        skip_weight.append(save.state_dict["skip.weight"].detach().cpu().numpy())

    df1 = pd.DataFrame({'n_selected': n_selected, 'lambda': lambda_})     
    file_name1 = str(fold) + "_params_vbm_231228.xlsx"
    path_name = os.path.join(folder_path, file_name1)
    with pd.ExcelWriter(path_name, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='SNP', index=False)


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

    ind = lambda_.index(min(lambda_, key=lambda x : abs(x-10)))     # FS  10  VBM 10
    w_data = skip_weight[ind]
    # w_data = skip_weight[mse.index(min(mse))]
    # print(w_data.shape)
    fig_snp(w_data, top_k, SNP_names, "select_roi", folder_path)

    df3 = pd.DataFrame({'snp_1_reg_weights': skip_weight[ind][0, :].ravel(),  'snp_2_reg_weights': skip_weight[ind][1, :].ravel(),
                        'snp_3_reg_weights': skip_weight[ind][2, :].ravel(), 'snp_4_reg_weights': skip_weight[ind][3, :].ravel(),
                        'snp_3_reg_weights': skip_weight[ind][4, :].ravel(), 'snp_4_reg_weights': skip_weight[ind][5, :].ravel(),
                        'snp_3_reg_weights': skip_weight[ind][6, :].ravel(), 'snp_4_reg_weights': skip_weight[ind][7, :].ravel(),
                        'snp_3_reg_weights': skip_weight[ind][8, :].ravel(), 'snp_4_reg_weights': skip_weight[ind][9, :].ravel()})

    file_name2 = str(fold) + "_weights_vbm_231220.xlsx"
    path_name = os.path.join(folder_path, file_name2)
    with pd.ExcelWriter(path_name, engine='openpyxl') as writer:
        df3.to_excel(writer, sheet_name='weights_snp', index=False)