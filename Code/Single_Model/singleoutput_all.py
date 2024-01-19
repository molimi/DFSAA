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
# from lassonet import LassoNetRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from loaddata import *
from utils import *
from plot import *

# device configuration
my_seed = 42069
set_random(my_seed)

device = get_device(0.7, my_seed)

in_start = 111000
in_end = 113000
data_num = 2   # 0--AV45  1--FDG 2--VBM
for i in range(1, 116, 2):
    set_index = i
    # set0_index = [36, 37, 40, 41, 42, 45, 47, 48, 68, 69]    # AV45
    # set1_index = [0, 2, 3, 27, 30, 31, 56, 57, 62, 77]       # FDG 

    top_k = 10
    X, y, feature_names = load_data('.././Data', 'Data_common_Img_adjusted.mat', in_start, in_end, data_num, set_index)


    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # [1024, 512, 128], [1000, 512, 128], [1024, 128, 64], [1000, 128, 64], [1200, 512, 128], [900, 512, 128], [800, 512, 128], [512, 128]
    # [1024, 512, 128, 64], [1000, 512, 256, 128, 64],
    model = LassoNetRegressor(hidden_dims=(1000, 512, 128), verbose=2, patience=(100, 5), torch_seed=my_seed, backtrack=True, device=device)
    path = model.path(X_train, y_train)

    n_selected = []
    mse = []
    lambda_ = []
    skip_weight = []

    for save in path:
        model.load(save.state_dict)
        y_pred = model.predict(X_test)
        n_selected.append(save.selected.sum().cpu().numpy())
        mse.append(mean_squared_error(y_test, y_pred))
        lambda_.append(save.lambda_)
        skip_weight.append(save.state_dict["skip.weight"].detach().cpu().numpy().flatten())

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
    file_name1 = str(set_index) + "_vbm_loss-230330.png"
    loss_path_name = os.path.join("./VBM/Loss", file_name1)
    plt.savefig(loss_path_name)
    plt.clf()


    w_data = skip_weight[mse.index(min(mse))]
    fig_snp(w_data, top_k, feature_names, set_index)



    n_features = X.shape[1]
    importances = model.feature_importances_.numpy()
    order = np.argsort(importances)[::-1][:top_k]
    importances = importances[order]
    ordered_feature_names = [feature_names[i][0] for i in order]
    color = np.array(['g'] * top_k)


    plt.switch_backend('agg')
    plt.subplot(211)
    plt.bar(
        np.arange(top_k),
        importances,
        color=color,
    )
    plt.xticks(np.arange(top_k), ordered_feature_names, rotation=90)
    colors = {"real features": "g"}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.ylabel("Feature importance")


    _, order = np.unique(importances, return_inverse=True)
    plt.subplot(212)
    plt.bar(
        np.arange(top_k),
        order + 1,
        color=color,
    )
    plt.xticks(np.arange(top_k), ordered_feature_names, rotation=90)
    plt.legend(handles, labels)
    plt.ylabel("Feature order")
    file_name2 = str(set_index) + "_vbm_feature_order-230330.png"
    feature_path_name = os.path.join("./VBM/Feature", file_name2)
    plt.savefig(feature_path_name)