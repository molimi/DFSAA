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
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

from loaddata import *
from utils import *
from plot import *

# device configuration
my_seed = 42069
set_random(my_seed)
device = get_device(0.5, my_seed)

# without five-fold cross-validation
top_k = 10
X, y_reg, y_cla = load_dataset(['AD', 'MCI'])
feature_names = load_features("./data/749_Data_230512", "FS_ROI_Name_230418.xlsx")

X_train, X_test, y_reg_train, y_reg_test, y_cla_train, y_cla_test = train_test_split(X, y_reg, y_cla)

model = LassoNetRegressorClassifier(hidden_dims=(128, 64, 16), verbose=2, patience=(100, 5), torch_seed=my_seed, backtrack=True, device=device)
path = model.path(X_train, y_reg_train, y_cla_train)

n_selected = []
mse = []
accuracy = []
lambda_ = []
objective = []
skip_reg_weight = []
skip_cla_weight = []

for save in path:
    model.load(save.state_dict)
    y_reg_pred, y_cla_pred = model.predict(X_test)
    n_selected.append(save.selected.sum().cpu().numpy())
    mse.append(mean_squared_error(y_reg_test, y_reg_pred))
    accuracy.append(accuracy_score(y_cla_test, y_cla_pred))
    objective.append(save.val_objective)
    lambda_.append(save.lambda_)
    skip_reg_weight.append(save.state_dict["skip_reg.weight"].detach().cpu().numpy())
    skip_cla_weight.append(save.state_dict["skip_cla.weight"].detach().cpu().numpy())


# 回归
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
file_name1 = str("Regressor") + "_fs_loss_241228.png"
loss_path_name = os.path.join("./MCI&AD/Regressor", file_name1)
plt.savefig(loss_path_name)
plt.clf()

# 分类
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

file_name1 = str('classify') + "_loss_241228.png"
loss_path_name = os.path.join("./MCI&AD/Classify", file_name1)
plt.savefig(loss_path_name)
plt.clf()

# ind = objective.index(min(objective))
ind = lambda_.index(min(lambda_, key=lambda x : abs(x-100)))
save_path = "./MCI&AD/Regressor"
w_reg_data = skip_reg_weight[ind]
# print(w_reg_data.shape)
fig_snp(w_reg_data, top_k, feature_names, "regressor", save_path)
print(mse[ind])

save_path = "./MCI&AD/Classify"
w_cla_data = skip_cla_weight[ind][0, :].reshape(1, -1)          # 这里不加权取平均，AD和HC分别讨论
# print(w_cla_data.shape)
fig_snp(w_cla_data, top_k, feature_names, "classify", save_path)
print(accuracy[ind])

n_features = X.shape[1]
importances = model.feature_importances_.numpy()
order = np.argsort(importances)[::-1][:top_k]
importances = importances[order]
ordered_feature_names = [feature_names[i] for i in order]
color = np.array(['g'] * top_k)

save_path = "./MCI&AD/Share_Result"
file_name3 = str("regressor_classify") + "_fs_feature_order_241228.txt"
snp_path_name = os.path.join(save_path, file_name3)
with open(snp_path_name, 'a+', encoding='utf-8') as f:
    f.write("\n********************************\n")
    f.write('Top %d risk ROIs in Task:\n' % top_k)
    for i in range(len(order)):
        f.write("ROI %d: %s\n" % (order[i], ordered_feature_names[i]))
    f.write("********************************\n")



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
file_name2 = str("regressor_classify") + "_fs_feature_order_241228.png"
feature_path_name = os.path.join("./MCI&AD/Share_Result", file_name2)
plt.savefig(feature_path_name)