"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2022/3/9
@Description: 1. 损失函数、特征选择的个数、正则化系数三者的关系
              2. 交叉验证损失函数、正则化系数、特征选择个数的关系
              3. 打印SNP权重图
                3. 获取设备
"""
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from interfaces import BaseLassoNetCV

from utils import confidence_interval, eval_on_path


def plot_path(model, path, X_test, y_test, *, score_function=None):
    """
    Plot the evolution of the model on the path, namely:
    - lambda
    - number of selected variables
    - score


    Parameters
    ==========
    model : LassoNetClassifier or LassoNetRegressor
    path
        output of model.path
    X_test : array-like
    y_test : array-like
    score_function : function or None
        if None, use score_function=model.score
        score_function must take as input X_test, y_test
    """
    # TODO: plot with manually computed score
    score = eval_on_path(model, path, X_test, y_test, score_function=score_function)
    n_selected = [save.selected.sum() for save in path]
    lambda_ = [save.lambda_ for save in path]

    plt.figure(figsize=(16, 16))

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, score, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("score")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, score, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.tight_layout()


def plot_cv(model: BaseLassoNetCV, X_test, y_test, *, score_function=None):
    # TODO: plot with manually computed score
    lambda_ = [save.lambda_ for save in model.path_]
    lambdas = [[h.lambda_ for h in p] for p in model.raw_paths_]

    score = eval_on_path(
        model, model.path_, X_test, y_test, score_function=score_function
    )

    plt.figure(figsize=(16, 16))

    plt.subplot(211)
    plt.grid(True)
    first = True
    for sl, ss in zip(lambdas, model.raw_scores_):
        plt.plot(
            sl,
            ss,
            "r.-",
            markersize=5,
            alpha=0.2,
            label="cross-validation" if first else None,
        )
        first = False
    avg = model.interp_scores_.mean(axis=1)
    ci = confidence_interval(model.interp_scores_)
    plt.plot(
        model.lambdas_,
        avg,
        "g.-",
        markersize=5,
        alpha=0.2,
        label="average cv with 95% CI",
    )
    plt.fill_between(model.lambdas_, avg - ci, avg + ci, color="g", alpha=0.1)
    plt.plot(lambda_, score, "b.-", markersize=5, alpha=0.2, label="test")
    plt.legend()
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")

    plt.subplot(212)
    plt.grid(True)
    first = True
    for sl, path in zip(lambdas, model.raw_paths_):
        plt.plot(
            sl,
            [save.selected.sum() for save in path],
            "r.-",
            markersize=5,
            alpha=0.2,
            label="cross-validation" if first else None,
        )
        first = False
    plt.plot(
        lambda_,
        [save.selected.sum() for save in model.path_],
        "b.-",
        markersize=5,
        alpha=0.2,
        label="test",
    )
    plt.legend()
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.tight_layout()


def fig_snp(weights, top_k, features, set_index):
    w_data = np.mean(weights.T, axis=1)                            
    # w_data = weights.T
    data_count= np.arange(len(w_data))
    w_data_abs = np.abs(w_data)
    show_index = w_data_abs.argsort()[::-1][0:top_k]
    # w_data = list(map(str, w_data))
    file_name3 = str(set_index) + "_vbm_weights_230728.txt"
    weights_path_name = os.path.join("./VBM", file_name3)
    with open(weights_path_name, 'a+', encoding='utf-8') as f:
        f.write("\n********************************\n")
        [f.write(str(value)+'\n') for value in w_data]
    print("\n********************************")
    print('Top %d risk SNPs in Task:' % top_k)
    for i in range(show_index.shape[-1]):
        print("gene %d: %s" % (show_index[i], features[show_index[i]][0]))
    print("********************************\n")
    file_name4 = str(set_index) + "_vbm_snp_230728.txt"
    snp_path_name = os.path.join("./VBM", file_name4)
    with open(snp_path_name, 'a+', encoding='utf-8') as f:
        f.write("\n********************************\n")
        f.write('Top %d risk SNPs in Task:\n' % top_k)
        for i in range(show_index.shape[-1]):
            f.write("gene %d: %s\n" % (show_index[i], features[show_index[i]][0]))
        f.write("********************************\n")

    plt.switch_backend('agg')
    fig = plt.figure()
    ax1 = fig.subplots()
    ax1.set_title('Weight of SNPs')
    plt.xlabel('Feature_index')
    plt.ylabel('W')
    ax1.scatter(data_count, w_data, c='r', s=30, linewidths=1, marker='o', edgecolors='k')
    for i in range(show_index.shape[-1]):
        plt.annotate((show_index[i]), xytext=(show_index[i], w_data[show_index[i]]),
                     xy=(show_index[i], w_data[show_index[i]]))                        
    file_name5 = str(set_index) + "_vbm_weights_230728.png"
    weight_path_name = os.path.join("./VBM", file_name5)
    plt.savefig(weight_path_name)
    plt.show()