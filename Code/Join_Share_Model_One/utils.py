"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2022/3/9
@Description: 1. 获取设备
              2、导入训练的参数
"""

from typing import Iterable
import torch
import scipy.stats
import torch
import numpy as np
import random
import os

def set_random(my_seed):
    """set a random seed to ensure consistent results when the parameters are the same"""
    print("set random seed: %d" % my_seed)
    random.seed(my_seed)
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(memory_rate, my_seed):
    """cpu/gpu"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(my_seed)
        torch.cuda.set_device(3)             # Choose to use the first GPU
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.empty(int(total_memory * memory_rate), dtype=torch.int8, device='cuda')
        return 'cuda'
    else:
        return 'cpu'
    
def eval_on_path(model, path, X_test, y_test, *, score_function=None):
    if score_function is None:
        score_fun = model.score
    else:
        assert callable(score_function)

        def score_fun(X_test, y_test):
            return score_function(y_test, model.predict(X_test))

    score = []
    for save in path:
        model.load(save.state_dict)
        score.append(score_fun(X_test, y_test))
    return score


if hasattr(torch.Tensor, "scatter_reduce_"):
    # version >= 1.12
    def scatter_reduce(input, dim, index, reduce, *, output_size=None):
        src = input
        if output_size is None:
            output_size = index.max() + 1
        return torch.empty(output_size).scatter_reduce(
            dim=dim, index=index, src=src, reduce=reduce, include_self=False
        )


else:
    scatter_reduce = torch.scatter_reduce


def scatter_logsumexp(input, index, *, dim=-1, output_size=None):
    """
    Inspired by torch_scatter.logsumexp
    Uses torch.scatter_reduce for performance
    """
    max_value_per_index = scatter_reduce(
        input, dim=dim, index=index, output_size=output_size, reduce="amax"
    )
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = input - max_per_src_element
    sum_per_index = scatter_reduce(
        recentered_scores.exp(),
        dim=dim,
        index=index,
        output_size=output_size,
        reduce="sum",
    )
    return max_value_per_index + sum_per_index.log()


def log_substract(x, y):
    """log(exp(x) - exp(y))"""
    return x + torch.log1p(-(y - x).exp())


def confidence_interval(data, confidence=0.95):
    if isinstance(data[0], Iterable):
        return [confidence_interval(d, confidence) for d in data]
    return scipy.stats.t.interval(
        confidence,
        len(data) - 1,
        scale=scipy.stats.sem(data),
    )[1]