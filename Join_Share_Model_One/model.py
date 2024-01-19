# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2023/3/9
@Description: LassoNet 框架搭建
@Improvement: 自由设置网络模型，默认使用前馈神经网络
"""
from itertools import islice

import torch
from torch import nn
from torch.nn import functional as F

from prox import inplace_prox, prox


class LassoNet(nn.Module):
    def __init__(self, *hidden_dims, dropout=None):
        """
        first dimension is input
        last dimension is output
        """
        assert len(hidden_dims) > 2
        super().__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.layers_share = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 3)])
        
        self.layers_reg = nn.Linear(hidden_dims[-3], hidden_dims[-2])           # 回归部分
        self.layers_cla = nn.Linear(hidden_dims[-3], hidden_dims[-1])           # 分类部分
        self.skip_reg = nn.Linear(hidden_dims[0], hidden_dims[-2], bias=False)
        self.skip_cla = nn.Linear(hidden_dims[0], hidden_dims[-1], bias=False)


    def forward(self, inp):
        current_layer = inp
        result_reg = self.skip_reg(inp)
        result_cla = self.skip_cla(inp)

        for theta in self.layers_share:
            current_layer = theta(current_layer)
            if self.dropout is not None:
                current_layer = self.dropout(current_layer)
            current_layer = F.relu(current_layer)

        return result_reg + self.layers_reg(current_layer), result_cla + self.layers_cla(current_layer)

    def prox(self, *, lambda_, lambda_bar=0, M=1):                  # 在这里把分类和回归融合在一起做
        with torch.no_grad():
            inplace_prox(
                alpha=self.skip_reg,
                beta=self.skip_cla,
                theta=self.layers_share[0],
                lambda_=lambda_,
                lambda_bar=lambda_bar,
                M=M,
            )

    def lambda_start(
        self,
        M=1,
        lambda_bar=0,
        factor=2,
    ):
        """Estimate when the model will start to sparsify."""

        def is_sparse(lambda_):
            with torch.no_grad():
                alpha_beta = torch.cat([self.skip_reg.weight.data, self.skip_cla.weight.data])
                theta = self.layers_share[0].weight.data

                for _ in range(10000):
                    new_alpha_beta, theta = prox(
                        alpha_beta,
                        theta,
                        lambda_=lambda_,
                        lambda_bar=lambda_bar,
                        M=M,
                    )
                    if torch.abs(alpha_beta - new_alpha_beta).max() < 1e-5:
                        # print(_)
                        break
                    alpha_beta = new_alpha_beta
                return (torch.norm(alpha_beta, p=2, dim=0) == 0).sum()

        start = 1e-6
        while not is_sparse(factor * start):
            start *= factor
        return start

    def l2_regularization(self):
        """
        L2 regulatization of the MLP without the first layer
        which is bounded by the skip connection
        """
        ans = 0
        for layer in islice(self.layers_share, 1, None):
            ans += (
                torch.norm(
                    layer.weight.data,
                    p=2,
                )
                ** 2
            )
        ans += torch.norm(self.layers_reg.weight.data, p=2,)**2 + torch.norm(self.layers_cla.weight.data, p=2,)**2
        return ans

    def l1_regularization_skip(self):                       # regularization term
        alpha_beta = torch.cat([self.skip_reg.weight.data, self.skip_cla.weight.data]) 
        return torch.norm(alpha_beta, p=2, dim=0).sum()

    def l2_regularization_skip(self):
        alpha_beta = torch.cat([self.skip_reg.weight.data, self.skip_cla.weight.data]) 
        return torch.norm(alpha_beta, p=2)

    def input_mask(self):
        with torch.no_grad():
            alpha_beta = torch.cat([self.skip_reg.weight.data, self.skip_cla.weight.data]) 
            return torch.norm(alpha_beta, p=2, dim=0) != 0

    def selected_count(self):
        return self.input_mask().sum().item()

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}