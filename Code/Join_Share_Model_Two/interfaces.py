# -*- coding: utf-8 -*-
"""
@Version: 2.0
@Author: CarpeDiem
@Date: 2023/12/12
@Description: LassoNet 实例化模型
              LassoNetClassifier,
              LassoNetRegressor,
              LassoNetRegressorClassifier,
              LassoNetClassifierCV,
              LassoNetRegressorCV,
@Improvement: 实现影像的AD分类和认知预测，同时实现基因与影像的回归
"""

from itertools import islice
from abc import ABCMeta, abstractmethod, abstractstaticmethod
from dataclasses import dataclass
from functools import partial
import itertools
from typing import List
import warnings
from torch.nn import functional as F

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.model_selection import check_cv, train_test_split
import torch
from tqdm import tqdm

from model import LassoNet, RegressorClassifier

def abstractattr(f):
    return property(abstractmethod(f))


@dataclass
class HistoryItem:
    lambda_: float
    state_dict: dict
    objective: float            # loss + lambda_ * regularization
    loss: float
    val_objective: float        # val_loss + lambda_ * regularization
    val_loss: float
    regularization: float
    l2_regularization: float
    l2_regularization_skip: float
    selected: torch.BoolTensor
    n_iters: int

    def log(item):
        print(
            f"{item.n_iters} epochs, "
            f"val_objective "
            f"{item.val_objective:.2e}, "
            f"val_loss "
            f"{item.val_loss:.2e}, "
            f"regularization {item.regularization:.2e}, "
            f"l2_regularization {item.l2_regularization:.2e}"
        )


class BaseLassoNet(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        hidden_dims_D=(100,),
        hidden_dims_A=(100,),
        lambda_start_D="auto",
        lambda_seq_D=None,
        lambda_start_A="auto",
        lambda_seq_A=None,
        gamma_D=0.0,
        gamma_skip_D=0.0,
        gamma_A=0.0,
        gamma_skip_A=0.0,
        path_multiplier_D=1.01,
        path_multiplier_A=1.02,
        M_D=10,
        M_A=10,
        dropout=0,
        batch_size=None,
        optim_D=None,
        optim_A=None,
        n_iters=(1000, 100),
        patience=(100, 10),
        tol=0.99,
        backtrack=False,
        val_size=None,
        device=None,
        verbose=1,
        random_state=None,
        torch_seed=None,
        class_weight=None,
    ):
        """
        Parameters
        ----------
        hidden_dims : tuple of int, default=(100,)
            Shape of the hidden layers.
        lambda_start : float, default='auto'
            First value on the path. Leave 'auto' to estimate it automatically.
        lambda_seq : iterable of float
            If specified, the model will be trained on this sequence
            of values, until all coefficients are zero.
            The dense model will always be trained first.
            Note: lambda_start and path_multiplier will be ignored.
        gamma : float, default=0.0
            l2 penalization on the network
        gamma : float, default=0.0
            l2 penalization on the skip connection
        path_multiplier : float, default=1.02
            Multiplicative factor (:math:`1 + \\epsilon`) to increase
            the penalty parameter over the path
        M : float, default=10.0
            Hierarchy parameter.
        dropout : float, default = None
        batch_size : int, default=None
            If None, does not use batches. Batches are shuffled at each epoch.
        optim : torch optimizer or tuple of 2 optimizers, default=None
            Optimizer for initial training and path computation.
            Default is Adam(lr=1e-3), SGD(lr=1e-3, momentum=0.9).
        n_iters : int or pair of int, default=(1000, 100)
            Maximum number of training epochs for initial training and path computation.
            This is an upper-bound on the effective number of epochs, since the model
            uses early stopping.
        patience : int or pair of int or None, default=10
            Number of epochs to wait without improvement during early stopping.
        tol : float, default=0.99
            Minimum improvement for early stopping: new objective < tol * old objective.
        backtrack : bool, default=False
            If true, ensures the objective function decreases.
        val_size : float, default=None
            Proportion of data to use for early stopping.
            0 means that training data is used.
            To disable early stopping, set patience=None.
            Default is 0.1 for all models except Cox for which training data is used.
            If X_val and y_val are given during training, it will be ignored.
        device : torch device, default=None
            Device on which to train the model using PyTorch.
            Default: GPU if available else CPU
        verbose : int, default=1
        random_state
            Random state for validation
        torch_seed
            Torch state for model random initialization
        class_weight : iterable of float, default=None
            If specified, weights for different classes in training.
            There must be one number per class.
        """
        assert isinstance(hidden_dims_D, tuple), "`hidden_dims` must be a tuple"
        self.hidden_dims_D = hidden_dims_D
        self.hidden_dims_A = hidden_dims_A
        self.lambda_start_D = lambda_start_D
        self.lambda_start_A = lambda_start_A
        self.lambda_seq_D = lambda_seq_D
        self.lambda_seq_A = lambda_seq_A
        self.gamma_D = gamma_D
        self.gamma_A = gamma_A
        self.gamma_skip_D = gamma_skip_D
        self.gamma_skip_A = gamma_skip_A
        self.path_multiplier_D = path_multiplier_D
        self.path_multiplier_A = path_multiplier_A
        self.M_D = M_D
        self.M_A = M_A
        self.dropout = dropout
        self.batch_size = batch_size
        self.optim_D = optim_D
        self.optim_A = optim_A
        if optim_D is None:
            # Learning rate affects whether the loss function converges
            optim_D = (
                partial(torch.optim.Adam, lr=1e-3),
                partial(torch.optim.SGD, lr=1e-3, momentum=0.9),
            )
        if optim_A is None:
            # The learning rate affects whether the loss function converges, first momentum and then gradient descent
            optim_A = (
                partial(torch.optim.Adam, lr=1e-3),
                partial(torch.optim.SGD, lr=1e-3, momentum=0.9),
            )
        
        if isinstance(optim_D, partial):
            optim_D = (optim_D, optim_D)
        if isinstance(optim_A, partial):
            optim_A = (optim_A, optim_A)
        self.optim_init_D, self.optim_path_D = optim_D
        self.optim_init_A, self.optim_path_A = optim_A
        if isinstance(n_iters, int):
            n_iters = (n_iters, n_iters)
        self.n_iters = self.n_iters_init, self.n_iters_path = n_iters
        if patience is None or isinstance(patience, int):
            patience = (patience, patience)
        self.patience = self.patience_init, self.patience_path = patience
        self.tol = tol
        self.backtrack = backtrack
        if val_size is None:
            # TODO: use a cv parameter following sklearn's interface
            val_size = 0.1
        self.val_size = val_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.verbose = verbose

        self.random_state = random_state
        self.torch_seed = torch_seed

        self.model_D = None
        self.model_A = None
        self.class_weight = class_weight
        if self.class_weight is not None:
            assert isinstance(
                self, LassoNetClassifier
            ), "Weighted loss is only for classification"
            self.class_weight = torch.FloatTensor(self.class_weight).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=self.class_weight, reduction="mean"
            )


    @abstractmethod
    def _convert_y_D(self, y_reg, y_cla) -> torch.TensorType:
        """Convert y to torch tensor"""
        raise NotImplementedError

    @abstractstaticmethod
    def _output_shape_D(cls, y_reg, y_cla):
        """Number of model outputs"""
        raise NotImplementedError

    @abstractattr
    def criterion_D(cls):
        raise NotImplementedError
    

    @abstractmethod
    def _convert_y_A(self, y_MRI) -> torch.TensorType:
        """Convert y to torch tensor"""
        raise NotImplementedError

    @abstractstaticmethod
    def _output_shape_A(cls, y_MRI):
        """Number of model outputs"""
        raise NotImplementedError

    @abstractattr
    def criterion_A(cls):
        raise NotImplementedError

    def _init_model(self, X_MRI, X_SNP, y_reg, y_cla):
        """Create a torch model"""
        output_shape_reg, output_shape_cla = self._output_shape_D(y_reg, y_cla)
        if self.class_weight is not None:
            assert output_shape_cla == len(self.class_weight)
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)
        self.model_D = RegressorClassifier(
            X_MRI.shape[1], *self.hidden_dims_D, output_shape_reg, output_shape_cla, dropout=self.dropout
        ).to(self.device)
        self.model_A = LassoNet(
            X_SNP.shape[1], *self.hidden_dims_A, output_shape_reg+output_shape_cla, dropout=self.dropout
        ).to(self.device)
  

    def _cast_input(self, X_MRI=None, X_SNP=None, y_reg=None, y_cla=None):
        if y_reg is None and y_cla is None:
            if X_MRI is None:
                X_SNP = torch.FloatTensor(X_SNP).to(self.device)
                return X_SNP
            if X_SNP is None:
                X_MRI = torch.FloatTensor(X_MRI).to(self.device)
                return X_MRI

        X_MRI = torch.FloatTensor(X_MRI).to(self.device)
        X_SNP = torch.FloatTensor(X_SNP).to(self.device)
        y_reg, y_cla = self._convert_y_D(y_reg, y_cla)
        return X_MRI, X_SNP, y_reg, y_cla


    def fit(self, X_MRI, X_SNP, y_reg, y_cla, *, X_MRI_val=None, X_SNP_val=None, y_reg_val=None, y_cla_val=None):
        """Train the model.
        Note that if `lambda_` is not given, the trained model
        will most likely not use any feature.
        """
        self.path_ = self.path(X_MRI, X_SNP, y_reg, y_cla, X_MRI_val=X_MRI_val, X_SNP_val=X_SNP_val, y_reg_val=y_reg_val, y_cla_val=y_cla_val, return_state_dicts=False)
        return self

    def _train(
        self,
        X_MRI_train,
        X_SNP_train,
        y_reg_train,
        y_cla_train,
        X_MRI_val,
        X_SNP_val,
        y_reg_val,
        y_cla_val,
        *,
        batch_size,
        epochs,
        lambda_D,
        lambda_A,
        optimizer_D,
        optimizer_A,
        return_state_dict,
        patience=None,
    ) -> HistoryItem:
        model_D = self.model_D
        model_A = self.model_A

        def validation_obj_D():
            with torch.no_grad():
                y_model_reg_val, y_model_cla_val = model_D(X_MRI_val)
                return (
                    self.criterion_D(y_reg_val, y_cla_val, y_model_reg_val, y_model_cla_val).item()
                    + lambda_D * model_D.l1_regularization_skip().item()
                    + self.gamma_D * model_D.l2_regularization().item()
                    + self.gamma_skip_D * model_D.l2_regularization_skip().item()
                )
            
        def validation_obj_A():
            with torch.no_grad():
                y_MRI_val = torch.mm(X_MRI_val, torch.cat([model_D.skip_reg.weight.data, model_D.skip_cla.weight.data]).T) 
                return (
                    self.criterion_A(model_A(X_SNP_val), y_MRI_val).item()
                    + lambda_A * model_A.l1_regularization_skip().item()
                    + self.gamma_A * model_A.l2_regularization().item()
                    + self.gamma_skip_A * model_A.l2_regularization_skip().item()
                )
            
        best_val_obj_D = validation_obj_D()
        best_val_obj_A = validation_obj_A()
        epochs_since_best_val_obj_D = 0
        epochs_since_best_val_obj_A = 0
        if self.backtrack:
            best_state_dict_D = self.model_D.cpu_state_dict()
            real_best_val_obj_D = best_val_obj_D
            real_loss_D = float("nan")                    # if epochs == 0
            best_state_dict_A = self.model_A.cpu_state_dict()
            real_best_val_obj_A = best_val_obj_A
            real_loss_A = float("nan")                    # if epochs == 0

        n_iters_D = 0
        n_iters_A = 0

        n_train = len(X_MRI_train)
        if batch_size is None:
            batch_size = n_train
            randperm = torch.arange
        else:
            randperm = torch.randperm
        batch_size = min(batch_size, n_train)

        for epoch in range(epochs):
            indices = randperm(n_train)
            model_D.train()
            loss_D = 0
            model_A.train()
            loss_A = 0
            for i in range(n_train // batch_size):
                # don't take batches that are not full
                batch = indices[i * batch_size : (i + 1) * batch_size]

                def closure_D():
                    '''Model1 的反向传播'''
                    nonlocal loss_D
                    optimizer_D.zero_grad()
                    y_model_reg_train, y_model_cla_train = model_D(X_MRI_train[batch])
                    ans = (
                        self.criterion_D(y_reg_train[batch], y_cla_train[batch], y_model_reg_train, y_model_cla_train)
                        + self.gamma_D * model_D.l2_regularization()
                        + self.gamma_skip_D * model_D.l2_regularization_skip()
                    )
                    
                    ans.backward()
                    loss_D += ans.item() * len(batch) / n_train
                    return ans

                optimizer_D.step(closure_D)
                model_D.prox(lambda_=lambda_D * optimizer_D.param_groups[0]["lr"], M=self.M_D)


                def closure_A():
                    '''Model2 的反向传播'''
                    nonlocal loss_A
                    # Only use the output of the linear part to perform regression on genes
                    # y_MRI_train = torch.mm(X_MRI_train, torch.cat([model_D.skip_reg.weight.data, model_D.skip_cla.weight.data]).T)    
                    # Use the prediction results of the image model directly as the input of the genetic regression model
                    y_MRI_train = torch.cat(model_D(X_MRI_train[batch]), dim=1)      
                    optimizer_A.zero_grad()
                    ans = (
                        self.criterion_A(y_MRI_train[batch], model_A(X_SNP_train[batch]))
                        + self.gamma_A * model_A.l2_regularization()
                        + self.gamma_skip_A * model_A.l2_regularization_skip()
                    )
                    ans.backward()
                    loss_A += ans.item() * len(batch) / n_train
                    return ans

                optimizer_A.step(closure_A)
                model_A.prox(lambda_=lambda_A * optimizer_A.param_groups[0]["lr"], M=self.M_A)

            if epoch == 0:
                # fallback to running loss of first epoch
                real_loss_D = loss_D
                real_loss_A = loss_A
            
            val_obj_D = validation_obj_D()
            if val_obj_D < self.tol * best_val_obj_D:
                best_val_obj_D = val_obj_D
                epochs_since_best_val_obj_D = 0
            else:
                epochs_since_best_val_obj_D += 1
            if self.backtrack and val_obj_D < real_best_val_obj_D:
                best_state_dict_D = self.model_D.cpu_state_dict()
                real_best_val_obj_D = val_obj_D
                real_loss_D = loss_D
                n_iters_D = epoch + 1

            val_obj_A = validation_obj_A()
            if val_obj_A < self.tol * best_val_obj_A:
                best_val_obj_A = val_obj_A
                epochs_since_best_val_obj_A = 0
            else:
                epochs_since_best_val_obj_A += 1
            if self.backtrack and val_obj_A < real_best_val_obj_A:
                best_state_dict_A = self.model_A.cpu_state_dict()
                real_best_val_obj_A = val_obj_A
                real_loss_A = loss_A
                n_iters_A = epoch + 1

            if patience is not None and epochs_since_best_val_obj_D == patience and epochs_since_best_val_obj_A == patience:
                break

        if self.backtrack:
            self.model_D.load_state_dict(best_state_dict_D)
            val_obj_D = real_best_val_obj_D
            loss_D = real_loss_D
        else:
            n_iters_D = epoch + 1
        with torch.no_grad():
            reg_D = self.model_D.l1_regularization_skip().item()
            l2_regularization_D = self.model_D.l2_regularization()
            l2_regularization_skip_D = self.model_D.l2_regularization_skip()

        if self.backtrack:
            self.model_A.load_state_dict(best_state_dict_A)
            val_obj_A = real_best_val_obj_A
            loss_A = real_loss_A
        else:
            n_iters_A = epoch + 1
        with torch.no_grad():
            reg_A = self.model_A.l1_regularization_skip().item()
            l2_regularization_A = self.model_A.l2_regularization()
            l2_regularization_skip_A = self.model_A.l2_regularization_skip()

        return HistoryItem(
            lambda_=lambda_D,
            state_dict=self.model_D.cpu_state_dict() if return_state_dict else None,
            objective=loss_D + lambda_D * reg_D,
            loss=loss_D,
            val_objective=val_obj_D,
            val_loss=val_obj_D - lambda_D * reg_D,
            regularization=reg_D,
            l2_regularization=l2_regularization_D,
            l2_regularization_skip=l2_regularization_skip_D,
            selected=self.model_D.input_mask().cpu(),
            n_iters=n_iters_D,
        ), HistoryItem(
            lambda_=lambda_A,
            state_dict=self.model_A.cpu_state_dict() if return_state_dict else None,
            objective=loss_A + lambda_A * reg_A,
            loss=loss_A,
            val_objective=val_obj_A,
            val_loss=val_obj_A - lambda_A * reg_A,
            regularization=reg_A,
            l2_regularization=l2_regularization_A,
            l2_regularization_skip=l2_regularization_skip_A,
            selected=self.model_A.input_mask().cpu(),
            n_iters=n_iters_A,
        )

    @abstractmethod
    def predict_D(self, X_MRI):
        raise NotImplementedError
    
    @abstractmethod
    def predict_A(self, X_SNP):
        raise NotImplementedError

    def path(
        self,
        X_MRI,
        X_SNP,
        y_reg,
        y_cla,
        *,
        X_MRI_val=None,
        X_SNP_val=None,
        y_reg_val=None,
        y_cla_val=None,
        lambda_seq_D=None,
        lambda_seq_A=None,
        lambda_max=float("inf"),
        return_state_dicts=True,
        callback=None,
    ) -> List[HistoryItem]:
        """Train LassoNet on a lambda_ path.
        The path is defined by the class parameters:
        start at `lambda_start` and increment according to `path_multiplier`.
        The path will stop when no feature is being used anymore.
        callback will be called at each step on (model, history)
        """
        assert (X_MRI_val is None) == (
            y_reg_val is None
        ), "You must specify both or none of X_val and y_reg_val"
        sample_val = self.val_size != 0 and X_MRI_val is None and X_SNP_val is None
        if sample_val:
            X_MRI_train, X_MRI_val, X_SNP_train, X_SNP_val, y_reg_train, y_reg_val, y_cla_train, y_cla_val = train_test_split(
                X_MRI, X_SNP, y_reg, y_cla, test_size=self.val_size, random_state=self.random_state)
        elif X_MRI_val is None and X_SNP_val is None:
            X_MRI_train, X_SNP_train, y_reg_train, y_cla_train = X_MRI_val, X_SNP_val, y_reg_val, y_cla_val = X_MRI, X_SNP, y_reg, y_cla
        else:
            X_MRI_train, X_SNP_train, y_reg_train, y_cla_train = X_MRI, X_SNP, y_reg, y_cla
        X_MRI_train, X_SNP_train, y_reg_train, y_cla_train = self._cast_input(X_MRI_train, X_SNP_train, y_reg_train, y_cla_train)
        X_MRI_val, X_SNP_val, y_reg_val, y_cla_val = self._cast_input(X_MRI_val, X_SNP_val, y_reg_val, y_cla_val)

        hist_D: List[HistoryItem] = []
        hist_A: List[HistoryItem] = []

        # always init model
        self._init_model(X_MRI_train, X_SNP_train, y_reg_train, y_cla_train)
        
        temp_D, temp_A = self._train(
                                    X_MRI_train,
                                    X_SNP_train,
                                    y_reg_train,
                                    y_cla_train,
                                    X_MRI_val,
                                    X_SNP_val,
                                    y_reg_val,
                                    y_cla_val,
                                    batch_size=self.batch_size,
                                    lambda_D=0,
                                    lambda_A=0,
                                    epochs=self.n_iters_init,
                                    optimizer_D=self.optim_init_D(self.model_D.parameters()),
                                    optimizer_A=self.optim_init_A(self.model_A.parameters()),
                                    patience=self.patience_init,
                                    return_state_dict=return_state_dicts,
                                    )
        hist_D.append(temp_D)
        hist_A.append(temp_A)
        
        if callback is not None:
            callback(self, hist_D)
        if self.verbose > 1:
            print("Initialized dense model")
            hist_D[-1].log()
            hist_A[-1].log()

        optimizer_D = self.optim_path_D(self.model_D.parameters())
        optimizer_A = self.optim_path_A(self.model_A.parameters())

        # build lambda_seq
        if lambda_seq_D is not None:
            pass
        elif self.lambda_seq_D is not None:
            lambda_seq_D = self.lambda_seq_D
        else:

            def _lambda_seq(start):
                while start <= lambda_max:
                    yield start
                    start *= self.path_multiplier_D

            if self.lambda_start_D == "auto":
                # divide by 10 for initial training
                self.lambda_start_D = (
                    self.model_D.lambda_start(M=self.M_D)
                    / optimizer_D.param_groups[0]["lr"]
                    / 10
                )
                lambda_seq_D = _lambda_seq(self.lambda_start_D)
            else:
                lambda_seq_D = _lambda_seq(self.lambda_start_D)

        # extract first value of lambda_seq
        lambda_seq_D = iter(lambda_seq_D)
        lambda_start_D = next(lambda_seq_D)

        if lambda_seq_A is not None:
            pass
        elif self.lambda_seq_A is not None:
            lambda_seq_A = self.lambda_seq_A
        else:

            def _lambda_seq(start):
                while start <= lambda_max:
                    yield start
                    start *= self.path_multiplier_A

            if self.lambda_start_A == "auto":
                # divide by 10 for initial training
                self.lambda_start_A = (
                    self.model_A.lambda_start(M=self.M_A)
                    / optimizer_A.param_groups[0]["lr"]
                    / 10
                )
                lambda_seq_A = _lambda_seq(self.lambda_start_A)
            else:
                lambda_seq_A = _lambda_seq(self.lambda_start_A)

        # extract first value of lambda_seq
        lambda_seq_A = iter(lambda_seq_A)
        lambda_start_A = next(lambda_seq_A)

        is_dense = True
        for current_lambda_D, current_lambda_A in zip(itertools.chain([lambda_start_D], lambda_seq_D), itertools.chain([lambda_start_A], lambda_seq_A)):
            if self.model_D.selected_count() == 0 and self.model_A.selected_count() == 0:       # 如何保证模型同时达到稀疏
                break
            last_D, last_A = self._train(
                X_MRI_train,
                X_SNP_train,
                y_reg_train,
                y_cla_train,
                X_MRI_val,
                X_SNP_val,
                y_reg_val,
                y_cla_val,
                batch_size=self.batch_size,
                lambda_D=current_lambda_D,
                lambda_A=current_lambda_A,
                epochs=self.n_iters_path,
                optimizer_D=optimizer_D,
                optimizer_A=optimizer_A,
                patience=self.patience_path,
                return_state_dict=return_state_dicts,
            )
            if is_dense and self.model_D.selected_count() < X_MRI_train.shape[1] and self.model_A.selected_count() < X_SNP_train.shape[1]:
                is_dense = False
                if current_lambda_D / lambda_start_D < 2:
                    warnings.warn(
                        # f"lambda_start {self.lambda_start:.3f} "
                        f"lambda_start_D {self.lambda_start_D} "
                        "might be too large.\n"
                        f"Features start to disappear at {current_lambda_D:}."
                    )
                if current_lambda_A / lambda_start_A < 2:
                    warnings.warn(
                        # f"lambda_start {self.lambda_start:.3f} "
                        f"lambda_start_A {self.lambda_start_A} "
                        "might be too large.\n"
                        f"Features start to disappear at {current_lambda_A:}."
                    )

            hist_D.append(last_D)
            hist_A.append(last_A)
            if callback is not None:
                callback(self, hist_D)
            if callback is not None:
                callback(self, hist_A)
                
            if self.verbose > 1:
                print(
                    f"Lambda = {current_lambda_D:.2e}, "
                    f"selected {self.model_D.selected_count()} features "
                )
                last_D.log()
                print(
                    f"Lambda = {current_lambda_A:.2e}, "
                    f"selected {self.model_A.selected_count()} features "
                )
                last_A.log()

        self.feature_importances_D = self._compute_feature_importances(hist_D)
        self.feature_importances_A = self._compute_feature_importances(hist_A)
        """When does each feature disappear on the path?"""

        return hist_D, hist_A

    @staticmethod
    def _compute_feature_importances(path: List[HistoryItem]):
        """When does each feature disappear on the path?

        Parameters
        ----------
        path : List[HistoryItem]

        Returns
        -------
            feature_importances_
        """

        current = path[0].selected.clone()
        ans = torch.full(current.shape, float("inf"))
        for save in islice(path, 1, None):
            lambda_ = save.lambda_
            diff = current & ~save.selected
            ans[diff.nonzero().flatten()] = lambda_
            current &= save.selected
        return ans

    def load(self, state_dict_D, state_dict_A):
        if isinstance(state_dict_D, HistoryItem):
            state_dict_D = state_dict_D.state_dict
        if self.model_D is None:
            output_shape_reg, input_shape = state_dict_D["skip_reg.weight"].shape
            output_shape_cla, input_shape = state_dict_D["skip_cla.weight"].shape
            self.model_D = RegressorClassifier(
                input_shape, *self.hidden_dims_D, output_shape_reg, output_shape_cla, dropout=self.dropout
            ).to(self.device)

        if isinstance(state_dict_A, HistoryItem):
            state_dict_A = state_dict_A.state_dict
        if self.model_A is None:
            output_shape, input_shape = state_dict_A["skip.weight"].shape
            self.model_A = LassoNet(
                input_shape, *self.hidden_dims_A, output_shape, dropout=self.dropout
            ).to(self.device)

        self.model_D.load_state_dict(state_dict_D)
        self.model_A.load_state_dict(state_dict_A)
        return self


class LassoNetRegressor(
    RegressorMixin,
    MultiOutputMixin,
    BaseLassoNet,
):
    """Use LassoNet as regressor"""

    def _convert_y(self, y):
        y = torch.FloatTensor(y).to(self.device)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        return y

    @staticmethod
    def _output_shape(y):
        return y.shape[1]

    criterion = torch.nn.MSELoss(reduction="mean")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            ans = self.model(self._cast_input(X))
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans


class LassoNetClassifier(
    ClassifierMixin,
    BaseLassoNet,
):
    """Use LassoNet as classifier"""

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def _convert_y(self, y) -> torch.TensorType:
        y = torch.LongTensor(y).to(self.device)
        assert len(y.shape) == 1, "y must be 1D"
        return y

    @staticmethod
    def _output_shape(y):
        return (y.max() + 1).item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            ans = self.model(self._cast_input(X)).argmax(dim=1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            ans = torch.softmax(self.model(self._cast_input(X)), -1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans


class LassoNetRegressorClassifier(
    RegressorMixin,
    MultiOutputMixin,
    ClassifierMixin,
    BaseLassoNet,
):
    """Use LassoNet as regressor and classifier"""
    '''
    criterion_reg = torch.nn.MSELoss(reduction="mean")
    criterion_cla = torch.nn.CrossEntropyLoss(reduction="mean")

    '''
    # Brain image regression and classification fusion
    def criterion_D(self, y_reg, y_cla, y_reg_pred, y_cla_pred):
        criterion_reg = F.mse_loss(y_reg_pred, y_reg)
        cla_log_softmax = F.log_softmax(y_cla_pred)
        criterion_cla = F.nll_loss(cla_log_softmax, y_cla)
        return criterion_cla + criterion_reg

    def _convert_y_D(self, y_reg, y_cla):
        y_reg = torch.FloatTensor(y_reg).to(self.device)
        if len(y_reg.shape) == 1:
            y_reg = y_reg.view(-1, 1)
        y_cla = torch.LongTensor(y_cla).to(self.device)
        assert len(y_cla.shape) == 1, "y_cla must be 1D"
        return y_reg, y_cla

    @staticmethod
    def _output_shape_D(y_reg, y_cla):
        return y_reg.shape[1], (y_cla.max() + 1).item()

    def predict_D(self, X_MRI):
        self.model_D.eval()
        with torch.no_grad():
            ans_reg, ans_cla = self.model_D(self._cast_input(X_MRI=X_MRI))
            ans_cla = ans_cla.argmax(dim=1)         # Output category, 0 or 1
        if isinstance(X_MRI, np.ndarray):
            ans_reg = ans_reg.cpu().numpy()
            ans_cla = ans_cla.cpu().numpy()

        return ans_reg, ans_cla


    def predict_proba_D(self, X_MRI):
        self.model_D.eval()
        with torch.no_grad():
            ans_reg, ans_cla = self.model_D(self._cast_input(X_MRI=X_MRI))
            ans_cla = torch.softmax(ans_cla, -1)
        if isinstance(X_MRI, np.ndarray):
            ans_reg = ans_reg.cpu().numpy()
            ans_cla = ans_cla.cpu().numpy()
        return ans_reg, ans_cla
    

    # Imaging and genetic regression
    def _convert_y_A(self, y_MRI):
        y_MRI = torch.FloatTensor(y_MRI).to(self.device)
        if len(y_MRI.shape) == 1:
            y_MRI = y_MRI.view(-1, 1)
        return y_MRI

    @staticmethod
    def _output_shape_A(y_MRI):
        return y_MRI.shape[1]

    criterion_A = torch.nn.MSELoss(reduction="mean")

    def predict_A(self, X_SNP):
        self.model_A.eval()
        with torch.no_grad():
            ans = self.model_A(self._cast_input(X_SNP=X_SNP))
        if isinstance(X_SNP, np.ndarray):
            ans = ans.cpu().numpy()
        return ans
    

class BaseLassoNetCV(BaseLassoNet, metaclass=ABCMeta):
    def __init__(self, cv=None, **kwargs):
        """
        See BaseLassoNet for the parameters

        cv : int, cross-validation generator or iterable, default=None
            Determines the cross-validation splitting strategy.
            Default is 5-fold cross-validation.
            See <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.check_cv.html>
        """
        super().__init__(**kwargs)
        self.cv = check_cv(cv)

    def path(
        self,
        X,
        y,
        *,
        return_state_dicts=True,
    ):
        raw_lambdas_ = []
        self.raw_scores_ = []
        self.raw_paths_ = []

        # TODO: parallelize
        for train_index, test_index in tqdm(
            self.cv.split(X, y),
            total=self.cv.get_n_splits(X, y),
            desc="Choosing lambda with cross-validation",
            disable=self.verbose == 0,
        ):
            split_lambdas = []
            split_scores = []
            raw_lambdas_.append(split_lambdas)
            self.raw_scores_.append(split_scores)

            def callback(model, hist):
                split_lambdas.append(hist[-1].lambda_)
                split_scores.append(model.score(X[test_index], y[test_index]))

            path = super().path(
                X[train_index],
                y[train_index],
                return_state_dicts=False,  # avoid memory cost
                callback=callback,
            )
            self.raw_paths_.append(path)

        # build final path
        lambda_ = min(sl[1] for sl in raw_lambdas_)
        lambda_max = max(sl[-1] for sl in raw_lambdas_)
        self.lambdas_ = []
        while lambda_ < lambda_max:
            self.lambdas_.append(lambda_)
            lambda_ *= self.path_multiplier

        # interpolate new scores
        self.interp_scores_ = np.stack(
            [
                np.interp(np.log(self.lambdas_), np.log(sl[1:]), ss[1:])
                for sl, ss in zip(raw_lambdas_, self.raw_scores_)
            ],
            axis=-1,
        )

        # select best lambda based on cross_validation
        best_lambda_idx = np.nanargmax(self.interp_scores_.mean(axis=1))
        self.best_lambda_ = self.lambdas_[best_lambda_idx]
        self.best_cv_scores_ = self.interp_scores_[best_lambda_idx]
        self.best_cv_score_ = self.best_cv_scores_.mean()

        if self.lambda_start == "auto":
            # forget the previously estimated lambda_start
            self.lambda_start_ = self.lambdas_[0]

        # train with the chosen lambda sequence
        path = super().path(
            X,
            y,
            lambda_seq=self.lambdas_[: best_lambda_idx + 1],
            return_state_dicts=return_state_dicts,
        )
        self.path_ = path

        self.best_selected_ = path[-1].selected
        return path

    def fit(
        self,
        X,
        y,
    ):
        """Train the model.
        Note that if `lambda_` is not given, the trained model
        will most likely not use any feature.
        """
        self.path(X, y, return_state_dicts=False)
        return self