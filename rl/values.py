#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from typing import *

import numpy as np

from .features import FeatureVector

__all__ = [
    'ValueFunction',

    'ValueFunctionWithFeatureVector',
]


class ValueFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, state, G:float, alpha:float=None):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

class ValueFunctionWithFeatureVector(ValueFunction, FeatureVector):
    """ValueFunction from FeatureVector"""
    def __init__(self, X:FeatureVector, w:np.ndarray=None, alpha:float=None):
        if w is None: w = np.zeros(X.shape)  # weight vector

        self.X = X          # feature vector
        self.w = w          # weight vector
        self.alpha = alpha  # step size

        super().__init__()
    
    def __call__(self, state) -> float:
        """Return the value of given state."""
        value = np.sum(self.X[state] * self.w,
            axis=tuple(range(-self.X.ndim,0)), # in case multiple states are used at once
        )
        return value

    def update(self, state, G:float, alpha:float=None):
        """Update the value of given state (yet, update can affect the other states).

        Args:
            state: target state for updating
            G (float): TD-target
            alpha (float, optional): learning rate (default is self.alpha)
        """
        if alpha is None: alpha=self.alpha  # step size
        self.w += alpha * (G - self(state)) * self.X[state]
    
    def __getitem__(self, state): return self.X[state]

    @property
    def shape(self): return self.X.shape

    @property
    def size(self): return self.X.size

    @property
    def ndim(self): return self.X.ndim

