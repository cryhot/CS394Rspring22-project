#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from typing import *

import numpy as np

from .features import FeatureVector

__all__ = [
    'ValueFunction',

    'ValueFunctionWithFeatureVector',
    'CompositeValueFunctionFromFeatureVector',
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
        """Update the value of given state (yet, update can affect the other states)."""
        if alpha is None: alpha=self.alpha  # step size
        self.w += alpha * (G - self(state)) * self.X[state]
    
    def __getitem__(self, state): return self.X[state]

    @property
    def shape(self): return self.X.shape

    @property
    def size(self): return self.X.size

    @property
    def ndim(self): return self.X.ndim


class CompositeValueFunctionFromFeatureVector(ValueFunction):
    """Usefull to have several ValueFunction evaluator together (e.g., several NNs)."""
    def __init__(self,
        X:FeatureVector,
        V:Callable[[np.ndarray], ValueFunction],
    ):
        """Has one V for each X value.
        Args:
            X (FeatureVector): _description_
            V (Callable[[np.ndarray], ValueFunction]): either a ValueFunction with a copy() method, either a function x -> ValueFunction.
        """
        self.X = X
        self._V_creator = V
        self._V = dict()
        super().__init__()
    
    def _new_V(self, x):
        if isinstance(self._V_creator, ValueFunction):
            return self._V_creator.copy()
        if callable(self._V_creator):
            return self._V_creator(x)
    
    def _get_V(self, state):
        x = self.X[state]
        selx = tuple(x) # makes it hashable
        if not selx in self._V:
            self._V[selx] = self._new_V(x)
        V = self._V[selx]
        return V

    def __call__(self, state) -> float:
        """Return the value of given state."""
        V = self._get_V(state)
        value = V(state)
        return value

    def update(self, state, G:float, alpha:float=None):
        """Update the value of given state (yet, update can affect the other states)."""
        V = self._get_V(state)
        return V.update(state, G, alpha=alpha)
        