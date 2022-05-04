#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
from typing import *

import numpy as np
import itertools
import functools

__all__ = [
    'FeatureVector',

    'OneHotFeatureVector',
    'TileFeatureVector',

    'ProductFeatureVector',
    'FlatFeatureVector',
]


class FeatureVector(abc.ABC):

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int]:
        """Return the size of the feature vector"""
        raise NotImplementedError()

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the feature vector"""
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @abc.abstractmethod
    def __getitem__(self, state:np.ndarray) -> np.ndarray:
        """Return the feature vector for the given state."""
        raise NotImplementedError()
    # def __call__(self, s, done, a=None) -> np.array:
    #     """
    #     implement function x: S+ x A -> [0,1]^d
    #     if done is True, then return 0^d
    #     """
    #     raise NotImplementedError()


    @classmethod
    def prod(cls, *args, **kwargs):
        return ProductFeatureVector(*args, **kwargs)
    
    @property
    def flat(self):
        return FlatFeatureVector(self)



class OneHotFeatureVector(FeatureVector):

    def __init__(self, n):
        assert np.all(n >= 0)
        self.dtype = np.bool_
        self.n = np.asarray(n, dtype=int)

    @property
    def shape(self):
        return tuple(self.n.flat)
    
    def __getitem__(self, state):
        # TODO: assert state.dtype is discrete
        state = np.asarray(state, dtype=int)
        assert np.all((0 <= state) & (state < self.n)), "state components should be between 0 and n."
        # TODO: use sparse matrix
        x = np.zeros(self.n, self.dtype)
        x[tuple(state.flat)] = True
        return x


class TileFeatureVector(FeatureVector):

    @classmethod
    def uniform_offsets(cls,
        low:np.array,
        high:np.array,
        width:np.array,
        num:int,
        axis=None,
    ):
        """_summary_
        num: number of tilings
        axis: axes to offset (default: all axes)
        """
        # offsets = np.linspace(0, width, num=num, endpoint=False, dtype=low.dtype)
        if axis is None: axis = range(low.ndim)
        offsets = np.zeros((num,*low.shape), dtype=low.dtype)
        for a in axis:
            offsets[...,a] = np.linspace(0, width[a], num=num, endpoint=False)
        return cls(low, high, width, offsets)

    def __init__(self,
        low:np.array,
        high:np.array,
        width:np.array,
        offsets:np.array,
    ):
        """
        low: possible minimum value for each dimension in state
        high: possible maximum value for each dimension in state
        width: tile width for each dimension
        offsets: offset for each tiling and in each dimension
        """
        # assumptions
        if low.ndim != 1: raise NotImplementedError()
        assert low.shape == high.shape
        assert width.shape == low.shape
        assert width.shape == low.shape
        assert offsets.ndim >= low.ndim
        shape_tilings = offsets.shape[:offsets.ndim-low.ndim] # shape of the array of tilings (possibly multidimensional)
        ndim_tilings = len(shape_tilings)
        try: assert np.broadcast_shapes(offsets.shape[ndim_tilings:], low.shape) == low.shape
        except (ValueError, AssertionError): raise AssertionError("last dimensions of offsets should match the state dimension")

        # create tiles
        tiles = np.zeros(shape=shape_tilings, dtype=[
            ('low', low.dtype, low.shape),
            ('high', high.dtype, high.shape),
        ])
        tiles['low'] = offsets % width - width
        for d,l,h,w in zip(itertools.count(), low, high, width):
            tiles_low = np.arange(l, h+w, w)
            tiles = np.tile(tiles[...,np.newaxis], len(tiles_low)) # add dimension d
            tiles['low'][...,d] += tiles_low
            tiles['high'][...,:-1,d] = tiles['low'][...,1:,d]
            tiles['high'][...,-1,d] = tiles['low'][...,-1,d]+w
        
        # guarantees
        t0 = tiles.flat[0]
        assert np.allclose(tiles['high'], tiles['low']+width), "wrong tile width"
        assert np.allclose(tiles.shape[:ndim_tilings], shape_tilings), "wrong number of tilings"
        assert np.allclose(tiles.shape[ndim_tilings:], np.ceil((high-low)/width)+1), "wrong number of tiles"

        self.tiles = tiles
        self.ndim_tilings = ndim_tilings
        self.dtype = np.bool_

    @property
    def shape(self):
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.tiles.shape

    def __getitem__(self, state) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        assert np.shape(state) == self.tiles.dtype['low'].shape, "must evaluate a single state"
        x = np.all(
            (self.tiles['low'] < state) & (state <= self.tiles['high']),
            axis = -1,
        )
        assert np.count_nonzero(x) == np.prod(self.tiles.shape[:self.ndim_tilings]), "point not covered by the expected number of tiles"
        return x



class ProductFeatureVector(FeatureVector):
    """See also FeatureVector.prod
    Example usage:
    X = ProductFeatureVector(OneHotFeatureVector(...), TileFeatureVector(...))
    X = ProductFeatureVector([X1, X2, ...])
    X = ProductFeatureVector((X1, X2), state_indices=(0,[1,2]))
    """
    def __init__(self, Xs, *more_Xs, state_indices=None):
        """
        Args:
            *Xs: sequence of FeatureVector, one for each state part.
            state_indices: Specifies how a state is decomposed in parts.
                           If None (default), parts are state[0],...,state[-1].
                           If it is a tuple of arrays/indices, parts are state[indices] for each indices in state_indices.
        """
        super().__init__()
        if isinstance(Xs, FeatureVector): Xs = (Xs,)
        else: Xs = tuple(Xs)
        Xs += more_Xs
        if state_indices is not None:
            state_indices = tuple(state_indices)
            assert len(state_indices) == len(Xs)
        self.Xs = Xs
        self.state_indices = state_indices
    
    def _iter_state(self, state):
        """Return an object iterating through the state parts."""
        if self.state_indices is None:
            return state
        else:
            return (
                state[indices]
                for indices in self.state_indices
            )

    @property
    def shape(self):
        return sum((X.shape for X in self.Xs), tuple())

    def __getitem__(self, state):
        if not self.Xs: return np.empty(0,dtype=bool)
        x = functools.reduce(np.multiply.outer, (X[part] for X, part in zip(self.Xs, self._iter_state(state))))
        # for X, part in zip(self.Xs, self._iter_state(state)):
        #     print(">", X.shape, X[part].shape)
        # print(self.shape, x.shape)
        assert x.shape == self.shape
        return x


class FlatFeatureVector(FeatureVector):
    """See also FeatureVector.flat"""

    def __init__(self, X:FeatureVector):
        self.X = X

    @property
    def shape(self):
        return (self.X.size,)

    @property
    def size(self):
        return self.X.size

    @property
    def ndim(self):
        return 1

    def __getitem__(self, state):
        return self.X[state].flat
