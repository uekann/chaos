from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter

import numpy as np
from sortedcontainers import SortedList  # type: ignore

from chaos import Chaos
from delay_system import DelaySystem


def norm(x: tuple[float,...], y: tuple[float,...]) -> float:
    return sum((xi - yi) ** 2 for xi, yi in zip(x, y)) ** 0.5

class Predictor(Chaos[tuple[float, ...]],metaclass=ABCMeta):
    
    def __init__(self, model:Chaos):
        super().__init__()
        self.dt = model.dt
        self._observe_func = model._observe_func
        self._history_full_train = model.history_full
        self._state_full = model.state_full
        self._history_full = [self.state_full]
        self._history = [self.state]
        

class LocalConstantPredictor(Predictor):
    
    def __init__(self, model: Chaos, epsilon: float):
        super().__init__(model)
        self.epsilon = epsilon
        self._neighbors = []
        for i, x in enumerate(self._history_full_train):
            if norm(x, self.state_full) < self.epsilon:
                self._neighbors.append(i)
        self._p = 0
        self._num_neighbors = len(self._neighbors)
        if self._num_neighbors == 0:
            raise ValueError('No neighbors found')
    
    def step(self) -> float:
        self._p += 1
        s = (0,) * len(self.history_full)
        c = 0
        for i in self._neighbors:
            if i + self._p >=  len(self._history_full_train):continue
            
            si = self._history_full_train[i + self._p]
            s = tuple(sij + sj for sij, sj in zip(si, s))
            c += 1
        
        self._state_full = tuple(sij / c for sij in s)
        
        self._history_full.append(self.state_full)
        self._history.append(self.state)
        return self.state


class WeightedLocalConstantPredictor(Predictor):
    
    def __init__(self, model: Chaos, epsilon: float):
        super().__init__(model)
        self.epsilon = epsilon
        self._neighbors = []
        self._gains = dict()
        for i, x in enumerate(self._history_full_train):
            n = norm(x, self.state_full)
            if n < self.epsilon:
                self._neighbors.append(i)
                self._gains[i] = 1 / (n + 1e-7)
        self._p = 0
    
    def step(self) -> float:
        self._p += 1
        s = (0,) * len(self.history_full)
        gain_sum:float = 0
        for i in self._neighbors:
            if i + self._p >=  len(self._history_full_train):continue
            
            si = self._history_full_train[i + self._p]
            s = tuple(sij * self._gains[i] + sj for sij, sj in zip(si, s))
            gain_sum += self._gains[i]
        
        self._state_full = tuple(sij / gain_sum for sij in s)
        
        self._history_full.append(self.state_full)
        self._history.append(self.state)
        return self.state


class LocalLinarPredictor(Predictor):
    
    def __init__(self, model: DelaySystem, epsilon: float = 0.4, k: int = 20):
        self.dt = model.dt
        self._observe_func = itemgetter(0)
        self._history_full_train = model.history_full
        self._history_train = model.history
        self._state_full = tuple([model.state])
        self._history_full = [self.state_full]
        self._history = [self.state]
        self._first_state_full:np.ndarray = np.array([1] + list(model.state_full))
        
        self.epsilon = epsilon
        self._neighbors = []
        for i, x in enumerate(self._history_full_train):
            if i + 1 == len(self._history_full_train):break
            if norm(x, model.state_full) < self.epsilon:
                self._neighbors.append(i)
        
        # self.k = k
        # self._norms = [norm(x, self.state_full) for x in self._history_full_train]
        # self._norms_sorted = SortedList(self._norms)
        # self._norm2idx:dict[float, list[int]] = defaultdict(list)
        # for i, n in enumerate(self._norms):
        #     self._norm2idx[n].append(i)
        self._p = 0
    
    def step(self) -> float:
        self._p += 1
        # p = self._norms.pop()
        # self._norms_sorted.remove(p)
        # self._norm2idx[p].pop()
        
        # idxs = [i for n in self._norms_sorted[:self.k] for i in self._norm2idx[n]]
        
        idxs = [i for i in self._neighbors if i + self._p < len(self._history_train)]
        if len(idxs) == 0:
            raise ValueError('No neighbors found')
        Y = np.array([self._history_train[i + self._p] for i in idxs])
        X = np.array([[1] + list(self._history_full_train[i]) for i in idxs])
        
        w = np.linalg.inv(X.T @ X) @ X.T @ Y
        
        self._state_full = tuple([w @ self._first_state_full])
        
        self._history_full.append(self.state_full)
        self._history.append(self.state)
        return self.state