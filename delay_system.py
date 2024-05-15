from operator import itemgetter
from typing import Callable

from chaos import Chaos


class DelaySystem(Chaos[tuple[float, ...]]):
    def __init__(self, m:int, tau:float, model:Chaos):
        super().__init__()
        self.m = m
        self.dt = model.dt
        self.delay_step = int(tau / self.dt)
        self.model = model
        for _ in range(self.delay_step * (m - 1)):
            self.model.step()
        log = model.history
        self._state_full:tuple[float,...] = tuple(log[-1 - i * self.delay_step] for i in range(m))
        self._history_full = [self.state_full]
        self._observe_func = itemgetter(0)
        self._history = [self.state]
    
    def step(self) -> float:
        self.model.step()
        log = self.model.history
        self._state_full = tuple(log[-1 - i * self.delay_step] for i in range(self.m))
        self._history_full.append(self.state_full)
        self._history.append(self.state)
        return self.state