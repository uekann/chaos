from abc import ABCMeta, abstractmethod
from operator import itemgetter
from typing import Callable, Generic, TypeVar

T = TypeVar('T')

class Chaos(Generic[T],metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None:
        self._state_full:T
        self.dt:float
        self._observe_func:Callable[[T],float]
        self._history_full:list[T]
        self._history:list[float]
        
    @property
    def state(self) -> float:
        return self._observe_func(self._state_full)
    
    @property
    def history(self) -> list[float]:
        return self._history.copy()
    
    @property
    def state_full(self) -> T:
        return self._state_full
    
    @property
    def history_full(self) -> list[T]:
        return self._history_full.copy()
    
    @abstractmethod
    def step(self) -> float:
        pass

    def run(self, time: float) -> list[float]:
        steps = int(time / self.dt)
        for _ in range(steps):
            self.step()
        return self.history
    
    def clear_history(self) -> None:
        self._history.clear()
        self._history_full.clear()
        self._history_full.append(self.state_full)
        self._history.append(self.state)

class Lorenz(Chaos[tuple[float, float, float]]):
    def __init__(self, sigma:float, rho:float, beta:float, init_state: tuple[float, float, float] = (1, 2, 3), dt: float = 0.01, observe_func:Callable[[tuple[float, float, float]],float] = itemgetter(0)):
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self._state_full = init_state
        self._history_full = [init_state]
        self.dt = dt
        self._observe_func = observe_func
        self._history = [observe_func(init_state)]

    def step(self) -> float:
        x, y, z = self.state_full
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        self._state_full = (x + dx * self.dt, y + dy * self.dt, z + dz * self.dt)
        self._history_full.append(self.state_full)
        self._history.append(self.state)
        return self.state

class Henon(Chaos[tuple[float, float]]):
    def __init__(self, a:float, b:float, init_state: tuple[float, float] = (1, 2), dt: float = 1, observe_func:Callable[[tuple[float, float]],float] = itemgetter(0)):
        super().__init__()
        self.a = a
        self.b = b
        self._state_full = init_state
        self._history_full = [init_state]
        self.dt = dt
        self._observe_func = observe_func
        self._history = [observe_func(init_state)]

    def step(self) -> float:
        x, y = self.state_full
        self._state_full = (1 - self.a * (x ** 2) + self.b * y, x)
        self._history_full.append(self.state_full)
        self._history.append(self.state)
        return self.state