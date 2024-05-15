from math import exp

import numpy as np
from matplotlib import pyplot as plt  # type: ignore

from chaos import Lorenz
from cma_es import CMAES
from delay_system import DelaySystem
from prediction import LocalLinarPredictor

iteration = 100

def eval_history(true:list[float], pred:list[float]) -> float:
    steps = len(true)
    return sum((true[i] - pred[i]) ** 2 * exp(-i/10) for i in range(steps))

def eval_parm(parm:np.ndarray):
    tau, epsilon = parm
    if tau < 0 or epsilon < 0:
        return 1e10
    rng = np.random.default_rng()
    init_state:tuple[float, float, float] = tuple(rng.normal(10, 0.1, 3)) # type: ignore
    model = Lorenz(10, 25, 2.66, init_state, dt=0.02, observe_func=lambda x: x[0] + x[1])
    model_delay = DelaySystem(3, tau, model)
    model_delay.run(100)
    llp = LocalLinarPredictor(model_delay, epsilon)
    model_delay.clear_history()
    model_delay.run(7)
    try:
        llp.run(7)
    except:
        return 1e10
    return eval_history(model_delay.history, llp.history)

es = CMAES(func=eval_parm, init_mean=np.array([0.2, 0.4]), init_sigma=0.1, nsample=10)
means_parm = np.zeros((iteration, 2))
means_out = np.zeros((iteration, 1))

for i in range(iteration):
    es.sample()
    es.evaluate()
    es.update_param()
    mean_parm = es.mean
    mean_out = eval_parm(mean_parm)
    means_parm[i] = mean_parm
    means_out[i] = mean_out

plt.plot(means_out)
plt.show()

print(means_parm[-1], means_out[-1])