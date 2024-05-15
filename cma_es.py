from typing import Callable

import numpy as np


class CMAES(object):
    """CMA Evolution Strategy with CSA"""

    def __init__(self, func:Callable[[np.ndarray], float], init_mean:np.ndarray, init_sigma:float, nsample:int):
        """コンストラクタ

        Parameters
        ----------
        func : callable
            目的関数 (最小化)
        init_mean : ndarray (1D)
            初期平均ベクトル
        init_sigma : float
            初期ステップサイズ
        nsample : int
            サンプル数
        """
        self.func = func
        self.mean = init_mean
        self.sigma = init_sigma
        self.N = self.mean.shape[0]  # 探索空間の次元数
        self.arx = np.zeros((nsample, self.N)) * np.nan  # 候補解
        self.arf = np.zeros(nsample) * np.nan  # 候補解の評価値

        self.D = np.ones(self.N)  # 共分散行列の固有値
        self.B = np.eye(self.N)  # 共分散行列の固有ベクトル
        self.C = np.dot(self.B * self.D, self.B.T)  # 共分散行列

        self.weights = np.zeros(nsample)
        self.weights[: nsample // 4] = 1.0 / (nsample // 4)  # 重み．総和が1

        # For CSA
        self.ps = np.zeros(self.N)
        self.mueff = 1.0 / np.sum(self.weights**2)
        self.cs = (2.0 + self.mueff) / (self.N + 3.0 + self.mueff)
        self.ds = 1.0 + self.cs + max(1.0, np.sqrt(self.mueff / self.N))
        self.chiN = np.sqrt(self.N) * (
            1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N * self.N)
        )

        # For CMA
        self.cmu = self.mueff / (self.N**2 / 2 + self.N + self.mueff)

    def sample(self):
        """候補解を生成する．"""
        self.arz = np.random.normal(size=self.arx.shape)
        self.ary = np.dot(np.dot(self.arz, self.B) * np.sqrt(self.D), self.B.T)
        self.arx = self.mean + self.sigma * self.ary

    def evaluate(self):
        """候補解を評価する．"""
        for i in range(self.arf.shape[0]):
            self.arf[i] = self.func(self.arx[i])

    def update_param(self):
        """パラメータを更新する．"""
        idx = np.argsort(self.arf)  # idx[i]は評価値がi番目に良い解のインデックス
        # 進化パスの更新 (平均ベクトル移動量の蓄積)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * np.dot(self.weights, self.arz[idx])
        # 共分散行列の更新
        self.C = (1 - self.cmu) * self.C + self.cmu * np.dot(
            self.ary[idx].T * self.weights, self.ary[idx]
        )
        # 共分散行列の固有値分解
        self.D, self.B = np.linalg.eigh(self.C)

        # 進化パスの長さが，ランダム関数の下での期待値よりも大きければステップサイズを大きくする．
        self.sigma = self.sigma * np.exp(
            self.cs / self.ds * (np.linalg.norm(self.ps) / self.chiN - 1)
        )
        self.mean += np.dot(self.weights, self.arx[idx] - self.mean)