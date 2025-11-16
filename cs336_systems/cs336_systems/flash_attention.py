import torch
from typing import Any
import math


class FlashAttention2(torch.autograd.Function):
    """
    This is a PyTorch autograd.Function that implements FlashAttention2.
    It is a wrapper around the FlashAttention2 C++/CUDA code.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False) -> Any:
        """
        Q: Nq x d
        K: Nk x d
        V: Nk x d
        """
        Bq = 16  # tile size
        Bk = 16

        Nq = Q.shape[0]
        Nk = K.shape[0]
        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)
        # Split Q into Tq = ⌈Nq/Bq⌉ tiles Q1, . . . , QTq of size Bq × d
        Q_tiles = []
        for t in range(Tq):
            Q_tiles.append(Q[t * Bq : (t + 1) * Bq, :])
        # Split K, V into Tq = ⌈Nk/Bk⌉ tiles K1, . . K_Tk and tiles V1 ... V_Tk
        K_tiles = []
        V_tiles = []
        for t in range(Tk):
            K_tiles.append(K[t * Bk : (t + 1) * Bk, :])
            V_tiles.append(V[t * Bk : (t + 1) * Bk, :])

    @staticmethod
    def backward(ctx, Q, K, V, O, dO, L) -> Any:
        raise NotImplementedError
