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
        Q: :, Nq, d
        K: :, Nk, d
        V: :, Nk, d
        """
        B, Nq, d = Q.shape
        _, Nk, _ = K.shape
        # the Output:
        O = torch.empty_like(Q)
        L = torch.zeros(B, Nq)
        print(f"tlu7 ... Q.shape = {Q.shape}, K.shape = {K.shape}, V.shape = {V.shape}")

        Bq = 16  # tile size
        Bk = 16

        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)
        # Split Q into Tq = ⌈Nq/Bq⌉ tiles Q1, . . . , QTq of size Bq × d
        Q_tiles = []
        for t in range(Tq):
            Q_tiles.append(Q[:, t * Bq : (t + 1) * Bq, :])
        # Split K, V into Tq = ⌈Nk/Bk⌉ tiles K1, . . K_Tk and tiles V1 ... V_Tk of size Bk × d
        K_tiles = []
        V_tiles = []
        for t in range(Tk):
            K_tiles.append(K[:, t * Bk : (t + 1) * Bk, :])
            V_tiles.append(V[:, t * Bk : (t + 1) * Bk, :])

        # outer loop
        for i in range(Tq):
            Q_i = Q_tiles[i]
            print(f"tlu7 ... i = {i}, Q_i.shape = {Q_i.shape}")
            O_i_prev = torch.zeros(B, Bq, d)  # will increment this along the inner loop
            l_i_prev = torch.zeros(B, Bq)  # will increment this along the inner loop
            m_i_prev = torch.full(
                (
                    B,
                    Bq,
                ),
                -torch.inf,
            )
            # inner loop
            for j in range(Tk):
                K_j = K_tiles[j]
                V_j = V_tiles[j]
                print(f"tlu7 ... j = {j}, K_j.shape = {K_j.shape}")
                # Compute tile of pre-softmax attention scores
                S_ij = torch.matmul(Q_i, K_j.transpose(-1, -2)) / math.sqrt(d)  # B x Bq x Bk
                m_i_j = torch.maximum(m_i_prev, S_ij.max(dim=-1).values)  # B x Bq
                P_tmp_ij = torch.exp(S_ij - m_i_j.unsqueeze(-1))  # unnormalized softmax,  B xBq x Bk
                l_i_j = torch.exp(m_i_prev - m_i_j) * l_i_prev + P_tmp_ij.sum(dim=-1)  # B x Bq
                # O_i_j = torch.matmul(torch.diag(torch.exp(m_i_prev - m_i_j)), O_i_prev) + torch.matmul(
                #     P_tmp_ij, V_j
                # )  # Bq x d
                scale = torch.exp(m_i_prev - m_i_j).unsqueeze(-1)
                print(
                    f"tlu7 ... scale.shape = {scale.shape}, O_i_prev.shape = {O_i_prev.shape}, P_tmp_ij.shape = {P_tmp_ij.shape}, V_j.shape"
                )
                O_i_j = scale * O_i_prev + (P_tmp_ij @ V_j)  # B x Bq x d

                # update the _prev variables
                O_i_prev = O_i_j
                m_i_prev = m_i_j
                l_i_prev = l_i_j

            # O_i = (1.0 / torch.diag(l_i_j)) @ O_i_j  # Bq x d
            O_i = O_i_j / l_i_j.unsqueeze(-1)  # [B, Bq_i, d]
            L_i = m_i_j + torch.log(l_i_j)  # B x Bq
            O[:, i * Bq : (i + 1) * Bq, :] = O_i
            L[:, i * Bq : (i + 1) * Bq] = L_i
        ctx.save_for_backward(L, K, Q, V, O)
        return O

    @staticmethod
    def backward(ctx, Q, K, V, O, dO, L) -> Any:
        raise NotImplementedError
