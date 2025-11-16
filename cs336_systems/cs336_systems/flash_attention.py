import torch
from typing import Any
import math
import triton
import triton.language as tl


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


@triton.jit
def flash_fwd_kernel(
    Q_ptr,  # [B, Nq, D]
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(0, 1),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # will be updated in the inner loop below, thus setting to 0 here
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),  # will be updated in the inner loop below, thus setting to 0 here
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )
    # Output block pointers (where we'll store result for this query tile)
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(0, 1),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    # Note: on chip (SRAM) buffers (Oi, l, m) should have dtype tl.float32.
    # (1) running max per row (BQ,)
    m_i_prev = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    # (2) running denominator per row (BQ,)
    l_i_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # (3) running partial output per row (BQ, D)
    O_i_prev = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Load the Q tile for this thread
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))
    Q_i = tl.cast(Q_i, tl.float32)

    # inner loop over K, V tile
    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))
        K_j = tl.cast(K_j, tl.float32)
        V_j = tl.cast(V_j, tl.float32)
        # Compute tile of pre-softmax attention scores
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # Bq x Bk
        # row-wise max
        m_i_j = tl.maximum(m_i_prev, tl.max(S_ij, axis=1))
        m_i_j_b = tl.reshape(m_i_j, (Q_TILE_SIZE, 1))
        P_tmp_ij = tl.exp(S_ij - m_i_j_b)  # unnormalized softmax,  Bq x Bk
        l_i_j = tl.exp(m_i_prev - m_i_j) * l_i_prev + tl.sum(
            P_tmp_ij, axis=1
        )  # Bq; rowwise sum of exp of the unnormalized softmax
        scale_o = tl.exp(m_i_prev - m_i_j)
        O_i_j = scale_o * O_i_prev + tl.dot(P_tmp_ij, V_j)  # Bq x D
        # update the _prev variables
        O_i_prev = O_i_j
        m_i_prev = m_i_j
        l_i_prev = l_i_j

        # update the K, V pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    l_bcast = tl.reshape(l_i_prev, (Q_TILE_SIZE, 1))
    O_i = O_i_prev / l_bcast  # (Bq, d)
    L_i = m_i_j + tl.log(l_i_j)  # Bq

    # store to output
    tl.store(O_block_ptr, tl.cast(O_i, tl.float32), boundary_check=(0, 1))
    tl.store(L_block_ptr, tl.cast(L_i, tl.float32), boundary_check=(0,))


@triton.jit
def flash_fwd_kernel_gpt(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # program ids: one program per (query_tile, batch)
    query_tile_index = tl.program_id(0)  # range Tq = cdiv(N_QUERIES, Q_TILE_SIZE)
    batch_index = tl.program_id(1)

    # Block pointers for row-major tensors -> use order=(0,1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(0, 1),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )

    # Output block pointers (store)
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(0, 1),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q tile and cast to float32 for stable accumulation
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))  # shape (Q_TILE_SIZE, D)
    Q_i = tl.cast(Q_i, tl.float32)

    # initialize running state (use finite -inf sentinel)
    m_i_prev = tl.full((Q_TILE_SIZE,), -1e20, dtype=tl.float32)  # -inf approx
    l_i_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i_prev = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # precompute row mask for tail queries
    start_q = query_tile_index * Q_TILE_SIZE
    row_idx = start_q + tl.arange(0, Q_TILE_SIZE)
    row_mask = row_idx < N_QUERIES  # shape (Q_TILE_SIZE,)

    # inner loop over K tiles
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for key_tile_index in range(num_k_tiles):
        # load K_j, V_j and cast to float32
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1))  # (K_TILE_SIZE, D) or padded
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))  # (K_TILE_SIZE, D) or padded
        K_j = tl.cast(K_j, tl.float32)
        V_j = tl.cast(V_j, tl.float32)

        # compute column mask for last K tile columns (so exp of OOB columns = 0)
        start_k = key_tile_index * K_TILE_SIZE
        col_idx = start_k + tl.arange(0, K_TILE_SIZE)
        col_mask = col_idx < N_KEYS  # shape (K_TILE_SIZE,)

        # compute scores: (BQ, D) @ (D, BK) -> (BQ, BK)
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        # row-wise max
        row_max = tl.max(S_ij, axis=1)  # (Q_TILE_SIZE,)
        m_i_j = tl.maximum(m_i_prev, row_max)  # (Q_TILE_SIZE,)

        # broadcasting shapes
        m_i_j_b = tl.reshape(m_i_j, (Q_TILE_SIZE, 1))  # (BQ,1)
        m_i_prev_b = tl.reshape(m_i_prev, (Q_TILE_SIZE, 1))

        # unnormalized softmax for the tile, zero OOB columns via col_mask
        P_tmp_ij = tl.exp(S_ij - m_i_j_b)  # (BQ, BK)
        # apply column mask to zero out-of-bounds columns
        P_tmp_ij = P_tmp_ij * tl.reshape(tl.cast(col_mask, tl.float32), (1, K_TILE_SIZE))

        # running sum update
        sum_exp_new = tl.sum(P_tmp_ij, axis=1)  # (BQ,)
        exp_old_factor = tl.exp(m_i_prev - m_i_j)  # (BQ,)
        l_i_j = exp_old_factor * l_i_prev + sum_exp_new  # (BQ,)

        # update O: partial = P_tmp_ij @ V_j  -> (BQ, D)
        partial = tl.dot(P_tmp_ij, V_j)  # (BQ, D)

        exp_old_b = tl.reshape(exp_old_factor, (Q_TILE_SIZE, 1))
        O_i_j = exp_old_b * O_i_prev + partial  # (BQ, D)

        # assign for next iter
        O_i_prev = O_i_j
        m_i_prev = m_i_j
        l_i_prev = l_i_j

        # advance K/V block pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # finalize using the running _prev variables
    l_bcast = tl.reshape(l_i_prev, (Q_TILE_SIZE, 1))  # (BQ,1)
    O_final = O_i_prev / l_bcast  # (BQ, D)
    L_final = m_i_prev + tl.log(l_i_prev)  # (BQ,)

    # store with masks (cast back to output dtype later in wrapper if needed)
    # row_mask: (BQ,) -> used as mask for store; for O (BQ, D) use row_mask[:, None]
    tl.store(O_block_ptr, tl.cast(O_final, tl.float32))
    tl.store(L_block_ptr, tl.cast(L_final, tl.float32))


@triton.jit
def flash_fwd_kernel_safe(
    Q_ptr,  # [B, Nq, D]
    K_ptr,  # [B, Nk, D]
    V_ptr,  # [B, Nk, D]
    O_ptr,  # [B, Nq, D]
    L_ptr,  # [B, Nq]
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(1)
    q_tile_idx = tl.program_id(0)

    # load Q tile
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))
    Q_i = tl.cast(Q_i, tl.float32)

    # init output accumulators
    O_i_prev = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i_prev = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_i_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for k_tile_idx in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load K/V tile
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_idx * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(k_tile_idx * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_idx * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(k_tile_idx * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # boundary check last key tile
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))
        K_j = tl.cast(K_j, tl.float32)
        V_j = tl.cast(V_j, tl.float32)

        # mask for last key tile (ignore padded keys)
        key_mask = tl.arange(0, K_TILE_SIZE) + k_tile_idx * K_TILE_SIZE < N_KEYS
        key_mask = tl.cast(key_mask, tl.float32)  # 1.0 for valid, 0.0 for invalid

        # attention scores
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (BQ, BK)
        S_ij = S_ij * key_mask[None, :] + (-1e9) * (1.0 - key_mask[None, :])  # mask invalid keys

        # numerically stable softmax accumulation
        m_i_j = tl.maximum(m_i_prev, tl.max(S_ij, axis=1))
        P_tmp = tl.exp(S_ij - tl.reshape(m_i_j, (Q_TILE_SIZE, 1)))
        l_i_j = tl.exp(m_i_prev - m_i_j) * l_i_prev + tl.sum(P_tmp, axis=1)
        scale_o = tl.exp(m_i_prev - m_i_j)
        O_i_j = scale_o[:, None] * O_i_prev + tl.dot(P_tmp, V_j)

        # update prev
        O_i_prev = O_i_j
        m_i_prev = m_i_j
        l_i_prev = l_i_j

    # finalize using the running _prev variables
    l_bcast = tl.reshape(l_i_prev, (Q_TILE_SIZE, 1))  # (BQ, 1)
    O_final = O_i_prev / l_bcast  # (BQ, D)
    L_final = m_i_prev + tl.log(l_i_prev)  # (BQ,)

    # make block pointers for final store
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # store results using block pointers
    tl.store(O_block_ptr, tl.cast(O_final, tl.float32))
    tl.store(L_block_ptr, tl.cast(L_final, tl.float32))


class FlashAttention2Triton(torch.autograd.Function):
    """
    This is a autograd.Function that wraps the Triton kernel above
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
        O = torch.empty_like(Q, device=Q.device, dtype=torch.float32)
        L = torch.zeros(B, Nq, device=Q.device, dtype=torch.float32)
        print(
            f"tlu7 ... Q.shape = {Q.shape}, K.shape = {K.shape}, V.shape = {V.shape} , batch size =  {B} Nq = {Nq}, Nk = {Nk}, d =  {d}"
        )

        ctx.Bq = 16
        ctx.Bk = 16
        ctx.scale = 1.0 / math.sqrt(d)
        ctx.D = d
        print(f"tlu7 ... Q.stride = {Q.stride()}, K.stride = {K.stride()}, V.stride = {V.stride()}")
        flash_fwd_kernel[
            (
                math.ceil(Nq / ctx.Bq),
                B,
            )
        ](
            Q_ptr=Q,  # [Nq, D]
            K_ptr=K,
            V_ptr=V,
            O_ptr=O,
            L_ptr=L,
            stride_qb=Q.stride(0),  # should this just be Nq * d?
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lb=L.stride(0),
            stride_lq=L.stride(1),
            N_QUERIES=Nq,
            N_KEYS=Nk,
            scale=ctx.scale,
            D=ctx.D,
            Q_TILE_SIZE=ctx.Bq,
            K_TILE_SIZE=ctx.Bk,
        )
        ctx.save_for_backward(K, Q, V, O, L)
        return O

    @staticmethod
    def backward(ctx, Q, K, V, O, dO, L) -> Any:
        raise NotImplementedError


class FlashAttention2TritonGPT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False) -> Any:
        """
        Q: [B, Nq, D]
        K: [B, Nk, D]
        V: [B, Nk, D]
        returns O: [B, Nq, D]
        """
        B, Nq, d = Q.shape
        _, Nk, _ = K.shape

        # outputs: accumulate in float32 internally; cast to Q.dtype on store if desired
        O = torch.empty(B, Nq, d, device=Q.device, dtype=Q.dtype)
        L = torch.empty(B, Nq, device=Q.device, dtype=torch.float32)

        # tile params (tune for your hardware)
        Bq = 16
        Bk = 16

        ctx.Bq = Bq
        ctx.Bk = Bk
        ctx.scale = float(1.0 / math.sqrt(d))
        ctx.D = d

        # compute integer grid size
        grid_q = (Nq + Bq - 1) // Bq
        grid = (grid_q, B)

        # launch kernel
        flash_fwd_kernel_safe[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            Nq,
            Nk,
            ctx.scale,
            ctx.D,
            ctx.Bq,
            ctx.Bk,
        )

        # save for backward (note: heavy - for reference/backtesting only)
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward not implemented yet")
