from cs336_basics.model import BasicsTransformerLM, CausalMultiHeadSelfAttention, RotaryEmbedding
from cs336_basics.optimizer import AdamW
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

import numpy as np
import pandas as pd
import argparse


from timeit import default_timer as timer


def run_and_benchmark_single_config(warmup: int, steps: int, head_dim: int, seq_len: int, do_torch_compile: int):
    rope = RotaryEmbedding(context_length=seq_len, dim=head_dim)
    attention_layer = CausalMultiHeadSelfAttention(d_model=head_dim, num_heads=1, positional_encoder=rope)
    if do_torch_compile:
        attention_layer = torch.compile(attention_layer)
    batch_size = 8
    vocab_size = 10000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    attention_layer.to(device)
    # generate some random inputs
    print(f"model init done; starting benchmarking for head_dim={head_dim}, seq_len={seq_len}")

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # warmup
    for _ in range(warmup):
        x = torch.rand((batch_size, seq_len, head_dim), device=device)
        # targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # optimizer.zero_grad(set_to_none=True)
        output = attention_layer(x)
        # loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        # loss.backward()
        # optimizer.step()
        torch.cuda.synchronize()
    print("done with warmup")

    # start benchmarking
    times_fwd, times_bwd = [], []
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(steps):
        x = torch.rand((batch_size, seq_len, head_dim), device=device)
        # targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        # optimizer.zero_grad(set_to_none=True)

        # ---- Forward ----
        nvtx.range_push("benchmark_step_fwrd_1")
        torch.cuda.synchronize()
        start_fwd = timer()
        output = attention_layer(x)
        torch.cuda.synchronize()
        end_fwd = timer()
        nvtx.range_pop()

        # # ---- Backward ----
        # nvtx.range_push("benchmark_step_bwrd_1")
        # loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        # torch.cuda.synchronize()
        # start_bwd = timer()
        # loss.backward()
        # optimizer.step()
        # torch.cuda.synchronize()
        # end_bwd = timer()
        # nvtx.range_pop()

        times_fwd.append(end_fwd - start_fwd)
        # times_bwd.append(end_bwd - start_bwd)

    torch.cuda.memory._dump_snapshot(f"memory_snapshot_d_model_{head_dim}_seq_len_{seq_len}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    print("done with benchmarking")
    print(f"times in forward pass \n", times_fwd)
    # print(f"times in backward pass \n", times_bwd)
    print(f"mean of forward pass ,{np.mean(times_fwd)}")
    # print(f"std of forward pass ,{np.std(times_fwd)}")
    # print(f"mean of backward pass ,{np.mean(times_bwd)}")
    # print(f"std of backward pass ,{np.std(times_bwd)}")
    print(f"-----------------------")
    print(f"")


def run_benchmark_attention(warmup: int, steps: int, head_dims: str, seq_lens: str, do_torch_compile: int):
    head_dims = [int(x) for x in head_dims.split(",")]
    seq_lens = [int(x) for x in seq_lens.split(",")]
    for head_dim in head_dims:
        for seq_len in seq_lens:
            run_and_benchmark_single_config(warmup, steps, head_dim, seq_len, do_torch_compile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmarking attention script")
    parser.add_argument("--warmup", type=int, required=True, help="number of warmups")
    parser.add_argument("--steps", type=int, required=True, help="number of steps for benchmarking")
    parser.add_argument("--head_dims", type=str, required=True, help="head dimension(s)")
    parser.add_argument("--seq_lens", type=str, required=True, help="seq length(s)")
    parser.add_argument("--do_torch_compile", type=int, default=0, required=False, help="if do torch compile")

    args = parser.parse_args()

    run_benchmark_attention(args.warmup, args.steps, args.head_dims, args.seq_lens, args.do_torch_compile)
