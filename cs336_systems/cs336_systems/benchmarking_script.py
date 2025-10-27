from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import argparse


from timeit import default_timer as timer


def run_and_benchmark(warmup: int, steps: int):
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=128,
        d_model=64,
        num_layers=4,
        num_heads=4,
        d_ff=64,
        rope_theta=10000.0,
    )
    batch_size = 4
    vocab_size = 10000
    seq_len = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # generate some random inputs
    batch_size = 4
    print("model init done; starting benchmarking")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # warmup
    for _ in range(warmup):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    print("done with warmup")

    # start benchmarking
    times_fwd, times_bwd = [], []
    for _ in range(steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)

        # ---- Forward ----
        torch.cuda.synchronize()
        start_fwd = timer()
        logits = model(x)
        torch.cuda.synchronize()
        end_fwd = timer()

        # ---- Backward ----
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        torch.cuda.synchronize()
        start_bwd = timer()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        end_bwd = timer()

        times_fwd.append(end_fwd - start_fwd)
        times_bwd.append(end_bwd - start_bwd)

    print("done with benchmarking")
    print(f"times in forward pass \n", times_fwd)
    print(f"times in backward pass \n", times_bwd)
    print(f"mean of forward pass ,{np.mean(times_fwd)}")
    print(f"std of forward pass ,{np.std(times_fwd)}")
    print(f"mean of backward pass ,{np.mean(times_bwd)}")
    print(f"std of backward pass ,{np.std(times_bwd)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmarking script")
    parser.add_argument("--warmup", type=int, required=True, help="number of warmups")
    parser.add_argument("--steps", type=int, required=True, help="number of steps for benchmarking")

    args = parser.parse_args()

    run_and_benchmark(args.warmup, args.steps)
