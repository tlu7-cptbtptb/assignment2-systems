import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def generate_sample_data(seed=42):
    """Generate sample data with fixed seed for reproducibility."""
    torch.manual_seed(seed)
    batch_size = 128
    num_dim = 4
    data = torch.randn(batch_size, num_dim)
    return data


def get_init_params(in_dim, out_dim, device=None, layer_idx=0, seed=1234):
    """
    Initialize parameters with a fixed seed for reproducibility.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        device: Device to place the tensor on (None for CPU, or 'cuda:0', etc.)
        layer_idx: Layer index used to create unique but reproducible seeds per layer
        seed: Base random seed

    Returns:
        A parameter tensor with requires_grad=True
    """
    # Use layer_idx to create unique but reproducible initialization per layer
    torch.manual_seed(seed + layer_idx)
    param = torch.randn(in_dim, out_dim)
    if device is not None:
        param = param.to(device)
    param = param.requires_grad_(True)
    return param


def summarize_tensor(t: torch.Tensor) -> str:
    """Summarize a tensor for printing."""
    return f"mean={t.mean().item():.6f}, std={t.std().item():.6f}, norm={t.norm().item():.6f}"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # change to nccl for GPU
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def non_parallel_train(data: torch.Tensor, num_layers: int, num_steps: int, device: str = "cuda:0") -> None:
    num_dim = data.size(1)  # @inspect num_dim
    # Move data to GPU
    data = data.to(device)
    # Create MLP parameters params[0], ..., params[num_layers - 1]
    # Use layer_idx to ensure same initialization as DDP
    params = [get_init_params(num_dim, num_dim, device=device, layer_idx=i) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for step in range(num_steps):
        optimizer.zero_grad()
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        # Backward pass
        loss.backward()
        # Print gradients for comparison
        if step == 0:
            print(
                f"[Non-DDP]: step = {step}, gradients = {[summarize_tensor(p.grad) for p in params]}",
                flush=True,
            )
        # Update parameters
        optimizer.step()
        print(
            f"[Non-DDP] step = {step}, loss = {loss.item():.6f}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}",
            flush=True,
        )


def data_parallelism_main(
    rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int, flatten_grad: bool = False
) -> None:
    print(f"""Starting DDP,  flatten_grad = {flatten_grad}""")
    setup(rank, world_size)

    # Set the device for this rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = batch_size // world_size  # @inspect local_batch_size
    print(
        f"tlu7 ... rank = {rank}, world_size = {world_size}, batch_size = {batch_size}, local_batch_size = {local_batch_size}"
    )
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    # Move local data to this rank's GPU
    local_data = data[start_index:end_index].to(device)

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    # Use layer_idx to ensure same initialization across ranks
    params = [get_init_params(num_dim, num_dim, device=device, layer_idx=i) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state
    for step in range(num_steps):
        optimizer.zero_grad()
        # Forward pass
        x = local_data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        # Backward pass
        loss.backward()
        # Sync gradients across workers (only difference between standard training and DDP)
        if flatten_grad:
            grads = [p.grad for p in params]  # nested, needed for the unflatten below!
            flat_grads = _flatten_dense_tensors(grads)
            dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG, async_op=False)
            _unflatten_dense_tensors(flat_grads, grads)

        else:
            for param in params:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Print gradients for comparison (before optimizer step)
        if step == 0:
            print(
                f"[DDP] Rank {rank}: step = {step}, gradients (after all_reduce) = {[summarize_tensor(p.grad) for p in params]}",
                flush=True,
            )
        # Update parameters
        optimizer.step()
        dist.all_reduce(tensor=loss, op=dist.ReduceOp.AVG, async_op=False)
        print(
            f"[DDP] Rank {rank}: step = {step}, loss = {loss.item():.6f}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}",
            flush=True,
        )
    cleanup()


if __name__ == "__main__":
    # take args from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--flatten_grad", action="store_true", help="Use flattened gradient all_reduce")
    args = parser.parse_args()

    world_size = 2
    num_layers = 2
    num_steps = 3

    # Generate data with fixed seed for reproducibility
    data = generate_sample_data(seed=42)
    print(f"Data shape: {data.shape}")
    print("=" * 80)

    # Run non-parallel training first
    print("\n" + "=" * 80)
    print("NON-PARALLEL TRAINING (full batch)")
    print("=" * 80)
    non_parallel_train(data, num_layers, num_steps)

    # Run DDP training
    print("\n" + "=" * 80)
    print(f"DDP TRAINING (world_size={world_size})")
    print("=" * 80)
    mp.spawn(
        fn=data_parallelism_main,
        args=(world_size, data, num_layers, num_steps, args.flatten_grad),
        nprocs=world_size,
        join=True,
    )

    print("\n" + "=" * 80)
    print("COMPARISON NOTES:")
    print("- Non-DDP uses full batch, DDP uses local batches")
    print("- After all_reduce with AVG, DDP gradients should match Non-DDP gradients")
    print("- Both should have identical parameter updates if gradients match")
    print("=" * 80)
