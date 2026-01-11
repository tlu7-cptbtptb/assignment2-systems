import os
import sys
import time
import warnings
import contextlib
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Set spawn method for CUDA compatibility
mp.set_start_method("spawn", force=True)

# Suppress GLOO warnings about hostname resolution
os.environ["GLOO_SOCKET_IFNAME"] = "lo0"  # Use loopback interface on macOS (use "lo" on Linux)
warnings.filterwarnings("ignore")


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr (for C++ warnings)."""
    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Suppress C++ warnings by redirecting stderr temporarily
    with suppress_stderr():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def benchmark_all_reduce(rank, world_size, num_elements, num_warmup=5, num_iters=20, result_queue=None):
    """
    Benchmark all_reduce operation.

    Args:
        rank: Process rank
        world_size: Total number of processes
        num_elements: Number of float32 elements in the tensor
        num_warmup: Number of warmup iterations
        num_iters: Number of timed iterations
        result_queue: Queue to return timing results from rank 0
    """
    setup(rank, world_size)

    # Set the device for this rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create tensor with specified number of elements on GPU
    data = torch.randn(num_elements, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(num_warmup):
        dist.all_reduce(data, async_op=False)

    # Synchronize before timing
    dist.barrier()

    # Timed iterations
    start_time = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(data, async_op=False)
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) / num_iters * 1000  # Convert to ms

    # Only rank 0 reports the result
    if rank == 0 and result_queue is not None:
        result_queue.put(avg_time_ms)

    cleanup()


def run_benchmark(world_size, num_elements, num_warmup=5, num_iters=20):
    """
    Run benchmark with specified configuration.

    Returns:
        Average time in milliseconds
    """
    # Use Manager queue for spawn context compatibility
    manager = mp.Manager()
    result_queue = manager.Queue()
    mp.spawn(
        fn=benchmark_all_reduce,
        args=(world_size, num_elements, num_warmup, num_iters, result_queue),
        nprocs=world_size,
        join=True,
    )
    return result_queue.get()


def bytes_to_elements(num_bytes):
    """Convert bytes to number of float32 elements."""
    return num_bytes // 4  # float32 is 4 bytes


def format_size(num_bytes):
    """Format bytes as human-readable string."""
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.0f}GB"
    elif num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.0f}MB"
    elif num_bytes >= 1e3:
        return f"{num_bytes / 1e3:.0f}KB"
    else:
        return f"{num_bytes}B"


if __name__ == "__main__":
    # Configuration
    num_processes_list = [2]
    data_sizes_bytes = [
        1 * 1024 * 1024,  # 1MB
        10 * 1024 * 1024,  # 10MB
        100 * 1024 * 1024,  # 100MB
        1024 * 1024 * 1024,  # 1GB
    ]

    num_warmup = 5
    num_iters = 20

    print("=" * 70)
    print("All-Reduce Benchmark")
    print("=" * 70)
    print("Backend: nccl")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Timed iterations: {num_iters}")
    print("=" * 70)
    print()

    # Results table header
    print(f"{'Data Size':<12} | ", end="")
    for np in num_processes_list:
        print(f"{np} procs (ms)".center(15) + " | ", end="")
    print()
    print("-" * 70)

    # Run benchmarks
    results = {}
    for data_size in data_sizes_bytes:
        size_str = format_size(data_size)
        num_elements = bytes_to_elements(data_size)

        print(f"{size_str:<12} | ", end="", flush=True)

        for world_size in num_processes_list:
            avg_time_ms = run_benchmark(
                world_size=world_size,
                num_elements=num_elements,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            results[(size_str, world_size)] = avg_time_ms
            print(f"{avg_time_ms:>12.3f}   | ", end="", flush=True)

        print()

    print("-" * 70)
    print()

    # Print bandwidth analysis
    print("Bandwidth Analysis (GB/s):")
    print("-" * 70)
    print(f"{'Data Size':<12} | ", end="")
    for np in num_processes_list:
        print(f"{np} procs".center(15) + " | ", end="")
    print()
    print("-" * 70)

    for data_size in data_sizes_bytes:
        size_str = format_size(data_size)
        print(f"{size_str:<12} | ", end="")

        for world_size in num_processes_list:
            time_ms = results[(size_str, world_size)]
            time_s = time_ms / 1000
            # All-reduce transfers 2*(n-1)/n * data_size bytes theoretically
            # For simplicity, we report raw bandwidth = data_size / time
            bandwidth_gbps = (data_size / 1e9) / time_s
            print(f"{bandwidth_gbps:>12.3f}   | ", end="")

        print()

    print("-" * 70)
