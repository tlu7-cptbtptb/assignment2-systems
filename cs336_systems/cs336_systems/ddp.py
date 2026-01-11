import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        Given an instantiated PyTorch nn.Module to be
        parallelized, construct a DDP container that will handle gradient synchronization across ranks.
        """
        super().__init__()
        self.module = module

        # Broadcast parameters from rank 0 to all other ranks
        # so that all processes start with the same model weights
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module’s forward() method with the provided positional and keyword arguments.
        """
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        When called, wait for asynchronous communication to complete and calls to be queued on GPU
        """
        world_size = dist.get_world_size()

        # Asynchronous, return immediately after each call and wait on results at the end.
        handles = []
        for param in self.module.parameters():
            if param.grad is not None:
                handle = dist.all_reduce(param.grad, async_op=True, op=dist.ReduceOp.SUM)
                handles.append((handle, param))

        # Wait for all async operations and average the gradients
        for handle, param in handles:
            handle.wait()
            param.grad.div_(world_size)


class BucketDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient syn-
        chronization across ranks. Gradient synchronization should be bucketed, with each bucket holding
        at most bucket_size_mb of parameters.
        """
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size_num_param = int(bucket_size_mb * 1024 * 1024 / 4)  # assuming fp32

        # Broadcast parameters from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Bucket gradients in reverse parameter order (last layer first), flatten each bucket,
        launch async all_reduce, then wait and copy back.
        """
        world_size = dist.get_world_size()

        # Collect params with gradients in REVERSE order (last layer first)
        params_with_grads = [p for p in self.module.parameters() if p.grad is not None][::-1]

        # Build buckets
        buckets = []
        current_bucket = []
        current_bucket_size = 0

        for param in params_with_grads:
            param_size = param.grad.numel()
            if current_bucket_size + param_size > self.bucket_size_num_param and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [param]
                current_bucket_size = param_size
            else:
                current_bucket.append(param)
                current_bucket_size += param_size

        if current_bucket:
            buckets.append(current_bucket)

        # Launch async all_reduce for each bucket
        handles = []
        for bucket in buckets:
            grads = [p.grad for p in bucket]
            flat_grads = _flatten_dense_tensors(grads)
            handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
            handles.append((handle, flat_grads, bucket))

        # Wait, average, and copy back
        for handle, flat_grads, bucket in handles:
            handle.wait()
            flat_grads.div_(world_size)
            for unflat_grad, param in zip(_unflatten_dense_tensors(flat_grads, [p.grad for p in bucket]), bucket):
                param.grad.copy_(unflat_grad)
