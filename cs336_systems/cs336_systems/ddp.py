import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
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
