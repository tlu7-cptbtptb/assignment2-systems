import torch
import torch.distributed as dist
from typing import Type, Any


class ShardedStateOptimizer(torch.optim.Optimizer):
    """
    Optimizer that shards optimizer state across ranks (ZeRO Stage 1).

    Each rank only maintains optimizer state for a subset of parameters,
    reducing memory usage. After each step, updated parameters are broadcast
    to all ranks.
    """

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any) -> None:
        # Convert params to list to allow multiple iterations
        params = list(params)

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # Deduplicate params (for tied weights) using data_ptr
        seen_ptrs = set()
        unique_params = []
        for param in params:
            ptr = param.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                unique_params.append(param)

        # Store unique params for gradient zeroing and param sync
        self.all_params = unique_params
        self._seen_ptrs = seen_ptrs

        # Assign each unique param to a rank (round-robin)
        self.param_to_rank = {}
        for i, param in enumerate(unique_params):
            self.param_to_rank[param.data_ptr()] = i % self.world_size

        # Get only the params this rank is responsible for
        self.local_params = [p for p in unique_params if self.param_to_rank[p.data_ptr()] == self.rank]

        # Create the underlying optimizer with only local params
        if self.local_params:
            self.optimizer = optimizer_cls(self.local_params, **kwargs)
        else:
            self.optimizer = None

        # Initialize the base Optimizer class with unique params
        defaults = kwargs.copy()
        super().__init__(unique_params, defaults)

    def step(self, closure=None, **kwargs):
        """
        Calls the wrapped optimizer's step() method with the provided closure and keyword arguments.
        After updating the parameters, synchronize with the other ranks.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Each rank updates only its local params
        if self.optimizer is not None:
            self.optimizer.step(**kwargs)

        # Broadcast updated params from owning rank to all other ranks
        for param in self.all_params:
            owner_rank = self.param_to_rank[param.data_ptr()]
            dist.broadcast(param.data, src=owner_rank)

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        """
        This method should add a parameter group to the sharded optimizer.
        This is called during construction of the sharded optimizer by the super-class constructor
        and may also be called during training (e.g., for gradually unfreezing layers in a model).
        As a result, this method should handle assigning the model's parameters among the ranks.
        """
        # Get new params from the param_group
        new_params = param_group["params"]
        if isinstance(new_params, torch.Tensor):
            new_params = [new_params]
        else:
            new_params = list(new_params)

        # Deduplicate new params (skip those already tracked)
        unique_new_params = []
        for param in new_params:
            ptr = param.data_ptr()
            if ptr not in self._seen_ptrs:
                self._seen_ptrs.add(ptr)
                unique_new_params.append(param)

        if not unique_new_params:
            return

        # Assign new unique params to ranks (round-robin)
        start_idx = len(self.all_params)
        for i, param in enumerate(unique_new_params):
            self.param_to_rank[param.data_ptr()] = (start_idx + i) % self.world_size
            self.all_params.append(param)

        # Update param_group to only have unique params
        param_group = param_group.copy()
        param_group["params"] = unique_new_params

        # Add to base optimizer's param_groups
        super().add_param_group(param_group)

        # Get local params from this new group
        new_local_params = [p for p in unique_new_params if self.param_to_rank[p.data_ptr()] == self.rank]

        # Add to underlying optimizer if we have local params
        if new_local_params:
            if self.optimizer is None:
                self.optimizer = self.optimizer_cls(new_local_params, **self.optimizer_kwargs)
            else:
                local_param_group = param_group.copy()
                local_param_group["params"] = new_local_params
                self.optimizer.add_param_group(local_param_group)

    def zero_grad(self, set_to_none: bool = True):
        """
        Zero gradients for all params (not just local ones).
        """
        for param in self.all_params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()
