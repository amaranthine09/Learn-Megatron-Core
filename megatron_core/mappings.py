"""Megatron Core Tensor Parallel communication mappings (f and g operators).

In Megatron-LM tensor parallelism:
- Operator f:
    Forward: Identity (pass through)
    Backward: All-reduce (sum gradients across TP group)
- Operator g:
    Forward: All-reduce (sum outputs across TP group)
    Backward: Identity (pass through)
"""

import torch
import torch.distributed as dist


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass input through in forward, all-reduce gradients in backward (Operator f)."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_output)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce input in forward, pass gradients through in backward (Operator g)."""

    @staticmethod
    def forward(ctx, input_):
        if dist.is_available() and dist.is_initialized():
            # Clone input before in-place all-reduce
            output = input_.clone()
            dist.all_reduce(output)
            return output
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input along the last dimension across model parallel ranks."""

    @staticmethod
    def forward(ctx, input_):
        if not (dist.is_available() and dist.is_initialized()):
            return input_
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        dim_size = input_.size(-1) // world_size
        return input_.split(dim_size, dim=-1)[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if not (dist.is_available() and dist.is_initialized()):
            return grad_output
        world_size = dist.get_world_size()
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(tensor_list, grad_output)
        return torch.cat(tensor_list, dim=-1)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather tensors along the last dimension across model parallel ranks."""

    @staticmethod
    def forward(ctx, input_):
        if not (dist.is_available() and dist.is_initialized()):
            return input_
        world_size = dist.get_world_size()
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_)
        return torch.cat(tensor_list, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        if not (dist.is_available() and dist.is_initialized()):
            return grad_output
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        dim_size = grad_output.size(-1) // world_size
        return grad_output.split(dim_size, dim=-1)[rank].contiguous()


# Public helper functions
def copy_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)
