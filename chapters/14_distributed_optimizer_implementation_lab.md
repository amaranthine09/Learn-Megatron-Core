# Distributed Optimizer: API Reference & Memory Accounting
> **Megatron-Core DistributedOptimizer, ParamAndGradBuffer, and ZeRO-2 Internals**

---

## 3.1. Megatron-Core DistributedOptimizer API

The production ZeRO-2 optimizer lives in `megatron.core.optimizer.distrib_optimizer.DistributedOptimizer`. It is instantiated via the `get_megatron_optimizer` factory:

```python
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

optimizer_config = OptimizerConfig(
    optimizer='adam',
    lr=1e-4,
    min_lr=1e-5,
    weight_decay=0.1,
    bf16=True,
    use_distributed_optimizer=True,   # Enables ZeRO-2 partitioning
    clip_grad=1.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_eps=1e-8,
)

optimizer = get_megatron_optimizer(
    config=optimizer_config,
    model_chunks=model,                # List of GPTModel instances
    no_weight_decay_cond=no_wd_fn,    # Optional: fn(name, param) -> bool
    scale_lr_cond=None,
    lr_mult=1.0,
)
```

---

## 3.2. ParamAndGradBuffer: Pre-Allocated Contiguous Arena

In production Megatron, parameters are **not** dynamically flattened via `torch.cat` at each step. Instead, they are pre-allocated in a single contiguous memory arena at model initialization:

```python
# megatron/core/distributed/param_and_grad_buffer.py

class ParamAndGradBuffer:
    """
    Pre-allocated contiguous buffer for all model parameters and their gradients.
    Parameters are views (not copies) into this buffer, enabling:
    1. Zero-copy reduce_scatter on the gradient buffer directly.
    2. No dynamic memory allocation during training steps.
    3. Aligned bucket boundaries for efficient NCCL communication.
    """
    def __init__(
        self,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[nn.Parameter],
        data_parallel_group: dist.ProcessGroup,
        bucket_size: Optional[int],
        param_to_name: Dict[nn.Parameter, str],
        gradient_scaling_factor: float,
    ):
        # Allocates a single flat buffer; parameters are re-pointed as views
        self.param_data = torch.zeros(total_numel, dtype=param_dtype, device='cuda')
        self.grad_data = torch.zeros(total_numel, dtype=grad_dtype, device='cuda')

        # Remap parameters to be views into the contiguous buffer
        for param, start, end in param_index_map:
            param.data = self.param_data[start:end].view(param.shape)
```

> [!NOTE]
> This pre-allocation is why Megatron avoids the "flatten-then-cat" overhead seen in naive ZeRO implementations. `param.data` itself **is** the slice of the contiguous buffer — no copy ever occurs during reduce-scatter gradient reduction.

---

## 3.3. ZeRO-2 Gradient Reduction: Reduce-Scatter

Megatron's `DistributedDataParallel` wrapping fires `dist.reduce_scatter_tensor` on the gradient buffer bucket after the backward pass:

```python
# megatron/core/distributed/distributed_data_parallel.py

class DistributedDataParallel(MegatronModule):

    def finish_grad_sync(self):
        """
        Fires gradient synchronization. With use_distributed_optimizer=True,
        this executes Reduce-Scatter instead of All-Reduce (ZeRO-2 gradient partitioning).
        """
        for bucket in self.bucket_groups:
            # Reduce-Scatter: every rank receives only its 1/D gradient slice
            dist.reduce_scatter_tensor(
                output=bucket.grad_data[self.shard_start:self.shard_end],
                input=bucket.grad_data,
                op=dist.ReduceOp.AVG,       # Averages across DP ranks
                group=self.data_parallel_group,
            )
```

**Memory savings**: Standard DDP uses `All-Reduce`, keeping the full gradient buffer $[2\Phi]$ bytes on every rank. ZeRO-2's `Reduce-Scatter` means Rank $r$ only retains its $\frac{1}{D}$ gradient shard after reduction. Gradient memory scales as $\frac{2\Phi}{D}$.

---

## 3.4. FP32 Master Weights & Adam State Partitioning

Each DP rank maintains FP32 master weights and Adam states **only for its local shard**:

```python
# megatron/core/optimizer/distrib_optimizer.py (DistributedOptimizer internals)

class DistributedOptimizer(MixedPrecisionOptimizer):

    def build_model_and_main_param_groups(self, ...):
        """
        Constructs FP32 master weight shards for the local DP partition.
        Memory per rank: (4 + 4 + 4) bytes × (Phi / D) instead of × Phi.
        """
        for model_param in local_param_shard:
            # FP32 master weight for local shard only
            main_param = model_param.detach().float()
            # Adam first moment (m) and second moment (v) — FP32
            self.optimizer.state[main_param] = {
                'step': torch.tensor(0, dtype=torch.float32),
                'exp_avg': torch.zeros_like(main_param),
                'exp_avg_sq': torch.zeros_like(main_param),
            }
```

---

## 3.5. Global Gradient Clipping Across Shards

Since each rank holds only $\frac{1}{D}$ of the gradients after Reduce-Scatter, the global L2 norm must be computed cooperatively:

```python
# megatron/core/optimizer/clip_grads.py

def clip_grad_norm_fp32(parameters, max_norm, ...):
    """
    Computes global gradient L2 norm across all DP ranks' shards,
    then applies a single unified clip coefficient.
    """
    # Each rank computes its local squared norm
    total_norm = torch.zeros(1, dtype=torch.float32, device='cuda')
    for param in params_with_grad:
        param_norm = param.grad.detach().norm(2)
        total_norm += param_norm ** 2

    # All-Reduce (SUM) to aggregate squared norms across DP group
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=data_parallel_group)
    total_norm = total_norm.sqrt().item()

    # Single clip coefficient applied uniformly to every rank's shard
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in params_with_grad:
            param.grad.detach().mul_(clip_coef)
```

> [!IMPORTANT]
> Gradient clipping **must** use the global norm across all DP shards. Clipping each rank's local shard independently with its local norm produces a different effective clip threshold and breaks training convergence for large models.

---

## 3.6. Weight All-Gather After Optimizer Step

After each rank updates its local FP32 master weight shard, Megatron fires `dist.all_gather_into_tensor` to reconstruct full model parameters on every rank:

```python
# megatron/core/optimizer/distrib_optimizer.py

def _gather_model_params(self, async_op: bool = False):
    """
    All-Gathers updated parameter shards from all DP ranks into full
    model parameter tensors for the next forward pass.
    """
    for model_param, main_param in zip(self.model_params, self.main_params):
        # All-Gather updated 16-bit shard back into full model weight
        dist.all_gather_into_tensor(
            output_tensor=model_param.data,        # Full [Phi] view
            input_tensor=main_param.to(model_param.dtype),  # Local [Phi/D] shard
            group=self.data_parallel_group,
            async_op=async_op,
        )
```

---

## 3.7. Static Memory Breakdown

The 16 bytes/parameter law for full mixed-precision training:

| Component | Standard DDP | ZeRO-2 ($D$ ranks) |
|---|---|---|
| FP16/BF16 Weights | $2\Phi$ bytes | $2\Phi$ bytes |
| FP16/BF16 Gradients | $2\Phi$ bytes | $\frac{2\Phi}{D}$ bytes |
| FP32 Master Weights | $4\Phi$ bytes | $\frac{4\Phi}{D}$ bytes |
| FP32 Adam Momentum | $4\Phi$ bytes | $\frac{4\Phi}{D}$ bytes |
| FP32 Adam Variance | $4\Phi$ bytes | $\frac{4\Phi}{D}$ bytes |
| **Total per GPU** | $16\Phi$ bytes | $2\Phi + \frac{14\Phi}{D}$ bytes |

For a 70B parameter model with $D = 64$ DP ranks:
- **Standard DDP**: $16 \times 70 \times 10^9 / 10^9 = 1{,}120\text{ GB}$ per GPU ❌
- **ZeRO-2 (D=64)**: $\approx (2 + 14/64) \times 70 \approx 155\text{ GB}$ per GPU ✅

---

## 3.8. Common Bugs & Gotchas in Distributed Optimizer

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **FP16 Weight Swamping** | Model weights frozen; loss never decreases | Updating FP16 weights directly without FP32 master copy | Always maintain FP32 master weights; cast down only for forward propagation |
| **Premature Weight All-Gather** | Silent convergence stall or desync across ranks | Ranks execute `dist.all_gather` before local Adam update completes | Ensure `all_gather` is placed strictly after `local_master_weights.addcdiv_()` |
| **Unpadded Buffer Length Error** | `RuntimeError: Tensors must be equal size` | Total parameter count not evenly divisible by DP world size $D$ | Pad flattened parameter buffers with zeros up to `ceil(Phi / D) * D` elements |
| **Grad Clipping Norm Inaccuracy** | Exploding gradients despite clip threshold | Clipping gradient norm locally on rank's shard without cluster-wide All-Reduce | Compute $\|g_{\text{local}}\|^2$, run `dist.all_reduce(SUM)`, take square root for global norm |
| **Double Weight Decay Application**| Model parameters decay to zero too rapidly | Applying weight decay both in optimizer step and via explicit loss regularization | Use decoupled AdamW weight decay only on FP32 master weights |

---

## 3.9. Summary & What's Next

In **[Context Parallelism & MoE](/context-parallelism/)**, we study modern extensions for extreme scale: **Context Parallelism (CP / Ring Attention)** for million-token context windows and **Mixture of Experts (MoE / Expert Parallelism)**.
