# Foundations Implementation & Cluster Verification
> **Runnable Verification Scripts, NCCL Selection, and MFU Arithmetic**

---

## 3.1. Hands-on Runnable Implementation & Line-by-Line Breakdown

Here is a complete, self-contained implementation demonstrating process groups, ring all-reduce verification, and autograd conjugate operators that you can run directly on your Mac CPU (via Gloo) or on a GPU cluster:

```python
"""
foundations_demo.py
Demonstration of PyTorch Distributed Process Groups and Conjugate Autograd Operators.
Works seamlessly on Mac (CPU) with backend="gloo".
"""

import os
import torch
import torch.distributed as dist


class ConjugateOperatorF(torch.autograd.Function):
    """
    Operator f:
    Forward: Pass-through (Identity)
    Backward: All-Reduce Sum
    """
    @staticmethod
    def forward(ctx, x, group=None):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_output, group=ctx.group, op=dist.ReduceOp.SUM)
        return grad_output, None


class ConjugateOperatorG(torch.autograd.Function):
    """
    Operator g:
    Forward: All-Reduce Sum
    Backward: Pass-through (Identity)
    """
    @staticmethod
    def forward(ctx, x, group=None):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, group=group, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def run_demo():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    print(f"[Rank {rank}/{world_size}] Initialized on Gloo CPU Backend.")

    # 1. Test All-Reduce
    t = torch.tensor([float(rank + 1)], dtype=torch.float32)
    print(f"[Rank {rank}] Value before All-Reduce: {t.item()}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] Value after All-Reduce (Sum of 1..{world_size}): {t.item()}")

    # 2. Test Conjugate Operator f
    x = torch.tensor([10.0], requires_grad=True)
    out_f = ConjugateOperatorF.apply(x)
    # Simulate a loss
    loss = (out_f * (rank + 1)).sum()
    loss.backward()
    # Backward should all-reduce grad_output across ranks
    print(f"[Rank {rank}] Gradient on x after Operator f backward: {x.grad.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        run_demo()
    else:
        print("To run with 2 parallel processes on your Mac:")
        print("  torchrun --nproc_per_node=2 foundations_demo.py")
```

### 3.1.1 Deep Line-by-Line Breakdown of the Autograd Mechanics:

1. **`class ConjugateOperatorF(torch.autograd.Function)`**:
   - In PyTorch, an `nn.Module` forward pass records operations dynamically in a directed acyclic graph (DAG) of `Node` objects.
   - When we inherit from `torch.autograd.Function`, we are inserting a **custom C++ Autograd Node** into this graph.
   
2. **`@staticmethod def forward(ctx, x, group=None)`**:
   - `ctx` is the context object. Any tensor or metadata saved via `ctx.save_for_backward()` or `ctx.group = group` is preserved until the backward pass reaches this node.
   - In `ConjugateOperatorF`, the forward pass is an **identity**: `return x`. The tensor passes through without modification. No communication is executed!

3. **`@staticmethod def backward(ctx, grad_output)`**:
   - During backpropagation, the upstream gradient $\frac{\partial L}{\partial y}$ arrives as `grad_output`.
   - Because $N$ ranks independently computed downstream gradients, the total gradient with respect to the input $x$ is:
     $$\frac{\partial L}{\partial x} = \sum_{i=1}^N \frac{\partial L}{\partial y_i}$$
   - The line `dist.all_reduce(grad_output, group=ctx.group, op=dist.ReduceOp.SUM)` performs an in-place All-Reduce sum! Every rank now receives the true, global gradient $\frac{\partial L}{\partial x}$.
   - We return `grad_output, None` because `forward` took two arguments (`x` and `group`). PyTorch autograd requires returning a gradient for every input argument in `forward`. Since `group` is not a tensor, its gradient is `None`.

4. **`ConjugateOperatorG`**:
   - Performs the exact mathematical conjugate inversion:
   - **Forward**: Combines partial sums from Row-Parallel linear layers via `dist.all_reduce(x, op=dist.ReduceOp.SUM)`.
   - **Backward**: Every rank received the exact same output $y$ in the forward pass. Therefore, the incoming gradient $\frac{\partial L}{\partial y}$ is already identical across all ranks! No backward communication is required: `return grad_output, None`.

---

## 3.2. NCCL Algorithm Selection: Ring vs. Tree vs. NVLS

One of the most common misconceptions is that NCCL always uses Ring All-Reduce. In reality, NCCL **dynamically selects** the best algorithm per call based on message size, hardware topology, and cluster size.

### 3.2.1 The Three Core Algorithms:

#### 3.2.1.1 Algorithm A: Ring All-Reduce (Bandwidth-Optimal)
Best for **large tensors** (gradient buckets, embedding tables):
- GPUs form a logical ring; each sends & receives from only its neighbours.
- Communication volume: $2 \left(\frac{N-1}{N}\right) S$ bytes per rank.
- **Latency scales linearly** with $N$: $\mathcal{O}(N)$ hops.
- ✅ Ideal when $N$ is small and $S$ is large (typical for DP gradient sync).

#### 3.2.1.2 Algorithm B: Double Binary Tree (Latency-Optimal)
Best for **small tensors** or **very large** $N$:
- Two complementary binary trees; each rank is a non-leaf in one tree and leaf in the other.
- **Latency scales logarithmically**: $\mathcal{O}(2 \log_2 N)$ hops.
- ❌ Slightly lower bandwidth utilization than Ring for large messages.
- ✅ Ideal when $N$ is huge (hundreds of GPUs) or tensor is a small scalar/control message.

#### 3.2.1.3 Algorithm C: NVLS (NVLink SHARP — Hopper+ Exclusive)
- **Reductions happen physically inside the NVSwitch fabric!** No data ever traverses individual NVLink lanes redundantly.
- Effective bandwidth approaches the **full crossbar bandwidth**, not the per-link bandwidth.
- Only available on NVIDIA Hopper (H100) and later architectures within a single NVSwitch domain.

### 3.2.2 NCCL Decision Logic (Per Call):

```
For every dist.all_reduce(tensor) call:

  Message Size?
  ├── Large (> ~512 KB / link) → Ring All-Reduce (Bandwidth-optimal)
  └── Small (< ~512 KB) → Double Binary Tree (Latency-optimal)
              │
              └── Are we intra-node on Hopper/Blackwell?
                  └── Yes → NVLS (NVSwitch SHARP Reduction)
```

> [!TIP]
> You can override NCCL's algorithm selection with environment variables for experimentation:
> ```bash
> NCCL_ALGO=Ring torchrun ...      # Force Ring
> NCCL_ALGO=Tree torchrun ...      # Force Tree
> NCCL_PROTO=Simple torchrun ...   # Disable LL/LL128 protocols
> ```

---

## 3.3. Model FLOPs Utilization (MFU) & Hardware FLOPs Utilization (HFU)

When you read a Megatron performance paper claiming "$52\%$ MFU on 1024 H100s", what exactly does that mean?

**MFU and HFU are the standard metrics** for measuring how efficiently you are using expensive GPU hardware.

### 3.3.1 Definitions:

$$\text{MFU} = \frac{\text{Analytic FLOPs per Step}}{\text{GPU Peak FLOPs} \times \text{Step Time} \times \text{Number of GPUs}}$$

$$\text{HFU} = \frac{\text{Actual FLOPs executed (incl. recomputation)}}{\text{GPU Peak FLOPs} \times \text{Step Time} \times \text{Number of GPUs}}$$

**The key distinction**:
- **MFU** measures how efficiently the cluster trains the model (excludes activation checkpointing overhead).
- **HFU** measures how efficiently the hardware is actually utilized (includes recomputation overhead).

### 3.3.2 Computing Analytic FLOPs per Token:

For a dense transformer, the dominant cost is matrix multiplications. The standard approximation:
$$\text{FLOPs per Token} \approx 6 \Phi + 12 \times L \times h \times d_{head} \times S$$

Where:
- $\Phi$ = total model parameters
- $L$ = number of layers
- $h$ = number of attention heads
- $d_{head}$ = head dimension ($H / h$)
- $S$ = sequence length
- The $6\Phi$ term covers: forward pass $\approx 2\Phi$ + backward pass $\approx 4\Phi$ (backward is $2\times$ forward cost because it computes both $\partial L / \partial W$ and $\partial L / \partial X$).
- The $12 L h d_{head} S$ term covers the $O(S^2)$ attention computation.

### 3.3.3 Practical MFU Benchmarks (Reference):

| Hardware | Precision | Good MFU | Excellent MFU |
|---|---|---|---|
| A100 80GB | BF16 | 35–45% | >50% |
| H100 SXM | BF16 | 42–52% | >55% |
| H100 SXM | FP8 | 45–58% | >65% |

> [!NOTE]
> The ~$45\%$ gap from 100% is not wasted computation — it is the overhead of: network communication (TP all-reduces, DP reduce-scatter), pipeline bubbles, kernel launch overhead, and CUDA stream synchronization. Megatron's comm-compute overlap closes this gap significantly.

---

## 3.4. Common Bugs & Gotchas

Here are the most frequently encountered bugs when writing distributed PyTorch code for the first time:

| Bug | Symptom | Root Cause | Fix |
|---|---|---|---|
| **Non-Contiguous Tensor** | `RuntimeError: Tensor must be contiguous` | Transposed or sliced tensor passed to collective | Add `.contiguous()` before collective call |
| **Deadlock** | Job hangs forever | One rank called a collective the other didn't | Ensure all ranks in a group call the same collective in the same order |
| **NCCL Timeout** | `NCCL Watchdog: Timeout` | One rank crashes or diverges mid-training | Check all ranks are alive; reduce `NCCL_TIMEOUT` to catch earlier |
| **Gradient Leakage (TP)** | Loss diverges after a few steps | Bias added `N` times in RowParallelLinear | Add bias after All-Reduce, not inside the partitioned GEMM |
| **Rank 0 Bottleneck** | Very slow All-Reduce | Using `dist.reduce()` instead of `dist.all_reduce()` | `reduce()` sends everything to Rank 0; use `all_reduce()` for training |
| **Stale Grads (DP)** | NaN gradients after resume | Optimizer step ran before gradient reduction completed | Ensure `dist.barrier()` or `req.wait()` is called before `optimizer.step()` |

---

## 3.5. Summary & What's Next

In this book, we established:
1. The hardware hierarchy and why communication cost dominates scaling decisions.
2. The mathematics of Ring All-Reduce: transfer volume is strictly bounded to $2 \left(\frac{N-1}{N}\right) S$ bytes.
3. How `reduce_scatter` and `all_gather` compose the fundamental building blocks of modern distributed training.
4. The conjugate relationship between forward and backward autograd passes.

---

## 3.6. Runnable Checklist & Verification

To verify the foundational conjugate operator pair ($f$ and $g$) on your local machine using PyTorch's CPU backend:

```bash
# Verify process group initialization and autograd conjugate mechanics
torchrun --nproc_per_node=2 demo_megatron.py
```

**Environment Variables Checklist**:
- `MASTER_ADDR=127.0.0.1`: Localhost IP for coordination.
- `MASTER_PORT=29500`: Open TCP port for the Gloo/NCCL rendezvous.
- `OMP_NUM_THREADS=1`: Prevents CPU core thrashing when multiple distributed workers share a single machine.

In **[1D Tensor Parallelism](/tensor-parallelism/)**, we will build directly upon these foundations to construct the complete theory, mathematical proofs, and implementation of **Megatron-LM 1D Tensor Parallelism**.
