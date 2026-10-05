# Foundations: NCCL Algorithm Selection & MFU Arithmetic
> **NCCL Algorithm Decision Logic, MFU/HFU Definitions, and Common Distributed Bugs**

---

## 3.1. NCCL Algorithm Selection: Ring vs. Tree vs. NVLS

One of the most common misconceptions is that NCCL always uses Ring All-Reduce. In reality, NCCL **dynamically selects** the best algorithm per call based on message size, hardware topology, and cluster size.

### 3.1.1 The Three Core Algorithms

#### 3.1.1.1 Algorithm A: Ring All-Reduce (Bandwidth-Optimal)
Best for **large tensors** (gradient buckets, embedding tables):
- GPUs form a logical ring; each sends & receives from only its neighbours.
- Communication volume: `2 (((N-1) / N)) S` bytes per rank.
- **Latency scales linearly** with N: `O(N)` hops.
- ✅ Ideal when N is small and S is large (typical for DP gradient sync).

#### 3.1.1.2 Algorithm B: Double Binary Tree (Latency-Optimal)
Best for **small tensors** or **very large** N:
- Two complementary binary trees; each rank is a non-leaf in one tree and leaf in the other.
- **Latency scales logarithmically**: `O(2 \log_2 N)` hops.
- ❌ Slightly lower bandwidth utilization than Ring for large messages.
- ✅ Ideal when N is huge (hundreds of GPUs) or tensor is a small scalar/control message.

#### 3.1.1.3 Algorithm C: NVLS (NVLink SHARP — Hopper+ Exclusive)
- **Reductions happen physically inside the NVSwitch fabric!** No data ever traverses individual NVLink lanes redundantly.
- Effective bandwidth approaches the **full crossbar bandwidth**, not the per-link bandwidth.
- Only available on NVIDIA Hopper (H100) and later architectures within a single NVSwitch domain.

### 3.1.2 NCCL Decision Logic (Per Call)

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

## 3.2. Model FLOPs Utilization (MFU) & Hardware FLOPs Utilization (HFU)

When you read a Megatron performance paper claiming "52% MFU on 1024 H100s", what exactly does that mean?

**MFU and HFU are the standard metrics** for measuring how efficiently you are using expensive GPU hardware.

### 3.2.1 Definitions

```text
MFU = (Analytic FLOPs per Step / (GPU Peak FLOPs * Step Time * Number of GPUs))
```

```text
HFU = (Actual FLOPs executed (incl. recomputation) / (GPU Peak FLOPs * Step Time * Number of GPUs))
```

**The key distinction**:
- **MFU** measures how efficiently the cluster trains the model (excludes activation checkpointing overhead).
- **HFU** measures how efficiently the hardware is actually utilized (includes recomputation overhead).

### 3.2.2 Computing Analytic FLOPs per Token

For a dense transformer, the dominant cost is matrix multiplications. The standard approximation:
```text
FLOPs per Token ≈ 6 Phi + 12 * L * h * d_head * S
```

Where:
- Phi = total model parameters
- L = number of layers
- h = number of attention heads
- d_head = head dimension (`H / h`)
- S = sequence length
- The 6Phi term covers: forward pass `≈ 2Phi` + backward pass `≈ 4Phi` (backward is `2 *` forward cost because it computes both `d L / d W` and `d L / d X`).
- The `12 L h d_head S` term covers the `O(S^2)` attention computation.

### 3.2.3 Practical MFU Benchmarks (Reference)

| Hardware | Precision | Good MFU | Excellent MFU |
|---|---|---|---|
| A100 80GB | BF16 | 35–45% | >50% |
| H100 SXM | BF16 | 42–52% | >55% |
| H100 SXM | FP8 | 45–58% | >65% |

> [!NOTE]
> The ~45% gap from 100% is not wasted computation — it is the overhead of: network communication (TP all-reduces, DP reduce-scatter), pipeline bubbles, kernel launch overhead, and CUDA stream synchronization. Megatron's comm-compute overlap closes this gap significantly.

---

## 3.3. Conjugate Autograd Operators: f and g

Megatron-Core's tensor parallelism is built on two conjugate `torch.autograd.Function` operators. These are not simulation — they are the exact operators used in `megatron.core.tensor_parallel.mappings`:

```python
# megatron/core/tensor_parallel/mappings.py

class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    Operator f:
    Forward: Identity (pass-through, no communication).
    Backward: All-Reduce SUM across TP ranks.
    """
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)   # dist.all_reduce across TP group


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    Operator g:
    Forward: All-Reduce SUM across TP ranks.
    Backward: Identity (pass-through, no communication).
    """
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)        # dist.all_reduce across TP group

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
```

> [!IMPORTANT]
> The conjugate pairing guarantees that the combined forward+backward communication volume of a Column→Row parallel layer pair is **identical** to a single standard All-Reduce. There is zero communication overhead introduced by tensor parallelism.

---

## 3.4. Common Bugs & Gotchas

Here are the most frequently encountered bugs when writing distributed PyTorch code for the first time:

| Bug | Symptom | Root Cause | Fix |
|---|---|---|---|
| **Non-Contiguous Tensor** | `RuntimeError: Tensor must be contiguous` | Transposed or sliced tensor passed to collective | Add `.contiguous()` before collective call |
| **Deadlock** | Job hangs forever | One rank called a collective the other didn't | Ensure all ranks in a group call the same collective in the same order |
| **NCCL Timeout** | `NCCL Watchdog: Timeout` | One rank crashes or diverges mid-training | Check all ranks are alive; reduce `NCCL_TIMEOUT` to catch earlier |
| **Gradient Leakage (TP)** | Loss diverges after a few steps | Bias added N times in RowParallelLinear | Add bias after All-Reduce, not inside the partitioned GEMM |
| **Rank 0 Bottleneck** | Very slow All-Reduce | Using `dist.reduce()` instead of `dist.all_reduce()` | `reduce()` sends everything to Rank 0; use `all_reduce()` for training |
| **Stale Grads (DP)** | NaN gradients after resume | Optimizer step ran before gradient reduction completed | Ensure `dist.barrier()` or `req.wait()` is called before `optimizer.step()` |

---

## 3.5. Summary & What's Next

In this book, we established:
1. The hardware hierarchy and why communication cost dominates scaling decisions.
2. The mathematics of Ring All-Reduce: transfer volume is strictly bounded to `2 (((N-1) / N)) S` bytes.
3. How `reduce_scatter` and `all_gather` compose the fundamental building blocks of modern distributed training.
4. The conjugate relationship between forward and backward autograd passes.

In **[1D Tensor Parallelism](/tensor-parallelism/)**, we will build directly upon these foundations to construct the complete theory, mathematical proofs, and implementation of **Megatron-LM 1D Tensor Parallelism**.
