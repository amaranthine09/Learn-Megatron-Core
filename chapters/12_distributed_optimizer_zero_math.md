# Memory Accounting & The Megatron Distributed Optimizer
> **The 16-Bytes/Param Law, Floating-Point Swamping Proofs, and Native ZeRO-2 Sharding**

> **Reference Papers**:
> - *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (Rajbhandari et al., 2020, [arXiv:1910.02054](https://arxiv.org/abs/1910.02054))
> - *Reducing Activation Recomputation in Large Transformer Models* (Korthikanti et al., 2022, [arXiv:2205.05198](https://arxiv.org/abs/2205.05198))
> - *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (Shoeybi et al., 2019, [arXiv:1909.08053](https://arxiv.org/abs/1909.08053))
> - *Megatron Core Distributed Optimizer Specification* (NVIDIA, 2024)

---

## 1.1. The Physical Cause of the 16 Bytes/Param Law

To design large-scale distributed training systems, you must account for every byte of High Bandwidth Memory (HBM). A modern 70-billion-parameter model cannot be loaded naively onto any single GPU, nor can it be trained with vanilla Data Parallelism across a cluster without understanding the exact mathematical mechanics of floating-point representation and optimizer state dynamics.

In deep learning, memory is divided into two fundamental regimes:
1. **Static Memory (Model States)**: Memory that persists permanently across iterations: Model Parameters, Gradients, and Optimizer States.
2. **Dynamic Memory (Activations & Temporary Buffers)**: Memory allocated during forward propagation to store intermediate tensors for autograd, scratchpad GEMM workspaces, and communication ring buffers.

---

### 1.1.1 The Mathematical Necessity of FP32 Master Weights (Floating-Point Swamping)

Modern Large Language Models are executed on GPU Tensor Cores using 16-bit floating-point formats (**FP16** or **BF16**) to maximize compute throughput (TFLOPS) and minimize memory bandwidth pressure. However, **weights cannot be updated purely in 16-bit precision**.

Let us examine the mathematical proof of why direct 16-bit weight updates suffer from catastrophic numerical truncation, known as **floating-point swamping**.

#### 1.1.1.1 The IEEE 754 Floating-Point Anatomy

| Format | Total Bits | Sign Bits | Exponent Bits | Mantissa (Fraction) Bits | Machine Epsilon (eps_mach) | Dynamic Range |
|---|---|---|---|---|---|---|
| **FP32** (Single Precision) | 32 | 1 | 8 | 23 | `2^-23 ≈ 1.192 * 10^-7` | `≈ 10^-38 to 10^38` |
| **FP16** (Half Precision) | 16 | 1 | 5 | 10 | `2^-10 ≈ 9.765 * 10^-4` | `≈ 6 * 10^-5 to 6.5 * 10^4` |
| **BF16** (Bfloat16) | 16 | 1 | 8 | 7 | `2^-7 ≈ 7.812 * 10^-3` | `≈ 10^-38 to 10^38` |

Machine epsilon eps_mach is defined as the smallest positive number such that:

> `1.0 + eps_mach != 1.0`

In FP16 arithmetic, any value smaller than `2^-11 ≈ 4.88 * 10^-4` added to 1.0 is completely rounded away and destroyed!

#### 1.1.1.2 The Swamping Failure Theorem

Consider a model parameter W whose magnitude is normalized around `W ≈ 1.0`.
During an AdamW optimization step, the update magnitude `Delta W` applied to the parameter is:

> `Delta W = -eta ( (m_hat_t / (sqrt(v_hat_t) + eps)) + lambda W_t )`

Where:
- eta is the learning rate (typically `1 * 10^-4` for LLM pretraining).
- `(m_hat_t / (sqrt(v_hat_t) + eps))` is the normalized Adam step (bounded around order `~ 10^-1` to `10^-3` for stable gradients).
- lambda is the weight decay coefficient (e.g., 0.1).

The expected update magnitude is:

> `|Delta W| ≈ 10^-4 * 10^-3 = 10^-7`

Now, attempt to perform this addition directly in 16-bit precision (FP16 or BF16):

> `W_{t+1} = W_t + Delta W = 1.0 + 10^-7`

To add two floating-point numbers, hardware alignment logic must shift the mantissa of the smaller operand to match the exponent of the larger operand:
1. Exponent of `W_t = 1.0 = 2^0`: Exponent field is biased 0.
2. Exponent of `Delta W ≈ 10^-7 ≈ 2^{-23.25}`: Exponent field is biased -24.
3. Difference in exponents: `Delta e = 0 - (-24) = 24` bits.
4. Mantissa of `Delta W` must be shifted right by **24 bit positions**.
5. In FP16, the mantissa has only **10 bits**. A 24-bit right shift discards all significant bits into the hardware underflow sticky bits.
6. The rounded result in round-to-nearest-even mode is:

> `fl_16(1.0 + 10^-7) = 1.0000000000_2 = 1.0`

> `W_{t+1} == W_t (The weight never updates!)`

If trained purely in 16-bit floating point, the model's parameters freeze in place. The entire pretraining run stalls, resulting in complete gradient stagnation and zero loss convergence.

#### 1.1.1.3 The Solution: The FP32 Master Weight

To solve swamping without losing Tensor Core speed:
- Forward propagation uses **16-bit weights** (W_16).
- Backward propagation computes gradients in **16-bit** (g_16).
- The optimizer maintains an **FP32 Master Weight** (W_32), which has a 23-bit mantissa.
- The update is accumulated into FP32:

> `W_{32, t+1} = W[32, t] + Delta W_32`

- Because FP32 machine epsilon is `1.19 * 10^-7`, updates of magnitude `10^-7` preserve their significant bits and accumulate correctly over hundreds of thousands of micro-steps.
- At the end of the optimizer step, `W_{32, t+1}` is cast down to 16-bit to form `W_{16, t+1}` for the next iteration's forward pass.

---

### 1.1.2 The Complete 16 Bytes/Param Memory Audit

Let Phi denote the total number of trainable model parameters in the network.
For standard mixed-precision training using the AdamW optimizer:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   TOTAL STATIC MEMORY: 16 BYTES PER PARAMETER                    │
├───────────────────────────────┬──────────────────────────────────────────────────┤
│ Model Weights (FP16 / BF16)   │  2 bytes × Φ                                     │
├───────────────────────────────┼──────────────────────────────────────────────────┤
│ Model Gradients (FP16 / BF16) │  2 bytes × Φ                                     │
├───────────────────────────────┼──────────────────────────────────────────────────┤
│ FP32 Master Weights           │  4 bytes × Φ                                     │
├───────────────────────────────┼──────────────────────────────────────────────────┤
│ Adam 1st Moment (m_t, FP32)   │  4 bytes × Φ                                     │
├───────────────────────────────┼──────────────────────────────────────────────────┤
│ Adam 2nd Moment (v_t, FP32)   │  4 bytes × Φ                                     │
└───────────────────────────────┴──────────────────────────────────────────────────┘
```

> `Static Model Memory = (2 + 2 + 4 + 4 + 4) * Phi = 16 * Phi bytes`

#### 1.1.2.1 Concrete Numerical Scaling Across Model Classes

| Model Architecture | Parameters (Phi) | FP16 Weights | FP16 Grads | FP32 Master Weights | FP32 Momentum (m_t) | FP32 Variance (v_t) | Total Static VRAM |
|---|---|---|---|---|---|---|---|
| **Llama-3-8B** | `8.03 * 10^9` | `16.06 GB` | `16.06 GB` | `32.12 GB` | `32.12 GB` | `32.12 GB` | **`128.5 GB`** |
| **Llama-2-70B** | `70.55 * 10^9` | `141.1 GB` | `141.1 GB` | `282.2 GB` | `282.2 GB` | `282.2 GB` | **`1,128.8 GB`** |
| **GPT-3-175B** | `175.0 * 10^9` | `350.0 GB` | `350.0 GB` | `700.0 GB` | `700.0 GB` | `700.0 GB` | **`2,800.0 GB`** |
| **DeepSeek-V3 (671B MoE)** | `671.0 * 10^9` | `1,342 GB` | `1,342 GB` | `2,684 GB` | `2,684 GB` | `2,684 GB` | **`10,736 GB`** |

A single NVIDIA H100 GPU features **80 GB of HBM3 memory**.
For a 70B model (`1,128.8 GB` static state):

> `H100 GPUs Needed for Static Memory Alone = (1,128.8 GB / 80 GB) = 14.11 => 15 H100 GPUs`

*And this is before storing even a single token activation!*

---

### 1.1.3 Underflow Dynamics: FP16 Loss Scaling vs Native BF16

Why does standard FP16 require dynamic loss scaling, whereas BF16 does not?

1. **FP16 Underflow Trap**:
   The minimum positive normal number representable in FP16 is:

> `2^-14 ≈ 6.10 * 10^-5`

   During backpropagation through a 80-layer transformer, activation gradients undergo repeated fractional matrix multiplications. Gradients frequently drop to `10^-6` or `10^-7`. In FP16, these values underflow directly to zero (0.0).
   - **GradScaler Remedy**: PyTorch multiplies the loss by a large factor `S = 2^16 = 65,536` before backpropagation:

> `g_tilde = grad (S * Loss) = S * g`

     This scales the gradients into the center of FP16's representable dynamic range. Before the optimizer step, the gradients are unscaled: `g = g_tilde / S`. If an `inf` or `nan` is detected, the step is skipped and S is halved.

2. **BF16 Architectural Superiority**:
   BF16 truncates the FP32 mantissa from 23 bits down to 7 bits, but **preserves all 8 bits of the FP32 exponent**.
   - Minimum normal BF16 value: `2^-126 ≈ 1.175 * 10^-38`.
   - Maximum normal BF16 value: `2^127 ≈ 3.402 * 10^38`.
   Because BF16 possesses the exact same dynamic exponent range as FP32, gradients can never underflow or overflow under normal training dynamics. **No loss scaler is needed**, eliminating skipped optimizer steps and pipeline sync overhead.

---

## 1.2. Reference Implementation: Megatron-Style Mixed Precision Setup

The following code illustrates how master weights, dynamic casting, gradient unscaling, and optimizer state preservation interact in a production mixed-precision training step.

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class MasterWeightOptimizer:
    """
    Megatron-Core FP32 Master Weight Optimizer mechanism.
    Maintains FP32 master parameters for FP16/BF16 model parameters to prevent
    floating-point swamping during gradient descent.
    """
    def __init__(self, model_params: List[nn.Parameter], lr: float = 1e-4, 
                 betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1):
        self.model_params = [p for p in model_params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # ── Allocate FP32 Master Weights (4 bytes/param) ─────────────────
        self.master_params: List[torch.Tensor] = [
            p.detach().clone().to(torch.float32).requires_grad_(False)
            for p in self.model_params
        ]

        # ── Allocate FP32 Optimizer States (8 bytes/param) ───────────────
        self.exp_avg: List[torch.Tensor] = [
            torch.zeros_like(mp) for mp in self.master_params
        ]
        self.exp_avg_sq: List[torch.Tensor] = [
            torch.zeros_like(mp) for mp in self.master_params
        ]

    @torch.no_grad()
    def step(self):
        """
        Executes the AdamW update directly on FP32 master parameters,
        then copies the updated values back into FP16/BF16 model parameters.
        """
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count

        for p_model, p_master, m, v in zip(
            self.model_params, self.master_params, self.exp_avg, self.exp_avg_sq
        ):
            if p_model.grad is None:
                continue

            # Cast 16-bit gradient to FP32 for numerical stability
            grad_fp32 = p_model.grad.to(torch.float32)

            # 1. Apply Decoupled Weight Decay on FP32 Master Weight
            if self.weight_decay != 0.0:
                p_master.mul_(1.0 - self.lr * self.weight_decay)

            # 2. Update First Moment (m_t = beta1 * m[t-1] + (1 - beta1) * g_t)
            m.mul_(self.beta1).add_(grad_fp32, alpha=1.0 - self.beta1)

            # 3. Update Second Moment (v_t = beta2 * v[t-1] + (1 - beta2) * g_t^2)
            v.mul_(self.beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - self.beta2)

            # 4. Compute Bias-Corrected Step
            denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
            step_size = self.lr / bias_correction1

            # 5. Update FP32 Master Weight without swamping
            p_master.addcdiv_(m, denom, value=-step_size)

            # 6. Copy back to 16-bit model weight for forward propagation
            p_model.copy_(p_master.to(p_model.dtype))

    def zero_grad(self):
        for p in self.model_params:
            p.grad = None
```

### 1.2.1 Mechanism Breakdown: `MasterWeightOptimizer`

1. **Master Parameter Allocation (`self.master_params`)**:
   - Each model parameter (which lives in 16-bit precision, taking 2 bytes) has an independent, unaliased clone allocated in FP32 (taking 4 bytes).
   - `.detach()` ensures that autograd will not track graph operations on the master weights.
   - `requires_grad_(False)` prevents PyTorch from allocating gradient buffers for these tensors.
2. **Momentum & Variance State Allocation (`self.exp_avg`, `self.exp_avg_sq`)**:
   - Allocates the Adam first moment (m_t) and second moment (v_t) directly on the device matching the shape of `p_master`.
   - Each tensor occupies 4 bytes per parameter, summing to `4 + 4 = 8` bytes/param for optimizer momentum and variance.
3. **Gradient FP32 Casting (`grad_fp32`)**:
   - The backward pass produces a 16-bit gradient (`p_model.grad`).
   - We cast this gradient to FP32 before accumulating it into the moments. Doing this in 16-bit would cause catastrophic underflow when computing `g^2` (e.g., `(10^-4)^2 = 10^-8`, which is 0 in FP16).
4. **Decoupled Weight Decay**:
   - Implements decoupled AdamW weight decay: `W_t <- W_t (1 - eta lambda)`.
   - Performed directly in FP32 so that continuous tiny decay steps do not get truncated by machine epsilon.
5. **Fused Moments Update (`addcmul_`)**:
   - In-place point-wise fused linear combinations: `add_` and `addcmul_`.
   - `addcmul_` performs `v = beta_2 v + (1 - beta_2) * (g ⊙ g)` in a single hardware memory roundtrip, avoiding the allocation of an intermediate tensor for `g^2`.
6. **Master Weight Update (`addcdiv_`)**:
   - Performs `p_master = p_master - step_size * (m / denom)` in-place.
   - Because p_master is FP32, updates down to `10^-7` preserve their least significant bits.
7. **Model Weight Synchronization (`copy_`)**:
   - Casts the updated FP32 master weight back down to 16-bit (FP16 or BF16) and copies it directly into the parameter tensor that will be consumed by the Tensor Core GEMMs in the subsequent forward pass.

---

## 1.3. The Redundancy of Standard Data Parallelism (DDP)

In vanilla Distributed Data Parallel (DDP), the entire model state is replicated identically across all D data-parallel worker GPUs:

```
                    Standard Distributed Data Parallel (DDP, D = 4)
                    
GPU 0: [ FP16 W (2Φ) ] [ FP16 Grad (2Φ) ] [ FP32 Master W (4Φ) ] [ FP32 m_t (4Φ) ] [ FP32 v_t (4Φ) ]  = 16Φ
GPU 1: [ FP16 W (2Φ) ] [ FP16 Grad (2Φ) ] [ FP32 Master W (4Φ) ] [ FP32 m_t (4Φ) ] [ FP32 v_t (4Φ) ]  = 16Φ (IDENTICAL!)
GPU 2: [ FP16 W (2Φ) ] [ FP16 Grad (2Φ) ] [ FP32 Master W (4Φ) ] [ FP32 m_t (4Φ) ] [ FP32 v_t (4Φ) ]  = 16Φ (IDENTICAL!)
GPU 3: [ FP16 W (2Φ) ] [ FP16 Grad (2Φ) ] [ FP32 Master W (4Φ) ] [ FP32 m_t (4Φ) ] [ FP32 v_t (4Φ) ]  = 16Φ (IDENTICAL!)
```

### 1.3.1 The Architectural Flaw: Staggering Memory Waste

1. At the conclusion of backward propagation, DDP executes an `All-Reduce(SUM)` across all D GPUs so that every GPU holds the identical globally averaged gradient vector:

> `g_global = (1 / D) sum(r=0 to D-1) g_r`

2. Each GPU then executes the local AdamW optimizer on its local copy of the FP32 master weights.
3. Because the starting weights were identical and the averaged gradients are identical, **every GPU computes the exact same mathematical updates and produces the exact same optimizer state vectors (`m_t, v_t`)**.
4. **The Redundancy Ratio**: Across a cluster of `D = 64` GPUs, **63 out of 64 copies of the optimizer states are 100% redundant duplicates**.
   For a 70B parameter model:

> `Cluster-Wide Optimizer Waste = (D - 1) * 12Phi = 63 * 846.6 GB = 53,335 GB (53.3 Terabytes!)`

---

## 1.4. The ZeRO Partitioning Hierarchy & Megatron DistOpt

Rajbhandari et al. (2020) formalized the elimination of this memory redundancy through **ZeRO (Zero Redundancy Optimizer)**, partitioning model states across data-parallel ranks into three hierarchical stages:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE ZeRO HIERARCHY                                     │
├────────────────────┬─────────────────────────────┬─────────────────────────────────────┤
│ Stage              │ Partitioned States          │ Memory per GPU                      │
├────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
│ Baseline DDP       │ None (All Replicated)       │ 2Φ + 2Φ + 12Φ = 16Φ                 │
│ ZeRO-1 (P_os)      │ Optimizer States (12Φ)      │ 2Φ + 2Φ + (12Φ / D) = 4Φ + (12Φ / D)│
│ ZeRO-2 (P_g+os)    │ Gradients (2Φ) + Opt (12Φ)  │ 2Φ + (2Φ / D) + (12Φ / D) = 2Φ+14Φ/D│
│ ZeRO-3 (P_w+g+os)  │ Weights (2Φ) + All Above    │ (2Φ / D) + (2Φ / D) + (12Φ / D)=16Φ/D│
└────────────────────┴─────────────────────────────┴─────────────────────────────────────┘
```

---

### 1.4.1 Why Megatron-LM Chose ZeRO-2 (DistOpt) Over ZeRO-3 (FSDP)

A critical architectural debate in high-performance machine learning is: *Why does Megatron Core use ZeRO-1 / ZeRO-2 (Distributed Optimizer) rather than ZeRO-3 for 3D parallel model training?*

#### 1.4.1.1 The Communication Bottleneck of ZeRO-3 (FSDP)
In ZeRO-3 (Fully Sharded Data Parallelism):
- Model weights are sharded across D ranks: each GPU holds only `2Phi / D` bytes.
- Before **every single forward layer** can execute its GEMM, an `All-Gather` collective must run across the network to reconstruct the full layer weights W_l. Once the layer finishes, W_l is deleted from memory.
- In the **backward pass**, another `All-Gather` collective must run to reconstruct W_l for the input gradient computation `(d Loss / d X) = (d Loss / d Y) W_l^T`.
- Therefore, ZeRO-3 adds **two full All-Gather communications of the entire model parameter volume** per iteration:

> `Extra Comm Volume_ZeRO-3 = 2 * Phi * (((D-1) / D))`

#### 1.4.1.2 Network Hierarchy Contention in 3D Parallelism
In modern clusters (e.g., 8-GPU nodes connected via NVLink internally and InfiniBand externally):
- Tensor Parallelism (TP) already consumes intra-node NVLink bandwidth (`900 GB/s`) with multiple All-Reduces per transformer block.
- Pipeline Parallelism (PP) passes activation tensors across nodes over InfiniBand.
- If Data Parallelism is also executing ZeRO-3 weight All-Gathers over the exact same inter-node InfiniBand links on every micro-batch layer forward and backward, the network interfaces suffer catastrophic packet contention.
- In contrast, **Megatron Distributed Optimizer (ZeRO-2) requires ZERO communication during the forward pass!**
  - Weights remain resident in local VRAM (2Phi is partitioned by TP and PP, so it easily fits).
  - Backward gradients are reduced via a single `Reduce-Scatter`.
  - Updated weights are broadcast via a single `All-Gather` at the end of the iteration.
  - **Result: Model Flops Utilization (MFU) remains at 50%–60%, whereas ZeRO-3 across slow interconnects collapses to 25%–35%.**

---

## 1.5. The Communication Equivalence Theorem: Zero Overhead Proof

A widespread misconception is that sharding optimizer states and gradients introduces network communication overhead. **We will now prove algebraically that the communication volume of the Megatron Distributed Optimizer is strictly equal to vanilla DDP.**

### 1.5.1 Communication Volume of Standard DDP

In standard DDP, every GPU holds a full gradient tensor of size Phi parameters (2Phi bytes).
At the end of backward propagation, an `All-Reduce(SUM)` is performed across the D data-parallel ranks.

Using the standard Ring-AllReduce algorithm ([Distributed Foundations & Interconnects](/foundations/), Section 5):
1. Ring Reduce-Scatter transfers: `((D-1) / D) * (2Phi)` bytes.
2. Ring All-Gather transfers: `((D-1) / D) * (2Phi)` bytes.
3. Total communication volume per GPU:

> `Comm Volume_DDP = 2 ( ((D-1) / D) ) (2Phi) bytes`

---

### 1.5.2 Communication Volume of Megatron Distributed Optimizer (ZeRO-2)

In the Megatron Distributed Optimizer:
1. **Backward Pass**: Instead of executing an All-Reduce on gradients, ranks execute a **`Reduce-Scatter`**:

> `Comm Volume_RS = ( ((D-1) / D) ) (2Phi) bytes`

   At the end of this step, rank r holds the averaged gradient only for its assigned slice of parameters: `(2Phi / D)` bytes.
2. **Optimizer Step**: Rank r updates its assigned partition of FP32 master weights locally:

> `Communication Volume = 0 bytes`

3. **Weight Synchronization**: Rank r casts its updated partition of master weights back to 16-bit (`2Phi / D` bytes). An **`All-Gather`** is executed across the DP group to reconstruct the full model weights W across all ranks:

> `Comm Volume_AG = ( ((D-1) / D) ) (2Phi) bytes`

Summing the communication phases:

> `Total Comm_DistOpt = Comm Volume_RS + Comm Volume_AG`

> `Total Comm_DistOpt = ( ((D-1) / D) ) (2Phi) + ( ((D-1) / D) ) (2Phi) = 2 ( ((D-1) / D) ) (2Phi) bytes`

> `Comm Volume(Standard DDP) == Comm Volume(Megatron DistOpt)`

> `Delta Network Bandwidth Overhead == 0.00%`

The Distributed Optimizer yields massive VRAM reductions with **zero additional bytes transmitted across the network**.

---

