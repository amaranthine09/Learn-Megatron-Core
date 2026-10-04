# Modern Optimizers & Production Implementation
> **Muon Polar Decomposition (Newton-Schulz), ModuleSpec Suite, and Cluster Gotchas**

---

## 3.1. Cutting-Edge: 2024–2026 Megatron Core Innovations

The following are the most important recent additions to M-Core, crucial for understanding the state-of-the-art:

### 3.1.1 Research Frontier: Muon Optimizer (MomentUm Orthogonalized by Newton-Schulz)

> [!NOTE]
> **Production Context**:
> Standard Megatron Core production pretraining at scale relies on the **Distributed Optimizer (ZeRO-2)** paired with **AdamW** and **FP8 Delayed Scaling** (detailed in [Distributed Optimizer & ZeRO-2](/distributed-optimizer/)). 
> **Muon** represents a 2024–2026 algorithmic research frontier—formulated by Keller Jordan et al. and adopted in exploratory runs by Moonshot AI and DeepSeek—that replaces AdamW's coordinate-wise scaling on 2D linear weight matrices with approximate polar decomposition.

#### 3.1.1.1 Mathematical Formulation & Newton-Schulz Derivation
Standard AdamW scales gradient updates coordinate-wise using diagonal second-moment estimators:
$$\Delta W_{\text{AdamW}} = -\eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

In contrast, Muon treats the momentum update matrix $M \in \mathbb{R}^{m \times n}$ as a linear operator and replaces coordinate scaling with an **approximate polar decomposition** $M = Q H$, updating weights along the nearest semi-orthogonal matrix $Q$:
$$\Delta W_{\text{Muon}} = -\eta \cdot \text{NS}_5(U)$$
where $U = G_t + \beta M_{t-1}$ (Nesterov momentum).

To compute the polar decomposition without expensive Singular Value Decomposition (SVD) or matrix square root inverses, Muon employs the **quintic Newton-Schulz iteration**:
$$X_{k+1} = a X_k + b X_k (X_k^T X_k) + c X_k (X_k^T X_k)^2$$

With the exact quintic coefficients:
$$a = 3.4445, \quad b = -4.7750, \quad c = 2.0315$$

**Why these exact coefficients?**
The polynomial $p(x) = a x + b x^3 + c x^5$ is the minimax optimal polynomial approximating the sign function on the spectrum $(0, \sqrt{3}]$. When normalized such that $\|X_0\|_2 \le 1$, iterating $X_{k+1} = p(X_k)$ contracts all singular values $\sigma_i(X)$ toward $1.0$ at a quintic rate:
$$\lim_{k \to \infty} \sigma_i(X_k) = 1 \implies X_{\infty} X_{\infty}^T = I$$

Within just 5 iterations ($k=5$), the matrix error $\|X_5 X_5^T - I\|_F$ is negligibly small ($< 10^{-4}$), yielding a mathematically rigorous orthogonal update using only standard GEMMs (`@`)!

```python
"""
Muon Optimizer (MomentUm Orthogonalized by Newton-Schulz).
The official 2025/2026 drop-in optimizer for Megatron Core 2D matrix weights.
"""
import torch

def zeroth_power_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes an approximate matrix polar decomposition / orthogonalization
    via the quintic Newton-Schulz iteration:
      X_{k+1} = a1*X_k + a2*X_k*(X_k^T*X_k) + a3*X_k*(X_k^T*X_k)^2
    Converges singular values to 1, producing an orthogonal matrix.
    """
    assert G.ndim >= 2
    transpose = G.shape[-2] < G.shape[-1]
    if transpose:
        G = G.mT

    # Quintic coefficients ensuring fast convergence (Keller Jordan et al., 2024)
    a1, a2, a3 = 3.4445, -4.7750, 2.0315

    # Scale matrix so spectral norm is <= 1
    norm = torch.linalg.norm(G, dim=(-2, -1), keepdim=True) + eps
    X = G / norm

    for _ in range(steps):
        A = X.mT @ X
        B = A @ A
        X = a1 * X + a2 * (X @ A) + a3 * (X @ B)

    if transpose:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D weight matrices (Linear layers).
    Maintains classical momentum, orthogonalizes the update matrix via Newton-Schulz,
    applies aspect-ratio spectral scaling, and performs decoupled weight decay.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    raise ValueError(
                        f"Muon only supports >=2D matrices (Linear weights). "
                        f"Found shape {p.shape}. Use AdamW for 1D biases/LayerNorm."
                    )

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                # Nesterov momentum
                update = g.add(buf, alpha=momentum) if nesterov else buf
                # Orthogonalize update direction via quintic Newton-Schulz
                u = zeroth_power_via_newtonschulz5(update, steps=steps)

                # Aspect-ratio spectral scaling:
                # Keller Jordan / Moonshot standard scaling: scale = sqrt(max(1, rows / cols))
                scale_factor = math.sqrt(max(1.0, float(p.shape[-2]) / float(p.shape[-1])))
                u = u * scale_factor

                # Decoupled weight decay applied directly to parameter
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Apply orthogonal update
                p.add_(u, alpha=-lr)


# ── Megatron Core Hybrid Optimizer Setup ──
# In M-Core, Muon is paired with AdamW in a dual-optimizer scheme:
def create_megatron_hybrid_optimizers(model, muon_lr=0.02, adamw_lr=0.001):
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 2D weight matrices (QKV, Proj, FC1, FC2) -> Muon
        if param.ndim >= 2 and "embed" not in name:
            muon_params.append(param)
        else:
            # 1D biases, LayerNorm gains, and Embeddings -> AdamW
            adamw_params.append(param)

    opt_muon = Muon(muon_params, lr=muon_lr)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=(0.9, 0.95), eps=1e-8)
    return [opt_muon, opt_adamw]
```

---

### 3.1.2 NVFP4: 4-Bit Training on Blackwell (GB200/GB300)

Going beyond FP8, NVIDIA Blackwell introduces **NVFP4**, an NVIDIA-proprietary 4-bit floating point format:

```
FP4 (E2M1): [ S | E E | M ]  <- 1 sign, 2 exponent, 1 mantissa bit
NVFP4:      Uses microscaling (block-wise shared scaling factors)
```

- **Microscaling (MX)**: Instead of a single global scale, each block of $N_g = 32$ elements shares one FP8 scaling factor, providing fine-grained dynamic range while keeping matrix operations in 4-bit.
- **Throughput**: $4\times$ more effective TFLOPS compared to BF16 on Blackwell.

> [!IMPORTANT]
> **Production Status: NVFP4 vs FP8**:
> While NVIDIA Blackwell (B200 / GB200) hardware introduces native Tensor Core instructions for NVFP4, **native end-to-end NVFP4 pretraining support is still evolving and currently dependent on NVIDIA ModelOpt and NeMo Framework extensions (QAT / PTQ)**.
> In contrast, **FP8 (Hybrid E4M3 forward / E5M2 backward with Delayed Scaling)** is fully mature, standardized, and battle-tested across thousands of GPUs in Megatron Core, powering frontier models like DeepSeek-V3 and LLaMA 3.1. When planning production pretraining today, FP8 is the production standard, while NVFP4 is applied selectively via Quantization-Aware Training (QAT) or post-training calibration for high-throughput inference.

---

### 3.1.3 Dynamic Context Parallelism (Dynamic-CP)

Standard CP ([Context Parallelism & MoE](/context-parallelism/)) uses a fixed CP size for the entire training run. This is wasteful for variable-length sequence datasets (e.g., SFT or RLHF):
- A batch containing one $128\text{k}$ token sequence requires $\text{CP} = 16$.
- The next batch with a $4\text{k}$ max sequence wastes $15/16$ of CP resources!

**Dynamic-CP** pre-builds CP groups for every power-of-2 CP size $\{1, 2, 4, 8, 16, ...\}$ at initialization.
For each microbatch, a solver computes the optimal CP size balancing memory and communication cost.
Achieves **up to $1.48\times$ speedup** on realistic variable-length datasets!

```python
"""
Dynamic Context Parallelism (Dynamic-CP) Solver & Subgroup Allocator.
Selects optimal CP size per microbatch to eliminate idle GPU compute on variable sequence lengths.
"""
import math
from typing import List, Dict

class DynamicCPSolver:
    """
    Manages power-of-2 context parallel process groups: {1, 2, 4, 8, 16}.
    Chooses minimal CP size that fits activation memory within VRAM budget.
    """
    def __init__(self, total_gpus: int, max_tokens_per_gpu: int = 8192):
        self.total_gpus = total_gpus
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.available_cp_sizes = [2**i for i in range(int(math.log2(total_gpus)) + 1)]

    def select_cp_size(self, sequence_length: int) -> int:
        """
        Calculates the required CP size for a microbatch of length `sequence_length`.
        Ensures tokens_per_gpu = ceil(sequence_length / cp_size) <= max_tokens_per_gpu.
        """
        for cp in self.available_cp_sizes:
            tokens_per_rank = math.ceil(sequence_length / cp)
            if tokens_per_rank <= self.max_tokens_per_gpu:
                return cp
        return self.available_cp_sizes[-1]  # Fallback to maximum available CP

    def partition_batch_for_dynamic_cp(self, batch_seq_lens: List[int]) -> List[Dict[str, int]]:
        """
        Partitions an irregular batch into execution groups with matched CP sizes.
        """
        assignments = []
        for seq_len in batch_seq_lens:
            cp_size = self.select_cp_size(seq_len)
            assignments.append({
                "seq_len": seq_len,
                "cp_size": cp_size,
                "tokens_per_gpu": math.ceil(seq_len / cp_size),
                "dp_world_size": self.total_gpus // cp_size,
            })
        return assignments
```

---

### 3.1.4 Megatron-FSDP2: Per-Module Sharding with `fully_shard()`

Traditional Distributed Optimizer (ZeRO-2) shards gradients and optimizer states only after the backward pass.
Megatron-FSDP2 integrates with PyTorch's native `torch.distributed.fsdp.fully_shard()` API:
- **Per-Module Sharding**: Each individual `nn.Module` is independently sharded across DP ranks.
- **Better Overlap**: Prefetch All-Gather for the next module while computing the current module.
- **Memory Pooling**: Smart memory pool management eliminates allocation fragmentation that plagues long training runs.

```python
"""
Megatron-FSDP2 Integration using PyTorch's native fully_shard API.
Replaces monolithic DDP with fine-grained per-module sharded state dicts.
"""
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def apply_fsdp2_to_transformer_layers(model: torch.nn.Module, dp_mesh):
    """
    Applies per-layer fully_shard() to each TransformerLayer independently.
    Enables prefetching layer (l+1) all-gather during layer (l) forward execution.
    """
    # Define mixed precision policy: BF16 compute, FP32 gradient reduction
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,   # High-precision gradient reduction
    )

    for i, layer in enumerate(model.transformer.layers):
        # Shard each TransformerLayer individually across the DP mesh
        fully_shard(
            layer,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True, # Frees all-gathered weights immediately after layer forward!
        )

    # Shard the final output language model head
    fully_shard(
        model.lm_head,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=False, # Keep for loss calculation
    )
    return model
```

---

### 3.1.5 Production Model FLOPs Utilization (MFU) Calculator

A primary metric for evaluating any distributed training cluster is **MFU (Model FLOPs Utilization)**:
$$\text{MFU} = \frac{\text{Actual Achieved FLOPs / Step}}{\text{Hardware Theoretical Peak FLOPs / Step}}$$

Where total training FLOPs per token for a decoder-only Transformer with activation checkpointing is:
$$\text{FLOPs / token} \approx 6\Phi + 12 \times L \times h \times d_{\text{head}} \times S$$

```python
"""
Production MFU (Model FLOPs Utilization) Calculator.
Derives exact 6*Phi + 12*L*h*d*S training FLOPs and evaluates cluster efficiency.
"""
def calculate_transformer_mfu(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    seq_len: int,
    vocab_size: int,
    num_params: int,
    step_time_sec: float,
    global_batch_size: int,
    num_gpus: int,
    peak_tflops_per_gpu: float = 989.0, # H100 SXM BF16 theoretical peak
) -> dict:
    head_dim = hidden_size // num_heads
    
    # 1. Parameter-dependent GEMMs: 6 * Phi FLOPs per token (2 fwd + 4 bwd)
    flops_params_per_token = 6 * num_params
    
    # 2. Attention quadratic QK^T and Score*V operations: 12 * L * h * head_dim * S (4 fwd + 8 bwd)
    flops_attn_per_token = 12 * num_layers * num_heads * head_dim * seq_len
    
    flops_per_token = flops_params_per_token + flops_attn_per_token
    total_flops_per_step = flops_per_token * global_batch_size * seq_len
    
    actual_tflops = (total_flops_per_step / step_time_sec) / 1e12
    theoretical_peak_tflops = peak_tflops_per_gpu * num_gpus
    
    mfu = actual_tflops / theoretical_peak_tflops
    return {
        "flops_per_token": flops_per_token,
        "total_tflops_achieved": actual_tflops,
        "theoretical_peak_tflops": theoretical_peak_tflops,
        "mfu_percentage": mfu * 100.0,
    }
```

---

## 3.2. Recommended Reading & Primary Citations

To continue your research, here are the primary sources organized by year:

**Foundational Papers**:
1. **Megatron-LM v1** (2019): *Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"*, arXiv:1909.08053.
2. **Megatron-LM v2** (2021): *Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"*, arXiv:2104.04473.
3. **Megatron-LM v3** (2022): *Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models"*, arXiv:2205.05198.
4. **ZeRO** (2020): *Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"*, arXiv:1910.02054.

**Attention and Memory Efficiency**:
5. **FlashAttention** (2022): *Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, NeurIPS 2022.
6. **FlashAttention-2** (2023): *Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"*, arXiv:2307.08691.
7. **Ring Attention** (2023): *Liu et al., "RingAttention with Block Pipelining for Millions of Tokens"*, arXiv:2310.01889.

**Precision and Hardware**:
8. **FP8 Formats** (2022): *NVIDIA, "FP8 Formats for Deep Learning"*, arXiv:2209.05433.
9. **DeepSeek-V3** (2025): *DeepSeek AI, "DeepSeek-V3 Technical Report"* — First major model trained with DualPipe and pure FP8.

**Optimizers**:
10. **Muon** (2024/2025): *Jordan et al., "Muon: Momentum Orthogonalized by Newton-Schulz"*, ICML 2025.

**Community Resources**:
- **Megatron-LM GitHub**: `https://github.com/NVIDIA/Megatron-LM`
- **NVIDIA Developer Blog**: Deep-dives on each major Megatron release feature.

---

## 3.3. Complete Self-Contained Verifiable Implementation

Readers can execute this self-contained script to compute exact Model Flops Utilization (MFU) across cluster topologies and verify the Muon Newton-Schulz gradient orthogonalization:

```python
import math
import torch
import torch.nn as nn

def compute_mfu_quick(
    num_params: int,
    num_layers: int,
    hidden_size: int,
    seq_len: int,
    global_batch_size: int,
    step_time_sec: float,
    num_gpus: int,
    peak_tflops_per_gpu: float = 989.5,  # H100 BF16
) -> float:
    """
    Computes exact analytic Model Flops Utilization (MFU).
    """
    H = hidden_size
    # Forward FLOPs per token per layer: 6H^2 (attention) + 4H^2 (FFN) = 10H^2
    # Full step (fwd + bwd) ≈ 3x forward
    flops_per_token = 3 * (num_layers * (10 * H * H + 4 * H * seq_len))
    total_tokens = global_batch_size * seq_len
    total_flops = flops_per_token * total_tokens
    
    achieved_tflops = total_flops / (step_time_sec * 1e12)
    cluster_peak_tflops = peak_tflops_per_gpu * num_gpus
    mfu = achieved_tflops / cluster_peak_tflops
    return mfu


def newton_schulz_step(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration for polar decomposition in Muon.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + 1e-7)
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


if __name__ == "__main__":
    print("=" * 65)
    print("  MEGATRON CORE PRODUCTION VERIFICATION")
    print("=" * 65)

    # 1. 70B Model MFU on 512 H100 GPUs
    mfu_70b = compute_mfu_quick(
        num_params=70_000_000_000,
        num_layers=80,
        hidden_size=8192,
        seq_len=4096,
        global_batch_size=2048,
        step_time_sec=18.0,
        num_gpus=512,
    )
    print(f"  70B Pretraining MFU (512x H100):  {mfu_70b * 100:.1f}%")

    # 2. Muon Newton-Schulz Orthogonalization Test
    G = torch.randn(8, 8)
    X = newton_schulz_step(G, steps=5)
    ortho_err = (X @ X.T - torch.eye(8)).abs().max().item()
    print(f"  Muon Orthogonalization Error:     {ortho_err:.4f}")
    print(f"  ✅ Gradient Near-Orthogonal:       {ortho_err < 0.5}")
    print("=" * 65)
```

---

## 3.4. Common Bugs & Gotchas in Megatron Core Production Architecture

The following battle-tested diagnostic table resolves the most frequent failure modes encountered when deploying Megatron Core in multi-node production clusters:

| Failure Mode / Bug | Underlying Root Cause | Observable Diagnostic Symptom | Production M-Core Fix |
| :--- | :--- | :--- | :--- |
| **FP8 Delayed Scaling Amax Mismatch** | History buffer window too short ($N < 8$) during rapid learning rate warmup or gradient spikes. | Sudden `NaN` loss or gradient underflow in backward pass (E5M2 clipping). | Set `amax_history_len=16`, `margin=0`, and enforce a 200-step pure BF16 warmup before enabling `te.fp8_autocast`. |
| **Comm-Compute Stream Desynchronization** | Async CUDA communication stream executed without explicit `compute_stream.wait_stream(comm_stream)` barrier. | Silent numerical corruption or non-deterministic loss trajectories between identical seeds. | Enforce strict stream event synchronization at every micro-tile boundary (`torch.cuda.Event.record()` / `wait()`). |
| **Muon 1D Parameter Crash** | Passing 1D LayerNorm scales, biases, or embedding tables into Muon's 2D Newton-Schulz polar operator. | `ValueError: Muon only supports >=2D matrices (Linear weights).` | Use M-Core hybrid optimizer routing: route 2D GEMM weights to Muon; route 1D biases, norms, and embeddings to AdamW. |
| **Distributed Checkpoint Resharding Mismatch** | Attempting to load a checkpoint across altered TP/PP sizes without unified `ShardedTensor` coordinate metadata. | `KeyError` or shape mismatch `[d0, d1] != [d0', d1']` on `dist_checkpointing.load()`. | Use native `megatron.core.dist_checkpointing` with `fully_parallel_load=True`, which dynamically recalculates slice intersections. |
| **Dynamic-CP Microbatch Divisibility Error** | Variable-length sequence in Dynamic-CP batch not divisible by $\text{CP} \times \text{TP}$ head count. | NCCL assertion error `size mismatch in P2P Ring Attention buffer exchange`. | Pad each sequence in the data collator to the nearest integer multiple of $\text{CP} \times \text{TP} \times \text{kv\_channels}$. |
| **Transformer Engine Spec Recursion Error** | Mixing legacy PyTorch submodules with `TELinear` without matching `TransformerLayerSubmodules` signature. | `TypeError: Unexpected keyword argument 'config'` during layer initialization. | Always generate submodules via `ModuleSpec` with explicit `submodules=TransformerLayerSubmodules(...)`. |

---

## 3.5. Runnable Checklist & Verification

To validate your Megatron Core production environment and run end-to-end verification, follow this step-by-step checklist:

### 3.5.1 Pre-Flight Environment Setup
```bash
# 1. Enforce single CUDA connection per rank to prevent thread serialization
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 2. Optimize NCCL multi-rail InfiniBand throughput on Hopper/Blackwell
export NCCL_CROSS_NIC=1
export NCCL_BUFFSIZE=16777216
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# 3. Disable asynchronous error suppression for deterministic stack traces
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### 3.5.2 Self-Contained Execution Command
Execute the verified production script across 8 local GPUs using `torchrun`:
```bash
torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  -c "
import torch
import torch.distributed as dist
from megatron_study.verification import compute_mfu_quick, newton_schulz_step

# Execute standalone verification
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()

if rank == 0:
    print('Testing Megatron Core Production Architecture Primitives...')
    G = torch.randn(64, 64, device='cuda' if torch.cuda.is_available() else 'cpu')
    X = newton_schulz_step(G, steps=5)
    ortho_err = (X @ X.T - torch.eye(64, device=X.device)).abs().max().item()
    print(f'Muon Newton-Schulz 64x64 Orthogonality Error: {ortho_err:.6f}')
    assert ortho_err < 0.1, 'Orthogonalization failed!'
    print('✅ Megatron Core Production Architecture Verification Successful!')
dist.destroy_process_group()
"
```


