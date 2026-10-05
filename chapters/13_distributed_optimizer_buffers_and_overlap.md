# Memory Buffers & Communication Overlap
> **ParamAndGradBuffer Layouts, Async Reduce-Scatter Overlap, and 70B/175B Scaling Tables**

---

## 2.1. The Complete Distributed Optimizer Architecture

The diagram below traces the end-to-end dataflow, state transformations, and communication collective boundaries across a 4-GPU Data Parallel group.

```
                      MEGATRON DISTRIBUTED OPTIMIZER LIFECYCLE (DP = 4)
                      
Phase 1: Forward Pass (Local Compute)
  Every GPU has full FP16 Weights W [Φ]. Forward GEMMs execute at full speed.
  GPU 0: W[0..Φ] ──> Forward GEMMs ──> Loss
  GPU 1: W[0..Φ] ──> Forward GEMMs ──> Loss
  GPU 2: W[0..Φ] ──> Forward GEMMs ──> Loss
  GPU 3: W[0..Φ] ──> Forward GEMMs ──> Loss

Phase 2: Backward Pass & Gradient Reduce-Scatter (Comm-Compute Overlap)
  Gradients g[0..Φ] computed layer by layer.
  As bucket k fills, fire async Reduce-Scatter across DP ranks:
  
  GPU 0 holds unreduced g[0..Φ] ──┐
  GPU 1 holds unreduced g[0..Φ] ──┼── Reduce-Scatter(SUM) ──> GPU 0: g_reduced[ 0 .. Φ/4 ]
  GPU 2 holds unreduced g[0..Φ] ──┼                         ──> GPU 1: g_reduced[ Φ/4 .. 2Φ/4 ]
  GPU 3 holds unreduced g[0..Φ] ──┘                         ──> GPU 2: g_reduced[ 2Φ/4 .. 3Φ/4 ]
                                                            ──> GPU 3: g_reduced[ 3Φ/4 .. Φ ]

Phase 3: Local Optimizer Step (Zero Communication)
  Each GPU updates ONLY its assigned 1/4 slice of FP32 master weights & Adam states:
  GPU 0: updates Master_W[ 0 .. Φ/4 ]       using g_reduced[ 0 .. Φ/4 ]
  GPU 1: updates Master_W[ Φ/4 .. 2Φ/4 ]   using g_reduced[ Φ/4 .. 2Φ/4 ]
  GPU 2: updates Master_W[ 2Φ/4 .. 3Φ/4 ]   using g_reduced[ 2Φ/4 .. 3Φ/4 ]
  GPU 3: updates Master_W[ 3Φ/4 .. Φ ]       using g_reduced[ 3Φ/4 .. Φ ]

Phase 4: Weight All-Gather (Reconstruction)
  Updated FP32 master weight slices cast to FP16 W_slice [Φ/4].
  All-Gather collective reconstructs full FP16 W across all ranks:
  
  GPU 0: W_slice[ 0 .. Φ/4 ]       ──┐
  GPU 1: W_slice[ Φ/4 .. 2Φ/4 ]   ──┼── All-Gather ──> All GPUs: W_new[ 0 .. Φ ]
  GPU 2: W_slice[ 2Φ/4 .. 3Φ/4 ]   ──┼
  GPU 3: W_slice[ 3Φ/4 .. Φ ]       ──┘
```

---

## 2.2. Megatron Core Memory Layout: `ParamAndGradBuffer` & Contiguous Buckets

In naive PyTorch code, each `nn.Parameter` allocates its own isolated memory address. This leads to severe memory fragmentation (caching allocator thrashing) and necessitates thousands of tiny individual communication calls.

Megatron Core solves this by allocating two massive, contiguous memory arenas:
1. **Contiguous Parameter Buffer (`param_data`)**: All layer parameters in a model chunk are packed end-to-end into a single 1D tensor.
2. **Contiguous Gradient Buffer (`grad_data`)**: A mirrored 1D tensor holding gradients for all parameters.

Each individual layer parameter (`weight`, `bias`) is transformed into a **view** into these contiguous buffers!

```
                  Contiguous Memory Arena Layout (ParamAndGradBuffer)
                  
Flat Buffer:  [ Layer 0 QKV Weight | Layer 0 Dense Weight | Layer 1 QKV Weight | ... ]
              ├────────────────────┴──────────────────────┴────────────────────┤
Buckets:      │            Bucket 0 (40 MB)               │  Bucket 1 (40 MB)  │
              └───────────────────────────────────────────┴────────────────────┘
```

### 2.2.1 The `main_grad` Pattern & PyTorch Autograd Hook Mechanism

In standard PyTorch, autograd only writes backward gradients into `param.grad` in the parameter's native data type (e.g., FP16 or BF16). However, in high-precision mixed-precision training, accumulating gradients in 16-bit across multiple microbatches leads to catastrophic underflow and precision loss.

Megatron Core solves this through the **`main_grad` pattern** managed by `megatron.core.distributed.param_and_grad_buffer`:
1. **Contiguous FP32 Memory Slice**: Each parameter is assigned a slice of a contiguous FP32 buffer named `param.main_grad`.
2. **Autograd Hook (`_grad_accumulation_hook`)**: Megatron registers a PyTorch backward hook on each parameter tensor (via `param.register_post_accumulate_grad_hook` or custom autograd leaf hooks):
   ```python
   def _grad_accumulation_hook(param):
       if param.grad is not None:
           # Accumulate the 16-bit autograd gradient into the contiguous FP32 buffer
           param.main_grad.add_(param.grad.to(torch.float32))
           # Free the 16-bit autograd tensor immediately to reclaim VRAM!
           param.grad = None
   ```
3. **Bucket Dispatch**: As these hooks fire in reverse topological order during backpropagation, a bucket tracker monitors the contiguous buffer. When a bucket fills (e.g., reaches 40 MB), an asynchronous `reduce_scatter` launches immediately on a dedicated `comm_stream`, overlapping communication with the backward pass of earlier layers!

---

---

## 2.3. Overlapping Communication with Computation: The Asynchronous Bucket Engine

To achieve state-of-the-art Model Flops Utilization (MFU), Megatron Core does not wait for the entire backward pass to finish before starting `reduce_scatter`.

Instead, it divides the model's parameters into **buckets** (typically 40 MB each):
1. Parameters are registered in reverse topological order (matching the backward pass).
2. As backpropagation calculates gradients for Layer L, Layer L-1, Layer L-2, their gradients are accumulated into Bucket B.
3. The instant Bucket B's bytes exceed the threshold, an asynchronous, non-blocking `dist.reduce_scatter` is launched on a dedicated **Communication CUDA Stream** (`comm_stream`).
4. While the network interface (NIC / InfiniBand HCA) is transmitting Bucket B's gradients across the cluster, the GPU Tensor Cores are actively executing GEMM backprop for Bucket B-1 on the Default Compute Stream!

```
Compute Stream:  [ Backprop Layer 3 ]  [ Backprop Layer 2 ]  [ Backprop Layer 1 ]
                        │                     │                     │
Event Sync:             ▼ Bucket 2 Full       ▼ Bucket 1 Full       ▼ Bucket 0 Full
Comm Stream:            └──[Async RS B2]──────┴──[Async RS B1]──────┴──[Async RS B0]──>
```

### 2.3.1 Megatron-Core DDP Hook Registration Pattern

Megatron's `DistributedDataParallel` registers a post-accumulate gradient hook on every parameter to trigger async bucket flushing during backward — the exact pattern from `megatron/core/distributed/distributed_data_parallel.py`:

```python
# megatron/core/distributed/distributed_data_parallel.py

class DistributedDataParallel(MegatronModule):

    def _register_grad_sync_hooks(self):
        """
        Attaches post-accumulate gradient hooks in reverse parameter order.
        Each hook checks if the current bucket is full and fires async Reduce-Scatter.
        """
        for param in reversed(list(self.module.parameters())):
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_param_hook(param, self.data_parallel_group)
                )

    def _make_param_hook(self, param, dp_group):
        def hook(p):
            # Accumulate into bucket; flush if threshold exceeded
            self.grad_buffer.add_param(p)
            if self.grad_buffer.bucket_is_full():
                self._async_reduce_scatter_bucket(dp_group)
        return hook

    def _async_reduce_scatter_bucket(self, dp_group):
        """
        Fires dist.reduce_scatter_tensor on the dedicated comm_stream,
        overlapping with Tensor Core backward computation on the compute_stream.
        """
        bucket_flat = self.grad_buffer.get_flat_bucket()
        output_shard = self.grad_buffer.get_output_shard()

        # Launch non-blocking Reduce-Scatter on the communication stream
        with torch.cuda.stream(self.comm_stream):
            handle = dist.reduce_scatter_tensor(
                output=output_shard,
                input=bucket_flat,
                op=dist.ReduceOp.SUM,
                group=dp_group,
                async_op=True,
            )
            self.async_handles.append(handle)
        self.grad_buffer.reset_bucket()

    def finish_grad_sync(self):
        """
        Called before optimizer.step(). Flushes remaining tail parameters
        and waits for all async handles to complete.
        """
        self._flush_tail_bucket()
        for handle in self.async_handles:
            handle.wait()
        self.async_handles.clear()
```

---

## 2.4. Memory Comparison Matrix: 70B & 175B Scaling

The table below contrasts memory consumption per GPU across model scales and parallelism configurations.

### 2.4.1 70B Parameter Model (`Phi = 70 * 10^9`) on 64 GPUs (`D = 64`)

| Parallelism Strategy | Weights VRAM | Gradients VRAM | Optimizer VRAM | Total Model State | Feasibility on 80GB H100 |
|---|---|---|---|---|---|
| **Standard DDP (`D=64`)** | 140 GB | 140 GB | 840 GB | **`1,120 GB`** | **OOM (14x over limit)** |
| **ZeRO-1 (`P_os, D=64`)** | 140 GB | 140 GB | `(840 / 64) = 13.1 GB` | **`293.1 GB`** | **OOM (3.6x over limit)** |
| **Megatron DistOpt (`P_{g+os}`)** | 140 GB | `(140 / 64) = 2.2 GB` | `(840 / 64) = 13.1 GB` | **`155.3 GB`** | Needs TP=2 or PP=2 |
| **TP=4 + Megatron DistOpt (`D=16`)**| `(140 / 4) = 35 GB` | `(140 / (4 * 16)) = 2.2 GB` | `(840 / (4 * 16)) = 13.1 GB` | **`50.3 GB`** | **FITS! (`29.7 GB` for Activations)** |

Combining **Tensor Parallelism (`TP = 4`)** with the **Megatron Distributed Optimizer** brings the static model state down to **`50.3 GB`**, leaving nearly **30 GB of high-speed HBM** entirely free for batch activations and long sequence processing!

---

