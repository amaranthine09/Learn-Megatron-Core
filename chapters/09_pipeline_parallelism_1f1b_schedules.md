# Pipeline Parallelism & 1F1B Distributed Schedules
> **Vertical Layer Sharding, GPipe Comparison, 1F1B Steady-State, and Virtual Interleaving**

> **Reference Paper**: *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (Narayanan et al., NVIDIA 2021, [arXiv:2104.04473](https://arxiv.org/abs/2104.04473))

---

## 1.1. Inter-Node Latency Bounds: The Physical Imperative for Vertical Sharding

In Chapters 02 and 03, we analyzed **Tensor Parallelism (TP)** and **Sequence Parallelism (SP)**.
Both require collective communications (`all_reduce`, `reduce_scatter`, `all_gather`) in **every single transformer block**.

### 1.1.1 The NVLink Wall:
- Inside a server (e.g. 8x H100 node), GPUs communicate over **NVLink** at **`900 GB/s`** with `≈ 1 us` latency.
- Across servers, GPUs communicate over **InfiniBand / RoCE** at **`50 GB/s`** (400 Gbps) with `≈ 5 - 10 us` latency.

If you attempt to run Tensor Parallelism across nodes (`TP > 8`), the high-frequency all-reduce calls hit the slower inter-node network, and the GPUs spend over **60% of their time idling waiting for communication!**

**Rule of Thumb in Production:**

> `Tensor Parallel Size (TP) <= Number of GPUs per Node (typically 8)`

To scale a model across hundreds or thousands of GPUs, we must partition the model **vertically across layers** using **Pipeline Parallelism (PP)**, where communication happens only at stage boundaries!

---

## 1.2. Pipeline Partitioning: Vertical Layer Sharding

In Pipeline Parallelism with p pipeline stages, the L layers of a Transformer are distributed sequentially:

> `Layers per Stage = (L / p)`

```
                4-Stage Pipeline Architecture (L = 32 layers)
                
  Stage 0 (Rank 0):   [Embedding Table] + [Layers 0 .. 7]
                           │  (Send activations of Layer 7)
                           ▼  [P2P Inter-Node Network]
  Stage 1 (Rank 1):   [Layers 8 .. 15]
                           │  (Send activations of Layer 15)
                           ▼  [P2P Inter-Node Network]
  Stage 2 (Rank 2):   [Layers 16 .. 23]
                           │  (Send activations of Layer 23)
                           ▼  [P2P Inter-Node Network]
  Stage 3 (Rank 3):   [Layers 24 .. 31] + [LM Head & Loss]
```

### 1.2.1 Communication Advantage:
Between Stage k and Stage `k+1`, **only the activation tensor of the final layer** is transmitted. The intermediate representations within the 8 layers never cross the network!

---

## 1.3. The Pipeline Bubble & Scheduling Paradigms

A naive pipeline that passes an entire batch through Stage 0, then Stage 1, etc., suffers from catastrophic GPU idling: only one stage is active at any time, resulting in near-zero utilization.

To solve this, the batch is split into m smaller **microbatches**.

### 1.3.1 The GPipe Schedule & The Exact Bubble Mathematical Proof
In the GPipe approach (Huang et al., 2019):
1. All m microbatches run their forward pass sequentially through Stages `0 -> p-1`.
2. All m microbatches run their backward pass sequentially through Stages `p-1 -> 0`.

```
           GPipe Schedule (p = 4 stages, m = 8 microbatches)
           
Stage 3:           [F1][F2][F3][F4][F5][F6][F7][F8][B8][B7][B6][B5][B4][B3][B2][B1]
Stage 2:       [F1][F2][F3][F4][F5][F6][F7][F8]        [B8][B7][B6][B5][B4][B3][B2][B1]
Stage 1:   [F1][F2][F3][F4][F5][F6][F7][F8]                [B8][B7][B6][B5][B4][B3][B2][B1]
Stage 0: [F1][F2][F3][F4][F5][F6][F7][F8]                        [B8][B7][B6][B5][B4][B3][B2][B1]
Time ──>  |<- Bubble ->|                                  |<- Bubble ->|
```

#### 1.3.1.1 The Bubble Fraction Mathematical Derivation:
Let:
- p: Number of pipeline stages (physical GPUs or nodes in pipeline).
- m: Number of microbatches in the global batch.
- t_f: Time required for one forward microbatch on one stage.
- t_b: Time required for one backward microbatch on one stage (`t_b ≈ 2 t_f`, since backprop computes both input and weight gradients).

Let us trace the timeline:
1. **Warmup Phase**: Stage 0 starts immediately at `t = 0`. But Stage p-1 cannot start until Microbatch 1 has propagated through all preceding p-1 stages!

> `Warmup Idle Time = (p - 1) * t_f`

2. **Cooldown Phase**: After Stage 0 completes its final forward microbatch, it must wait for backward gradients to travel back from Stage p-1 through all p-1 stages!

> `Cooldown Idle Time = (p - 1) * t_b`

3. **Total Idle Bubble Time Across All Stages**:

> `t_bubble = (p - 1) * (t_f + t_b)`

The **Ideal Execution Time** (if all GPUs were computing with 100% efficiency with zero pipeline delays) is:

> `t_ideal = m * (t_f + t_b)`

The **Total Elapsed Time** of the entire step is:

> `t_total = t_ideal + t_bubble = (m + p - 1) * (t_f + t_b)`

The **Pipeline Bubble Fraction (F_bubble)** is defined as the ratio of idle time to total time:

> `F_bubble = (t_bubble / t_total) = (((p - 1) * (t_f + t_b)) / ((m + p - 1) * (t_f + t_b))) = ((p - 1) / (m + p - 1))`

#### 1.3.1.2 Concrete Numerical Case Study:
Look at what this formula means in practice:
- If `m = p = 8` (8 microbatches on 8 stages):

> `F_bubble = ((8 - 1) / (8 + 8 - 1)) = (7 / 15) ≈ 46.7%`

  Almost **half of your multimillion-dollar cluster is sitting idle**!
- If `m = 4p = 32` (`m = 32, p = 8`):

> `F_bubble = (7 / (32 + 7)) = (7 / 39) ≈ 17.9%`

- If `m = 8p = 64` (`m = 64, p = 8`):

> `F_bubble = (7 / (64 + 7)) = (7 / 71) ≈ 9.86%`

#### 1.3.1.3 The Fatal GPipe Flaw: The Memory Wall
To make the bubble small, you must make `m >> p`.
However, in GPipe, **all m microbatches run their forward pass before a single backward pass executes!**
Stage 0 must hold the activation tensors of all m microbatches in GPU VRAM simultaneously!

> `Peak Activation Memory_GPipe = O(m)`

If `m = 64`, Stage 0 must store 64 microbatches of activations. For a 70B or 405B parameter model, Stage 0 **crashes with Out-Of-Memory (OOM) before the first backward pass even begins!**

---

## 1.4. The Megatron 1F1B (One-Forward-One-Backward) Schedule

To break this memory wall, Megatron-LM v2 introduced the **1F1B (One-Forward-One-Backward) schedule**.

### 1.4.1 The Core Mechanism: Memory Recycling
Instead of running all forwards first, 1F1B enters a **steady-state** where:
> **Each stage strictly alternates between executing ONE backward pass and ONE forward pass.**

Crucially:
- Executing a backward pass **immediately computes gradients and destroys the stored activation tensor** for that microbatch!
- The freed memory is **immediately recycled** by the very next forward pass!

```
                  Megatron 1F1B Schedule (p = 4, m = 8)
                  
Stage 3:           [F1][B1][F2][B2][F3][B3][F4][B4][F5][B5][F6][B6][F7][B7][F8][B8]
Stage 2:       [F1][F2][B1][F3][B2][F4][B3][F5][B4][F6][B5][F7][B6][F8][B7]    [B8]
Stage 1:   [F1][F2][F3][B1][F4][B2][F5][B3][F6][B4][F7][B5][F8][B6]        [B7][B8]
Stage 0: [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4][F8][B5]            [B6][B7][B8]
Time ──>  |<-- Warmup -->|<----------- Steady State 1F1B ---------->|  |<- Cooldown ->|
```

### 1.4.2 The Three Execution Phases:
1. **Warmup Phase**:
   - Each Stage i executes p - i forward passes to fill the pipeline stages downstream.
   - Stage 0 executes p forward passes.
   - Stage p-1 executes 1 forward pass.
2. **Steady-State Phase**:
   - Every stage executes `1 Backward -> 1 Forward`.
   - Memory is at an equilibrium: 1 microbatch freed, 1 microbatch allocated.
3. **Cooldown Phase**:
   - After all m forward passes are completed, stages drain their remaining saved microbatches with purely backward passes.

### 1.4.3 The Mathematical Memory Guarantee: Decoupled from m
At Stage 0, the maximum number of outstanding un-freed microbatch activations is capped at:

> `Peak Activation Memory_1F1B = O(p)`

More precisely, on Stage i, the maximum number of in-flight activations is:

> `In-Flight Activations_Stage i <= p - i`

Stage 0 holds at most p microbatches; Stage p-1 holds at most 1 microbatch!
**This completely breaks the memory barrier**: You can increase m from 32 to `1,000` to shrink the pipeline bubble without consuming a single extra byte of activation VRAM!

---

**Code — 1F1B schedule: layer assignment and real Megatron Core process group API:**

```python
import torch.distributed as dist

from megatron.core import parallel_state

# Megatron Core pipeline parallel process group query:
pp_rank  = parallel_state.get_pipeline_model_parallel_rank()   # 0 .. p-1
pp_size  = parallel_state.get_pipeline_model_parallel_world_size()
is_first = parallel_state.is_pipeline_first_stage()
is_last  = parallel_state.is_pipeline_last_stage()

# Contiguous layer partitioning across pipeline stages:
def get_local_layers(num_total_layers, pp_rank, pp_size):
    """
    Assigns a contiguous slice of transformer layers to each PP stage.
    e.g., 32 layers / 4 stages = 8 layers per stage.
    PP rank 0 gets layers [0..7], rank 1 gets [8..15], etc.
    """
    assert num_total_layers % pp_size == 0
    layers_per_stage = num_total_layers // pp_size
    start = pp_rank * layers_per_stage
    end   = start + layers_per_stage
    return list(range(start, end))

# 1F1B Steady-State Loop (conceptual, no dist calls for clarity):
def run_1f1b_steady_state(microbatches, pp_rank, pp_size, model_chunk):
    """
    In steady state, each rank alternates F then B.
    Key property: each B immediately frees memory from the corresponding F!
    """
    num_microbatches = len(microbatches)
    # Warmup: fill the pipeline (rank 0 does pp_size F-passes first)
    warmup_steps = pp_size - pp_rank - 1
    activations = {}   # microbatch_id -> saved activations (freed after B!)

    for i in range(min(warmup_steps, num_microbatches)):
        activations[i] = model_chunk.forward(microbatches[i])  # accumulate

    # Steady state: 1F then immediately 1B
    for i in range(warmup_steps, num_microbatches):
        activations[i] = model_chunk.forward(microbatches[i])  # forward
        j = i - warmup_steps
        model_chunk.backward(activations.pop(j))  # backward + FREE memory

    # Cooldown: drain remaining activations
    for j in range(warmup_steps - pp_rank, warmup_steps):
        model_chunk.backward(activations.pop(j))
```

### 1.4.4 Mechanism Breakdown: `run_1f1b_steady_state`

1. **Warmup Steps (`warmup_steps = pp_size - pp_rank - 1`)**:
   - In 1F1B, downstream stages cannot execute backward until they receive forward activations.
   - Stage r executes p - r - 1 extra forward passes upfront to prime the pipeline down to Stage p-1.
   - Stage 0 has the longest warmup: it must push enough microbatches into the pipeline so that by the time it finishes warmup, the very first microbatch has completed its forward pass on Stage p-1, computed the loss, and sent its backward gradient back to Stage 0.
2. **Warmup Allocation Phase**:
   - Each forward pass stores its intermediate activations in the dictionary `activations[i]`.
   - On Stage 0, memory climbs up to p microbatches.
3. **The 1F1B Steady-State Equilibrium**:
   - For every new microbatch pushed into forward (`model_chunk.forward(microbatches[i])`), an older microbatch completes its backward pass (`model_chunk.backward(activations.pop(j))`).
   - `activations.pop(j)` **immediately deletes the Python reference to the stashed activation tensor**.
   - As autograd consumes the tensor during backprop, PyTorch's Caching Allocator reclaims the exact HBM memory block, making it available for the next forward microbatch without triggering a new OS `cudaMalloc` call!
   - Result: Steady-state memory consumption remains flat and constant (`O(p)`), completely independent of total microbatches m.
4. **Cooldown Drain Phase**:
   - Once all m forward passes have run, no new activations are created.
   - Ranks iterate through the remaining stashed activations in FIFO order, running pure backward passes until the dictionary is empty.

---

## 1.5. The Interleaved 1F1B Schedule (Megatron v2 Breakthrough)

Even with 1F1B, the bubble fraction is still `F_bubble = ((p - 1) / m)`.
If `p = 8` and `m = 32`, the bubble wastes `≈ 22%` of total compute!

Megatron-LM v2 solved this with **Interleaved 1F1B**:
Instead of assigning a single contiguous chunk of layers to each device, **each physical GPU manages v virtual stages**:

```
Example: 32 Layers, p = 4 Physical Devices, v = 2 Virtual Stages per Device

Physical Device 0 holds:  Virtual Chunk 0 (Layers 0..3)   AND Virtual Chunk 4 (Layers 16..19)
Physical Device 1 holds:  Virtual Chunk 1 (Layers 4..7)   AND Virtual Chunk 5 (Layers 20..23)
Physical Device 2 holds:  Virtual Chunk 2 (Layers 8..11)  AND Virtual Chunk 6 (Layers 24..27)
Physical Device 3 holds:  Virtual Chunk 3 (Layers 12..15) AND Virtual Chunk 7 (Layers 28..31)
```

```
                 Interleaved 1F1B Execution Timeline (p = 4, v = 2)
                 
Physical Stage 3:      [F1.1]      [F2.1]      [F1.2][B1.2][F2.2][B2.2]...[B1.1][B2.1]
Physical Stage 2:   [F1.1]      [F2.1]      [F1.2]      [F2.2][B1.2]...
Physical Stage 1: [F1.1]      [F2.1]      [F1.2]      [F2.2]      [B1.2]...
Physical Stage 0:[F1.1]      [F2.1]      [F1.2]      [F2.2]
Time ─────────>   |<- Bubble/2 ->|

Notation: F_m.v = Microbatch m, Virtual Stage v
```

Now, microbatch 1 travels through Device `0 -> 1 -> 2 -> 3` for Virtual Chunk 1, and immediately loops back to Device `0 -> 1 -> 2 -> 3` for Virtual Chunk 2!

### 1.5.1 The New Bubble Fraction: Exact vs Asymptotic Formulation

In the seminal Megatron-LM v2 paper (*Narayanan et al., 2021*), the exact bubble fraction for Interleaved 1F1B with v virtual stages is derived as:

> `F[bubble, interleaved] = ((p - 1) / (v * m + p - 1))`

#### 1.5.1.1 Asymptotic Approximation:
When the number of microbatches is very large (`v * m >> p`), the `+ (p - 1)` term in the denominator becomes negligible, yielding the commonly quoted asymptotic rule of thumb:

> `F[bubble, interleaved] ≈ (1 / v) * ((p - 1) / m)`

#### 1.5.1.2 Worked Numerical Comparison (`m = 8, p = 8, v = 2`):
Let us evaluate a practical pretraining setup with 8 pipeline stages, 8 microbatches, and 2 virtual stages per GPU:
- **Standard 1F1B (`v = 1`)**:

> `F_bubble = ((8 - 1) / (8 + 8 - 1)) = (7 / 15) ≈ 46.67%`

- **Exact Interleaved 1F1B (`v = 2`)**:

> `F[bubble, interleaved] = ((8 - 1) / (2 * 8 + 8 - 1)) = (7 / 23) ≈ 30.43%`

- **Asymptotic Approximation**:

> `F[bubble, approx] ≈ (1 / 2) * (7 / 8) = 43.75%`

Notice the critical distinction: the exact formula shows that interleaving with `v=2` reduces the idle bubble from **46.67% down to 30.43%**, cutting the absolute idle time substantially more than the naive asymptotic approximation suggests!

> [!TIP]
> **Why This Matters**:
> In large models where microbatch count m cannot be arbitrarily increased due to global batch size training stability limits, increasing v from 1 to 2 drops the idle bubble fraction by over 16 percentage points. On a 1,024-GPU cluster, this single scheduling change reclaims **over 150 GPUs worth of wasted compute throughput** without changing any model weights or hyperparameters!

```
Tradeoff Analysis:
  + Bubble fraction drops significantly (Massive compute utilization increase)
  - Point-to-Point communication increases by factor of v
Because P2P communication between consecutive layers is small and easily overlapped with compute, virtual interleaving is almost always a decisive net throughput win!
```

**Code — Interleaved 1F1B: virtual stage (v-chunk) layer assignment:**

```python
def get_interleaved_local_layers(num_total_layers, pp_rank, pp_size, num_virtual_stages):
    """
    In Interleaved 1F1B (Virtual Pipeline), each physical GPU holds
    `num_virtual_stages` non-contiguous chunks of layers.

    Example: 32 layers, 4 GPUs, v=2 virtual stages
      Chunk size per virtual stage = 32 / (4 * 2) = 4 layers
      GPU 0 holds: layers [0..3] (chunk 0) + layers [16..19] (chunk 4)
      GPU 1 holds: layers [4..7] (chunk 1) + layers [20..23] (chunk 5)
    """
    assert num_total_layers % (pp_size * num_virtual_stages) == 0
    layers_per_vchunk = num_total_layers // (pp_size * num_virtual_stages)

    local_layers = []
    for v in range(num_virtual_stages):
        # Which global chunk index does this virtual stage correspond to?
        global_chunk_idx = v * pp_size + pp_rank
        chunk_start = global_chunk_idx * layers_per_vchunk
        chunk_end   = chunk_start + layers_per_vchunk
        local_layers.extend(range(chunk_start, chunk_end))

    return local_layers

# Example usage:
for gpu_rank in range(4):
    layers = get_interleaved_local_layers(32, gpu_rank, pp_size=4, num_virtual_stages=2)
    print(f"GPU {gpu_rank}: layers {layers}")
# GPU 0: layers [0, 1, 2, 3, 16, 17, 18, 19]
# GPU 1: layers [4, 5, 6, 7, 20, 21, 22, 23]
# GPU 2: layers [8, 9, 10, 11, 24, 25, 26, 27]
# GPU 3: layers [12, 13, 14, 15, 28, 29, 30, 31]

# In Megatron Core the real API:
# from megatron.core import parallel_state
# parallel_state.initialize_model_parallel(
#     tensor_model_parallel_size=tp,
#     pipeline_model_parallel_size=pp,
#     virtual_pipeline_model_parallel_size=v,   # enables Interleaved 1F1B!
# )
# vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
# pp_schedule is then handled by megatron.core.pipeline_parallel.schedules:
#   forward_backward_pipelining_with_interleaving() if vpp_rank else
#   forward_backward_pipelining_without_interleaving()
```

---

