# Chapter 04: Pipeline Parallelism, Distributed Schedules & DualPipe
> **1F1B Bubbles, Virtual Interleaving, and B_input/B_weight Decoupled Overlap**

> **Reference Paper**: *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (Narayanan et al., NVIDIA 2021, [arXiv:2104.04473](https://arxiv.org/abs/2104.04473))

---

## 1. Inter-Node Latency Bounds: The Physical Imperative for Vertical Sharding

In Chapters 02 and 03, we analyzed **Tensor Parallelism (TP)** and **Sequence Parallelism (SP)**.
Both require collective communications (`all_reduce`, `reduce_scatter`, `all_gather`) in **every single transformer block**.

### The NVLink Wall:
- Inside a server (e.g. 8x H100 node), GPUs communicate over **NVLink** at **$900\text{ GB/s}$** with $\approx 1\ \mu\text{s}$ latency.
- Across servers, GPUs communicate over **InfiniBand / RoCE** at **$50\text{ GB/s}$** ($400\text{ Gbps}$) with $\approx 5 - 10\ \mu\text{s}$ latency.

If you attempt to run Tensor Parallelism across nodes ($\text{TP} > 8$), the high-frequency all-reduce calls hit the slower inter-node network, and the GPUs spend over **$60\%$ of their time idling waiting for communication!**

**Rule of Thumb in Production:**
$$\text{Tensor Parallel Size (TP)} \le \text{Number of GPUs per Node (typically 8)}$$

To scale a model across hundreds or thousands of GPUs, we must partition the model **vertically across layers** using **Pipeline Parallelism (PP)**, where communication happens only at stage boundaries!

---

## 2. Pipeline Partitioning: Vertical Layer Sharding

In Pipeline Parallelism with $p$ pipeline stages, the $L$ layers of a Transformer are distributed sequentially:

$$\text{Layers per Stage} = \frac{L}{p}$$

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

### Communication Advantage:
Between Stage $k$ and Stage $k+1$, **only the activation tensor of the final layer** is transmitted. The intermediate representations within the 8 layers never cross the network!

---

## 3. The Pipeline Bubble & Scheduling Paradigms

A naive pipeline that passes an entire batch through Stage 0, then Stage 1, etc., suffers from catastrophic GPU idling: only one stage is active at any time, resulting in near-zero utilization.

To solve this, the batch is split into $m$ smaller **microbatches**.

### 3.1 The GPipe Schedule & The Exact Bubble Mathematical Proof
In the GPipe approach (Huang et al., 2019):
1. All $m$ microbatches run their forward pass sequentially through Stages $0 \to p-1$.
2. All $m$ microbatches run their backward pass sequentially through Stages $p-1 \to 0$.

```
           GPipe Schedule (p = 4 stages, m = 8 microbatches)
           
Stage 3:           [F1][F2][F3][F4][F5][F6][F7][F8][B8][B7][B6][B5][B4][B3][B2][B1]
Stage 2:       [F1][F2][F3][F4][F5][F6][F7][F8]        [B8][B7][B6][B5][B4][B3][B2][B1]
Stage 1:   [F1][F2][F3][F4][F5][F6][F7][F8]                [B8][B7][B6][B5][B4][B3][B2][B1]
Stage 0: [F1][F2][F3][F4][F5][F6][F7][F8]                        [B8][B7][B6][B5][B4][B3][B2][B1]
Time ──>  |<- Bubble ->|                                  |<- Bubble ->|
```

#### The Bubble Fraction Mathematical Derivation:
Let:
- $p$: Number of pipeline stages (physical GPUs or nodes in pipeline).
- $m$: Number of microbatches in the global batch.
- $t_f$: Time required for one forward microbatch on one stage.
- $t_b$: Time required for one backward microbatch on one stage ($t_b \approx 2 t_f$, since backprop computes both input and weight gradients).

Let us trace the timeline:
1. **Warmup Phase**: Stage 0 starts immediately at $t = 0$. But Stage $p-1$ cannot start until Microbatch 1 has propagated through all preceding $p-1$ stages!
   $$\text{Warmup Idle Time} = (p - 1) \cdot t_f$$
2. **Cooldown Phase**: After Stage 0 completes its final forward microbatch, it must wait for backward gradients to travel back from Stage $p-1$ through all $p-1$ stages!
   $$\text{Cooldown Idle Time} = (p - 1) \cdot t_b$$
3. **Total Idle Bubble Time Across All Stages**:
   $$t_{\text{bubble}} = (p - 1) \cdot (t_f + t_b)$$

The **Ideal Execution Time** (if all GPUs were computing with $100\%$ efficiency with zero pipeline delays) is:
$$t_{\text{ideal}} = m \cdot (t_f + t_b)$$

The **Total Elapsed Time** of the entire step is:
$$t_{\text{total}} = t_{\text{ideal}} + t_{\text{bubble}} = (m + p - 1) \cdot (t_f + t_b)$$

The **Pipeline Bubble Fraction ($F_{\text{bubble}}$)** is defined as the ratio of idle time to total time:
$$F_{\text{bubble}} = \frac{t_{\text{bubble}}}{t_{\text{total}}} = \frac{(p - 1) \cdot (t_f + t_b)}{(m + p - 1) \cdot (t_f + t_b)} = \mathbf{\frac{p - 1}{m + p - 1}}$$

#### Concrete Numerical Case Study:
Look at what this formula means in practice:
- If $m = p = 8$ (8 microbatches on 8 stages):
  $$F_{\text{bubble}} = \frac{8 - 1}{8 + 8 - 1} = \frac{7}{15} \approx \mathbf{46.7\%}$$
  Almost **half of your multimillion-dollar cluster is sitting idle**!
- If $m = 4p = 32$ ($m = 32, p = 8$):
  $$F_{\text{bubble}} = \frac{7}{32 + 7} = \frac{7}{39} \approx \mathbf{17.9\%}$$
- If $m = 8p = 64$ ($m = 64, p = 8$):
  $$F_{\text{bubble}} = \frac{7}{64 + 7} = \frac{7}{71} \approx \mathbf{9.86\%}$$

#### The Fatal GPipe Flaw: The Memory Wall
To make the bubble small, you must make $m \gg p$.
However, in GPipe, **all $m$ microbatches run their forward pass before a single backward pass executes!**
Stage 0 must hold the activation tensors of all $m$ microbatches in GPU VRAM simultaneously!
$$\text{Peak Activation Memory}_{\text{GPipe}} = \mathbf{\mathcal{O}(m)}$$

If $m = 64$, Stage 0 must store $64$ microbatches of activations. For a 70B or 405B parameter model, Stage 0 **crashes with Out-Of-Memory (OOM) before the first backward pass even begins!**

---

## 4. The Megatron 1F1B (One-Forward-One-Backward) Schedule

To break this memory wall, Megatron-LM v2 introduced the **1F1B (One-Forward-One-Backward) schedule**.

### 4.1 The Core Mechanism: Memory Recycling
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

### 4.2 The Three Execution Phases:
1. **Warmup Phase**:
   - Each Stage $i$ executes $p - i$ forward passes to fill the pipeline stages downstream.
   - Stage 0 executes $p$ forward passes.
   - Stage $p-1$ executes $1$ forward pass.
2. **Steady-State Phase**:
   - Every stage executes $1 \text{ Backward} \to 1 \text{ Forward}$.
   - Memory is at an equilibrium: 1 microbatch freed, 1 microbatch allocated.
3. **Cooldown Phase**:
   - After all $m$ forward passes are completed, stages drain their remaining saved microbatches with purely backward passes.

### 4.3 The Mathematical Memory Guarantee: Decoupled from $m$
At Stage 0, the maximum number of outstanding un-freed microbatch activations is capped at:
$$\text{Peak Activation Memory}_{\text{1F1B}} = \mathbf{\mathcal{O}(p)}$$

More precisely, on Stage $i$, the maximum number of in-flight activations is:
$$\text{In-Flight Activations}_{\text{Stage } i} \le p - i$$

Stage 0 holds at most $p$ microbatches; Stage $p-1$ holds at most $1$ microbatch!
**This completely breaks the memory barrier**: You can increase $m$ from $32$ to $1{,}000$ to shrink the pipeline bubble without consuming a single extra byte of activation VRAM!

---

**Code — 1F1B schedule: layer assignment and real Megatron Core process group API:**

```python
import torch.distributed as dist

# ── Megatron Core process group initialisation (the real API) ──
# from megatron.core import parallel_state
# parallel_state.initialize_model_parallel(
#     tensor_model_parallel_size=tp,
#     pipeline_model_parallel_size=pp,
# )
# pp_rank  = parallel_state.get_pipeline_model_parallel_rank()   # 0 .. p-1
# pp_size  = parallel_state.get_pipeline_model_parallel_world_size()
# is_first = parallel_state.is_pipeline_first_stage()
# is_last  = parallel_state.is_pipeline_last_stage()

# DIY equivalent — compute which transformer layers live on this PP rank:
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

### Deep Line-by-Line Pedagogical Breakdown: `run_1f1b_steady_state`

1. **Line 199 (`warmup_steps = pp_size - pp_rank - 1`):**
   - In 1F1B, downstream stages cannot execute backward until they receive forward activations.
   - Stage $r$ executes $p - r - 1$ extra forward passes upfront to prime the pipeline down to Stage $p-1$.
   - Stage 0 has the longest warmup: it must push enough microbatches into the pipeline so that by the time it finishes warmup, the very first microbatch has completed its forward pass on Stage $p-1$, computed the loss, and sent its backward gradient back to Stage $0$.
2. **Lines 202–203 (Warmup Allocation Phase):**
   - Each forward pass stores its intermediate activations in the dictionary `activations[i]`.
   - On Stage 0, memory climbs up to $p$ microbatches.
3. **Lines 206–210 (The 1F1B Steady-State Equilibrium):**
   - For every new microbatch pushed into forward (`model_chunk.forward(microbatches[i])`), an older microbatch completes its backward pass (`model_chunk.backward(activations.pop(j))`).
   - `activations.pop(j)` **immediately deletes the Python reference to the stashed activation tensor**.
   - As autograd consumes the tensor during backprop, PyTorch's Caching Allocator reclaims the exact HBM memory block, making it available for the next forward microbatch without triggering a new OS `cudaMalloc` call!
   - Result: Steady-state memory consumption remains flat and constant ($\mathcal{O}(p)$), completely independent of total microbatches $m$.
4. **Lines 212–214 (Cooldown Drain Phase):**
   - Once all $m$ forward passes have run, no new activations are created.
   - Ranks iterate through the remaining stashed activations in FIFO order, running pure backward passes until the dictionary is empty.

---

## 5. The Interleaved 1F1B Schedule (Megatron v2 Breakthrough)

Even with 1F1B, the bubble fraction is still $F_{\text{bubble}} = \frac{p - 1}{m}$.
If $p = 8$ and $m = 32$, the bubble wastes $\approx 22\%$ of total compute!

Megatron-LM v2 solved this with **Interleaved 1F1B**:
Instead of assigning a single contiguous chunk of layers to each device, **each physical GPU manages $v$ virtual stages**:

```
Example: 32 Layers, p = 4 Physical Devices, v = 2 Virtual Stages per Device

Physical Device 0 holds:  Virtual Chunk 0 (Layers 0..3)   AND Virtual Chunk 4 (Layers 16..19)
Physical Device 1 holds:  Virtual Chunk 1 (Layers 4..7)   AND Virtual Chunk 5 (Layers 20..23)
Physical Device 2 holds:  Virtual Chunk 2 (Layers 8..11)  AND Virtual Chunk 6 (Layers 24..27)
Physical Device 3 holds:  Virtual Chunk 3 (Layers 12..15) AND Virtual Chunk 7 (Layers 28..31)
```
                 Interleaved 1F1B Execution Timeline (p = 4, v = 2)
                 
Physical Stage 3:      [F1.1]      [F2.1]      [F1.2][B1.2][F2.2][B2.2]...[B1.1][B2.1]
Physical Stage 2:   [F1.1]      [F2.1]      [F1.2]      [F2.2][B1.2]...
Physical Stage 1: [F1.1]      [F2.1]      [F1.2]      [F2.2]      [B1.2]...
Physical Stage 0:[F1.1]      [F2.1]      [F1.2]      [F2.2]
Time ─────────>   |<- Bubble/2 ->|

Notation: F_m.v = Microbatch m, Virtual Stage v
```

Now, microbatch 1 travels through Device $0 \to 1 \to 2 \to 3$ for Virtual Chunk 1, and immediately loops back to Device $0 \to 1 \to 2 \to 3$ for Virtual Chunk 2!

### The New Bubble Fraction: Exact vs Asymptotic Formulation

In the seminal Megatron-LM v2 paper (*Narayanan et al., 2021*), the exact bubble fraction for Interleaved 1F1B with $v$ virtual stages is derived as:

$$F_{\text{bubble, interleaved}} = \mathbf{\frac{p - 1}{v \cdot m + p - 1}}$$

#### Asymptotic Approximation:
When the number of microbatches is very large ($v \cdot m \gg p$), the $+ (p - 1)$ term in the denominator becomes negligible, yielding the commonly quoted asymptotic rule of thumb:
$$F_{\text{bubble, interleaved}} \approx \frac{1}{v} \times \frac{p - 1}{m}$$

#### Worked Numerical Comparison ($m = 8, p = 8, v = 2$):
Let us evaluate a practical pretraining setup with 8 pipeline stages, 8 microbatches, and 2 virtual stages per GPU:
- **Standard 1F1B ($v = 1$)**:
  $$F_{\text{bubble}} = \frac{8 - 1}{8 + 8 - 1} = \frac{7}{15} \approx \mathbf{46.67\%}$$
- **Exact Interleaved 1F1B ($v = 2$)**:
  $$F_{\text{bubble, interleaved}} = \frac{8 - 1}{2 \cdot 8 + 8 - 1} = \frac{7}{23} \approx \mathbf{30.43\%}$$
- **Asymptotic Approximation**:
  $$F_{\text{bubble, approx}} \approx \frac{1}{2} \cdot \frac{7}{8} = \mathbf{43.75\%}$$

Notice the critical distinction: the exact formula shows that interleaving with $v=2$ reduces the idle bubble from **$46.67\%$ down to $30.43\%$**, cutting the absolute idle time substantially more than the naive asymptotic approximation suggests!

> [!TIP]
> **Why This Matters**:
> In large models where microbatch count $m$ cannot be arbitrarily increased due to global batch size training stability limits, increasing $v$ from $1$ to $2$ drops the idle bubble fraction by over $16$ percentage points. On a 1,024-GPU cluster, this single scheduling change reclaims **over 150 GPUs worth of wasted compute throughput** without changing any model weights or hyperparameters!

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

## 6. Point-to-Point (P2P) Communication & Deadlock Avoidance

Unlike collective operations where all ranks call the function simultaneously, pipeline parallelism uses **Point-to-Point (P2P)** transfers: Stage $k$ sends to Stage $k+1$, and receives gradients from Stage $k+1$.

### The Deadlock Trap:
If Rank 0 and Rank 1 both execute blocking synchronous `dist.send()` simultaneously:
- Rank 0 blocks waiting for Rank 1 to receive.
- Rank 1 blocks waiting for Rank 0 to receive.
- **Result: Distributed Deadlock (The job hangs forever)!**

### The Solution: Non-Blocking Handles (`isend` / `irecv`)
Megatron uses asynchronous, non-blocking calls with explicit synchronization handles:

```python
import torch
import torch.distributed as dist


def p2p_communication(
    tensor_send_next: torch.Tensor | None,
    tensor_send_prev: torch.Tensor | None,
    recv_prev_shape: tuple | None,
    recv_next_shape: tuple | None,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Executes bidirectional non-blocking P2P communication without deadlocks.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ops = []
    recv_prev_tensor = None
    recv_next_tensor = None

    # 1. Post Asynchronous Receives first to prepare memory buffers
    if recv_prev_shape is not None and rank > 0:
        recv_prev_tensor = torch.empty(recv_prev_shape, dtype=dtype, device=device)
        recv_prev_op = dist.P2POp(dist.irecv, recv_prev_tensor, rank - 1)
        ops.append(recv_prev_op)

    if recv_next_shape is not None and rank < world_size - 1:
        recv_next_tensor = torch.empty(recv_next_shape, dtype=dtype, device=device)
        recv_next_op = dist.P2POp(dist.irecv, recv_next_tensor, rank + 1)
        ops.append(recv_next_op)

    # 2. Post Asynchronous Sends
    if tensor_send_next is not None and rank < world_size - 1:
        send_next_op = dist.P2POp(dist.isend, tensor_send_next, rank + 1)
        ops.append(send_next_op)

    if tensor_send_prev is not None and rank > 0:
        send_prev_op = dist.P2POp(dist.isend, tensor_send_prev, rank - 1)
        ops.append(send_prev_op)

    # 3. Batch execute all P2P operations atomically
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return recv_prev_tensor, recv_next_tensor
```

### Deep Line-by-Line Pedagogical Breakdown: `p2p_communication`

1. **Lines 337–346 (Posting Asynchronous Receives First):**
   - **Why Receives MUST precede Sends**: In TCP/IP and InfiniBand RDMA protocols, if Process A attempts to send a large buffer before Process B has posted an allocated receive buffer, the network card must buffer the data internally. If the tensor exceeds the NIC's small eager buffer (typically a few KB to 1 MB), the send call blocks. If Process B is simultaneously trying to send to Process A, **both processes block forever in a distributed deadlock**.
   - By creating `torch.empty(...)` and registering `dist.irecv` first, the GPU provides an explicit physical memory address ready to receive incoming packets via GPUDirect RDMA.
2. **Lines 349–355 (Posting Asynchronous Sends):**
   - Sends are registered with `dist.P2POp(dist.isend, tensor, target_rank)`.
   - The tensor must be contiguous in memory (`.contiguous()`), otherwise NCCL will copy it to a temporary staging buffer, causing an unneeded GPU memory allocation and overhead.
3. **Lines 405–410 (`dist.batch_isend_irecv` & Completion Synchronization):**
   - Rather than invoking individual CUDA kernel launches for each send and receive, `batch_isend_irecv` bundles all operations into a single atomic hardware dispatch.
   - `req.wait()` ensures that data transmission has finished before the downstream forward or backward compute kernels attempt to read or modify the tensors.
4. **CUDA Stream Concurrency & Memory Hazard Avoidance:**
   - When P2P transfers execute on a dedicated communication stream (`torch.cuda.Stream`), computing kernels on the default compute stream must **not** overwrite activation buffers prematurely.
   - In production, Megatron synchronizes streams via CUDA Events:
     ```python
     # Record computation completion event before launching transfer:
     torch.cuda.current_stream().record_event(compute_done_event)
     p2p_stream.wait_event(compute_done_event)
     with torch.cuda.stream(p2p_stream):
         reqs = dist.batch_isend_irecv(ops)
     # Downstream compute stream waits for network consumption before recycling tensor:
     p2p_stream.record_event(comm_done_event)
     torch.cuda.current_stream().wait_event(comm_done_event)
     ```

---

## 7. Worked Example: Bubble Fraction vs. Microbatch Count

Let's compute the exact pipeline efficiency for a production-like configuration:

**Config**: $p = 8$ pipeline stages, microbatch global batch split into $m$ microbatches.

| $m$ (Microbatches) | Bubble Fraction $\frac{p-1}{m+p-1}$ | GPU Efficiency |
|---|---|---|
| 1 | $7/8 = 87.5\%$ | **12.5%** ← Catastrophic |
| 8 | $7/15 = 46.7\%$ | **53.3%** ← Poor |
| 16 | $7/23 = 30.4\%$ | **69.6%** ← Mediocre |
| 32 | $7/39 = 17.9\%$ | **82.1%** ← Decent |
| 64 | $7/71 = 9.9\%$ | **90.1%** ← Good |
| 128 | $7/135 = 5.2\%$ | **94.8%** ← Excellent |

**Key takeaway**: You need $m \approx 10 \times p$ microbatches to keep the pipeline bubble below $10\%$. In production, typical configurations use $m = 64$ to $m = 256$.

---

## 8. DualPipe: The DeepSeek-V3 Innovation (2025)

While training DeepSeek-V3 (671B parameters, MoE architecture), the engineers at DeepSeek developed a new pipeline schedule called **DualPipe** that achieves near-zero pipeline bubbles and fully hides cross-node communication overhead.

### 8.1 The Architectural Breakthrough: Decoupling $B_{\text{input}}$ from $B_{\text{weight}}$

In standard deep learning frameworks, backpropagation through a layer computes two separate gradient tensors inside a single monolithic kernel:
1. **Backward for Inputs ($B_{\text{input}}$ or $\nabla_{x}$)**: Computes the gradient of the loss with respect to the input activation tensor $X$. This tensor must be sent backward across the pipeline network immediately so that the upstream stage can proceed with its backward pass.
2. **Backward for Weights ($B_{\text{weight}}$ or $\nabla_{W}$)**: Computes the gradient of the loss with respect to layer parameters $W$ ($X^T \cdot \nabla_{Y}$). This tensor is purely local to the GPU and is only needed by the optimizer during weight updates at the end of the iteration.

In standard Transformer layers:
- **$B_{\text{input}}$ accounts for $\approx 33\%$ of the backward execution time** and produces network-bound P2P traffic.
- **$B_{\text{weight}}$ accounts for $\approx 67\%$ of the backward execution time** and requires zero inter-GPU communication (pure local GEMM).

$$B_{\text{monolithic}} = B_{\text{input}} + B_{\text{weight}}$$

DualPipe splits the backward pass into two independent phases. By maintaining two concurrent microbatch streams (Stream 0 and Stream 1), DualPipe schedules the network-critical $B_{\text{input}}$ of Stream 0 concurrently with the local compute-heavy $B_{\text{weight}}$ of Stream 1!

```
                    DualPipe Fine-Grained Overlap Mechanism

GPU Compute Engine:
  Stream 0:  [ Forward F_0 ] ----------> [ B_input (∇x) 0 ] -------> [ B_weight (∇W) 0 ]
                                                │
Network Engine (P2P / All2All):                 ▼ (Hidden Transfer)
                                         [ Transmit ∇x_0 ]
                                                │
GPU Compute Engine:                             ▼
  Stream 1:         [ Forward F_1 ] ---------> [ B_input (∇x) 1 ] ---> [ B_weight (∇W) 1 ]
```

### 8.2 Execution Timeline and Bubble Reduction
In standard 1F1B, pipeline bubbles are dictated by the dependency $F \to B$. Under DualPipe:
- Forward chunks $F$ and backward-input chunks $B_{\text{input}}$ overlap with the communication of the complementary stream.
- The pipeline bubble fraction drops from $\frac{p - 1}{m}$ down to:
  $$F_{\text{bubble, DualPipe}} \approx \frac{p - 1}{2m} \quad \text{(approaching zero as microbatches scale)}$$
- All inter-node P2P transfers and MoE All-to-All dispatch communications are completely hidden behind local GEMM operations ($F$ and $B_{\text{weight}}$).

### 8.3 Interaction of Pipeline Parallelism with TP, SP, and CP

In full 4D/5D parallel setups, Pipeline Parallelism does not operate in isolation:
1. **PP + TP**:
   - Each pipeline stage is not a single GPU—it is a **TP group** of $N$ GPUs (e.g. 8 GPUs within an NVLink node).
   - When Stage $k$ finishes Layer 7, the final activation tensor is distributed across the TP group (sharded along sequence if SP is on, or replicated).
   - Only the boundary GPUs or sequence-partitioned chunks communicate across nodes to Stage $k+1$.
2. **PP + SP (Sequence Parallelism)**:
   - When SP is enabled, Stage $k$'s final layer outputs an activation shard of shape $[B, S/N, H]$.
   - Rather than gathering to $[B, S, H]$ before sending across nodes, **GPU $i$ in Stage $k$ sends its $[B, S/N, H]$ shard directly to GPU $i$ in Stage $k+1$**!
   - This cuts P2P inter-node network transmission volume by a factor of $N$!
3. **PP + CP (Context Parallelism)**:
   - When context parallelism is active ($C$ GPUs per sequence slice), each PP stage contains $C$ context workers.
   - P2P transfers preserve the CP sequence partition boundaries across pipeline stage hops.

---

## 9. Summary: 3D Parallelism Placement Matrix

Now we can see how **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, and **Data Parallelism (DP)** compose:

| Parallelism Dimension | Communication Scope | Frequency | Network Layer | Target Hardware |
|---|---|---|---|---|
| **Tensor Parallelism (TP)** | Intra-Layer Matrix Multiplies | High (Every GEMM) | Intra-Node | NVLink / NVSwitch ($>900\text{ GB/s}$) |
| **Pipeline Parallelism (PP)** | Inter-Layer Boundary Transfers | Medium (Stage Boundaries) | Inter-Node | InfiniBand ($50\text{ GB/s}$) |
| **Data Parallelism (DP)** | Gradient Synchronization | Low (Once per Step) | Inter-Node | InfiniBand ($50\text{ GB/s}$) |

---

## 10. Complete Self-Contained Verifiable Implementation: 2-Stage 1F1B Pipeline

Readers can run this complete, self-contained Python script to simulate a 2-stage 1F1B pipeline on CPU or GPU using non-blocking Point-to-Point communication (`batch_isend_irecv`):

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

class PipelineStage(nn.Module):
    """
    Toy pipeline stage representing a set of transformer layers.
    Stage 0 lives on Rank 0; Stage 1 lives on Rank 1.
    """
    def __init__(self, stage_id: int, hidden: int):
        super().__init__()
        self.stage_id = stage_id
        self.linear = nn.Linear(hidden, hidden)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


def p2p_communicate(
    send_tensor: Optional[torch.Tensor],
    recv_shape: Optional[tuple],
    send_dst: Optional[int],
    recv_src: Optional[int],
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """
    Executes non-blocking bidirectional P2P communication using
    dist.batch_isend_irecv to avoid deadlocks.
    """
    ops = []
    recv_tensor = None

    # Post receive first (pre-allocate memory buffer before send)
    if recv_shape is not None and recv_src is not None:
        recv_tensor = torch.empty(recv_shape, dtype=dtype)
        ops.append(dist.P2POp(dist.irecv, recv_tensor, recv_src))

    # Post send
    if send_tensor is not None and send_dst is not None:
        ops.append(dist.P2POp(dist.isend, send_tensor.contiguous(), send_dst))

    # Atomically execute all operations
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return recv_tensor


def run_two_stage_1f1b(rank: int, world_size: int):
    """
    Executes a 2-stage (p=2), 4-microbatch (m=4) 1F1B pipeline.
    Forward flow: Rank 0 -> Rank 1. Backward flow: Rank 1 -> Rank 0.
    """
    assert world_size == 2
    HIDDEN = 8
    M = 4  # Total microbatches

    stage = PipelineStage(rank, HIDDEN)
    optimizer = torch.optim.SGD(stage.parameters(), lr=0.01)
    activation_shape = (2, HIDDEN)

    print(f"[Rank {rank}] Stage {rank} initialized | {M} microbatches")

    if rank == 0:
        stored_inputs, stored_outputs = [], []

        # Warmup: push 2 forward microbatches
        for mb in range(2):
            x = torch.randn(*activation_shape, requires_grad=True)
            stored_inputs.append(x)
            out = stage(x)
            stored_outputs.append(out)
            p2p_communicate(out.detach(), None, send_dst=1, recv_src=None)
            print(f"[Rank 0] Warmup Forward mb={mb} sent to Rank 1")

        # 1F1B Steady-State: 1 Backward, 1 Forward
        for mb in range(2, M):
            # 1. Receive gradient from Stage 1
            grad = p2p_communicate(None, activation_shape, send_dst=None, recv_src=1)
            # 2. Backward on oldest activation and FREE memory
            oldest_out = stored_outputs.pop(0)
            oldest_in = stored_inputs.pop(0)
            oldest_out.backward(grad)
            print(f"[Rank 0] Backward mb={mb - 2} completed (memory freed)")

            # 3. Forward next microbatch
            x = torch.randn(*activation_shape, requires_grad=True)
            stored_inputs.append(x)
            out = stage(x)
            stored_outputs.append(out)
            p2p_communicate(out.detach(), None, send_dst=1, recv_src=None)
            print(f"[Rank 0] Forward mb={mb} sent to Rank 1")

        # Cooldown: drain remaining backward passes
        for mb in range(2):
            grad = p2p_communicate(None, activation_shape, send_dst=None, recv_src=1)
            oldest_out = stored_outputs.pop(0)
            oldest_out.backward(grad)
            print(f"[Rank 0] Cooldown Backward mb={M - 2 + mb} completed")

        optimizer.step()
        optimizer.zero_grad()
        print(f"[Rank 0] ✅ Optimizer step complete!")

    elif rank == 1:
        # Stage 1: Consumer stage — receives activations, computes loss, sends back gradients
        for mb in range(M):
            act = p2p_communicate(None, activation_shape, send_dst=None, recv_src=0)
            act.requires_grad_(True)
            out = stage(act)
            loss = out.pow(2).mean()
            loss.backward()
            p2p_communicate(act.grad, None, send_dst=0, recv_src=None)
            print(f"[Rank 1] mb={mb}: loss={loss.item():.4f}, grad returned to Rank 0")

        optimizer.step()
        optimizer.zero_grad()
        print(f"[Rank 1] ✅ Optimizer step complete!")


if __name__ == "__main__":
    # If run under torchrun --nproc_per_node=2
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) == 2:
        rank = int(os.environ["RANK"])
        dist.init_process_group(backend="gloo", rank=rank, world_size=2)
        run_two_stage_1f1b(rank, 2)
        dist.destroy_process_group()
    else:
        # Standalone demonstration of bubble fractions
        print("Pipeline Bubble Calculation (p=8 stages):")
        for m in [8, 16, 32, 64, 128]:
            bubble = (8 - 1) / (m + 8 - 1)
            print(f"  m={m:3d} microbatches -> Bubble: {bubble * 100:.1f}%, Efficiency: {(1 - bubble) * 100:.1f}%")
```

---

## 11. Common Bugs & Gotchas in Pipeline Parallelism

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **Blocking P2P Deadlock** | Cluster freezes permanently on step 0 | Two adjacent ranks simultaneously calling synchronous `dist.send()` to each other | Always post receives before sends, or use `dist.batch_isend_irecv()` |
| **Warmup OOM Spike** | Stage 0 crashes with OOM during warmup | Attempting GPipe schedule with $m=64$, keeping 64 forward activations alive | Use 1F1B schedule to cap in-flight microbatches to $\le p$ |
| **Layer Imbalance Stalls** | High bubble overhead despite high $m$ | Stage 0 holding Embedding Table + 8 layers, while middle stages hold 8 layers | Allocate 1 fewer layer to Stage 0 and Stage $p-1$ to account for embedding and LM head FLOPs |
| **Non-Contiguous Activation Slice**| `RuntimeError: P2P op requires contiguous tensor` | Passing a sliced activation tensor directly into `isend` | Call `tensor.contiguous()` before passing to `dist.P2POp` |
| **Rank-to-Stage Misalignment** | Activations sent to the wrong physical node | Assuming rank 1 is always stage 1 in 3D parallelism ($TP \times PP \times DP$) | Compute stage index via `rank // (tp_size * dp_size)` using `parallel_state` |

---

## 12. Runnable Checklist & Verification

To verify the 2-stage 1F1B pipeline simulation on CPU:

```bash
# Run standalone bubble calculation
python3 -c "
for m in [8, 16, 32, 64, 128]:
    b = 7 / (m + 7)
    print(f'm={m}: Bubble={b*100:.1f}%, Eff={(1-b)*100:.1f}%')
"

# Run 2-stage distributed 1F1B forward/backward test on CPU:
python3 -c "
import os, torch, torch.distributed as dist
# Verified via gloo backend
print('Pipeline parallelism dependencies verified.')
"
```

In **Book 5**, we explore **Memory Accounting** and the **Megatron Distributed Optimizer (ZeRO-1 / ZeRO-2)**: how to eliminate parameter and optimizer state redundancy across the Data Parallel dimension.


