# Distributed Compute Foundations & Interconnect Topologies
> **SPMD Process Groups, Autograd Conjugate Operators, and Communication Mechanics**

---

## 1.1. Introduction: The Physical Reality of Distributed AI

To understand distributed training architectures like **Megatron-LM** and **Megatron Core**, you must first understand the hard physical bottlenecks that dictate hardware design. 

> [!NOTE]
> **Megatron-LM vs. Megatron Core (M-Core)**:
> - **Megatron-LM**: The historical end-to-end training repository, scripts, and pre-baked CLI entry points (e.g., `pretrain_gpt.py`) maintained by NVIDIA.
> - **Megatron Core (`megatron.core`)**: The modern, modular, library-first refactoring introduced in 2023–2024. M-Core decouples parallelism primitives (`ColumnParallelLinear`, `DistributedOptimizer`, `MoETokenDispatcher`, `TransformerConfig`) into standalone, reusable PyTorch modules that can be plugged into NeMo, PyTorch Lightning, or custom pretraining pipelines.

Large Language Model (LLM) training is not limited by software algorithms; it is strictly governed by the physics of semiconductor memory and network interconnects.

### 1.1.1 The Physical Reality of VRAM Limits
A common misconception is that a 70-billion-parameter model requires only `70B * 2 bytes = 140 GB` of GPU memory. In reality, mixed-precision pretraining requires **`16 bytes per parameter`** of static model state (`1,120 GB` for 70B), plus hundreds of gigabytes of dynamic activation memory.

> [!IMPORTANT]
> The complete mathematical proof of the **16Phi Law**—including the IEEE 754 floating-point swamping theorem and AdamW optimizer state dynamics—is derived rigorously in **[Book 5: Memory Accounting & The Megatron Distributed Optimizer](#book-5-memory-accounting--the-megatron-distributed-optimizer)**.
>
> Because no single GPU provides `>1 TB` of High Bandwidth Memory, partitioning model state across multiple accelerators is not merely an optimization—it is a physical necessity.

---

### 1.1.2 The Interconnect Bandwidth & Latency Hierarchy
Distributed training splits computation across hundreds or thousands of GPUs. But GPUs can only work together if they can exchange data at high speeds. 

Crucially, **not all communication paths are created equal**. As data moves further away from the compute cores, bandwidth drops by orders of magnitude, and latency skyrockets:

| Memory / Interconnect Tier | Physical Location | Typical Bandwidth | Typical Latency | Appropriate Parallelism Strategy |
|---|---|---|---|---|
| **SRAM (Register / L1)** | On-Die (Inside Streaming Multiprocessor) | `~ 19.0 TB/s` | `~ 1 ns` | Fused Kernels (FlashAttention, SwiGLU) |
| **HBM3e (GPU VRAM)** | On-Substrate (Stacks around GPU die) | `~ 3.35 - 4.8 TB/s` | `~ 50 - 100 ns` | Local Matrix Multiply (GEMM) |
| **NVLink 4 / NVLink 5** | Intra-Node (GPU-to-GPU within 1 server) | `~ 900 - 1,800 GB/s` | `~ 0.5 - 1.0 us` | **Tensor Parallelism (TP) & Sequence Parallelism (SP)** |
| **PCIe Gen 5** | Motherboard Bus (CPU `<= ftrightarrow` GPU / Host RAM) | `~ 64 GB/s` | `~ 2 - 5 us` | CPU Offload (ZeRO-Offload), Checkpoint Saving |
| **InfiniBand NDR / RoCE v2** | Inter-Node Network (Cross-Server Cables) | `~ 50 GB/s` (400 Gbps) | `~ 5 - 15 us` | **Pipeline Parallelism (PP) & Data Parallelism (DP)** |

### 1.1.3 The Golden Rule of Distributed Scaling:
> **High-frequency, latency-sensitive operations (Tensor Parallelism) must NEVER cross the NVLink boundary! Low-frequency, bandwidth-tolerant operations (Pipeline Parallelism, Data Parallelism) are mapped across nodes over InfiniBand.**

Why? Because Tensor Parallelism inserts communication *into every single Transformer layer* (two All-Reduces per layer, forward and backward). If an 80-layer model executes 320 All-Reduces per training step across a high-latency 10 us InfiniBand network, the GPU Streaming Multiprocessors (SMs) spend >80% of their time idling, waiting for network packets!

---

## 1.2. Distributed Topologies & Cluster Architecture

A production distributed training cluster is organized into a hierarchical topology:

```
+=============================================================================+
|                                 NODE 0                                      |
|                                                                             |
|  +--------------+   NVLink (900 GB/s)   +--------------+                    |
|  |    GPU 0     |<=====================>|    GPU 1     |                    |
|  +--------------+                       +--------------+                    |
|         ^                                      ^                            |
|         | NVSwitch (Full Mesh / Crossbar)      |                            |
|         v                                      v                            |
|  +--------------+   NVLink (900 GB/s)   +--------------+                    |
|  |    GPU 2     |<=====================>|    GPU 3     |                    |
|  +--------------+                       +--------------+                    |
|         || PCIe Gen 5 (64 GB/s)                ||                           |
|         vv                                     vv                           |
|  +---------------------------------------------------+                      |
|  |                   HOST CPU & RAM                  |                      |
|  +---------------------------------------------------+                      |
|                           ||                                                |
|                   Host / PCIe Bus                                           |
|                           vv                                                |
|  +---------------------------------------------------+                      |
|  |              InfiniBand / RoCE NICs               |                      |
|  +---------------------------------------------------+                      |
+===========================||================================================+
                            || Inter-Node Network (400 - 800 Gbps InfiniBand)
+===========================||================================================+
|                                 NODE 1                                      |
|  +---------------------------------------------------+                      |
|  |              InfiniBand / RoCE NICs               |                      |
|  +---------------------------------------------------+                      |
|                           ||                                                |
|  +---------------------------------------------------+                      |
|  |                   HOST CPU & RAM                  |                      |
|  +---------------------------------------------------+                      |
|         ^^                                     ^^                           |
|  +--------------+                       +--------------+                    |
|  |    GPU 4     |<=====================>|    GPU 5     |                    |
|  +--------------+                       +--------------+                    |
|         ^                                      ^                            |
|         | NVSwitch                             |                            |
|         v                                      v                            |
|  +--------------+                       +--------------+                    |
|  |    GPU 6     |<=====================>|    GPU 7     |                    |
|  +--------------+                       +--------------+                    |
+=============================================================================+
```

### 1.2.1 Intra-Node Architecture: The Role of NVSwitch
In modern HGX systems (like H100 8-GPU nodes), the GPUs are not simply wired in a point-to-point ring. Instead, they are connected via physical **NVSwitch chips** that form a non-blocking crossbar fabric:
- Any GPU can send data directly to any other GPU inside the node at the full bidirectional NVLink speed (`900 GB/s`).
- GPU 0 can read GPU 7's memory without involving the CPU, host RAM, or the operating system kernel. This is known as **Peer-to-Peer (P2P) Direct Memory Access (DMA)**.

### 1.2.2 Inter-Node Architecture: GPUDirect RDMA
When communicating across servers (from Node 0 to Node 1):
- **Traditional Networking (Slow)**: GPU memory `->` PCIe Bus `->` Host CPU RAM `->` Linux TCP/IP Stack `->` NIC `->` Network Wire. This path incurs massive memory-copy overhead and CPU interrupts.
- **GPUDirect RDMA (Remote Direct Memory Access)**: The InfiniBand Network Interface Card (NIC) accesses the GPU's HBM memory directly over PCIe/NVLink without touching the CPU or system memory! This reduces inter-node latency from `>50 us` down to `<5 us`.

---

---

## 1.3. The `torch.distributed` Execution Model

PyTorch relies on the **SPMD** (Single Program, Multiple Data) paradigm. Every worker executes the exact same Python script, but operates on different ranks and slices of data.

### 1.3.1 Fundamental Concepts
1. **World Size (W)**: The total number of parallel processes in the distributed job.
2. **Global Rank (`r in [0, W-1]`)**: A unique integer identifier assigned to each process across the entire cluster.
3. **Local Rank (`r_local in [0, G-1]`)**: The rank of the process relative to its local machine/node (e.g., 0 to 7 on an 8-GPU node).
4. **Backend**: The underlying communication library:
   - **`nccl` (NVIDIA Collective Communications Library)**: Hardware-accelerated for NVIDIA GPUs via NVLink, NVSwitch, and GPUDirect RDMA. The mandatory production backend for all LLM pretraining.
   - **`gloo`**: Multi-platform collective communications engine that runs over CPU memory, POSIX threads, and standard TCP/IP sockets. Used for CPU-only environments; not a production backend.
   - **`mpi`**: Message Passing Interface (used primarily in traditional supercomputing clusters).

#### 1.3.1.1 PyTorch Collective Backend Compatibility Matrix

The table below summarizes the key collective APIs and their NCCL support status:

| Collective Primitive | PyTorch Functional API | `nccl` (NVIDIA GPU Cluster) | Notes |
| :--- | :--- | :--- | :--- |
| **All-Reduce** | `dist.all_reduce(tensor, op)` | ✅ Hardware Accelerated (NVLink / Ring) | Primary primitive for DDP and Tensor Parallelism. |
| **All-Gather** | `dist.all_gather_into_tensor(out, in)` | ✅ Hardware Accelerated | Used in Sequence Parallelism and ZeRO weight gathering. |
| **Broadcast** | `dist.broadcast(tensor, src)` | ✅ Hardware Accelerated (Tree) | Used for parameter initialization sync. |
| **Reduce-Scatter** | `dist.reduce_scatter_tensor(out, in, op)` | ✅ Hardware Accelerated | Core primitive for ZeRO-2 gradient partitioning (DistributedOptimizer). |
| **P2P Send / Recv** | `dist.isend()`, `dist.irecv()`, `dist.batch_isend_irecv()` | ✅ GPUDirect P2P / NVLink | Core primitive for Pipeline Parallelism (1F1B) and Ring Attention. |
| **All-to-All** | `dist.all_to_all_single(out, in)` | ✅ Hardware Accelerated | Core primitive for MoE token routing (Expert Parallelism). |

---

## 1.4. Collective Communication Primitives & Ring Algorithms

In distributed deep learning, workers communicate using **Collectives** (operations involving a group of processes) and **Point-to-Point (P2P)** operations.

### 1.4.1 Broadcast
One root rank sends its local tensor to all other ranks.

```
Rank 0 (Root): [ A ] ──────────┬──────────────┬──────────────>
                               │              │
Rank 1:        [   ] <─────────┘              │
Rank 2:        [   ] <────────────────────────┘
Result: All ranks hold [ A ]
```

### 1.4.2 Scatter & Gather
- **Scatter**: Root splits a tensor along a specified dimension into N equal chunks and distributes chunk i to rank i.
- **Gather**: Root collects chunks from all N ranks and concatenates them into a single tensor.

```
Scatter:
Root [ A | B | C | D ]  ───>  Rank 0: [ A ]
                        ───>  Rank 1: [ B ]
                        ───>  Rank 2: [ C ]
                        ───>  Rank 3: [ D ]

Gather:
Rank 0: [ A ]  ───┐
Rank 1: [ B ]  ───┼───>  Root [ A | B | C | D ]
Rank 2: [ C ]  ───┤
Rank 3: [ D ]  ───┘
```

---

### 1.4.3 All-Reduce: The Workhorse of Model Training

Every rank starts with a tensor of size S. At the end of the operation, **every rank holds the elementwise sum (or reduction) of all input tensors**.

```
Rank 0: [ A0 ] ┐
Rank 1: [ A1 ] ┼── All-Reduce (SUM) ──> All Ranks hold: [ A0 + A1 + A2 + A3 ]
Rank 2: [ A2 ] │
Rank 3: [ A3 ] ┘
```

#### 1.4.3.1 The Ring All-Reduce Algorithm: Step-by-Step Mechanical Walkthrough

Why does naive All-Reduce fail at scale?
- In a naive centralized approach, all N workers send their tensors of size S to Rank 0. Rank 0 sums them and broadcasts the result back.
- **The Bottleneck**: Rank 0's network interface must ingest `(N-1) * S` bytes and transmit `(N-1) * S` bytes! As cluster size N grows to `1,024` GPUs, Rank 0 collapses under terabytes of incoming traffic. Communication time scales as `O(N * S)`.

To solve this, **Baidu Silicon Valley AI Lab (Gibiansky, 2017)** introduced **Ring All-Reduce** to deep learning, adapting classic high-performance computing algorithms to GPUs.

In Ring All-Reduce, all N GPUs are arranged in a logical ring. Every GPU only ever communicates with its **immediate right neighbor** (sending) and its **immediate left neighbor** (receiving):

```
       Rank 0 ───────> Rank 1
         ^               │
         │               │
         │               v
       Rank 3 <─────── Rank 2
```

The tensor of size S on each GPU is partitioned into N equal chunks: `[c_0, c_1, ..., c_N-1]`, each of size `(S / N)`.
The algorithm executes in **two distinct phases of N-1 steps each**:

---

#### 1.4.3.2 Phase 1: Scatter-Reduce (N-1 Steps)
In each step, every rank sends one chunk to its right neighbor and simultaneously receives one chunk from its left neighbor. Upon receiving a chunk, the rank **adds it in-place** to its local chunk.

Let us trace 4 GPUs (`N = 4`) with chunks `[c_0, c_1, c_2, c_3]`:

```
INITIAL STATE (Step 0):
  GPU 0 holds:  [ c0_0 ]  [ c1_0 ]  [ c2_0 ]  [ c3_0 ]
  GPU 1 holds:  [ c0_1 ]  [ c1_1 ]  [ c2_1 ]  [ c3_1 ]
  GPU 2 holds:  [ c0_2 ]  [ c1_2 ]  [ c2_2 ]  [ c3_2 ]
  GPU 3 holds:  [ c0_3 ]  [ c1_3 ]  [ c2_3 ]  [ c3_3 ]

STEP 1 (GPU sends chunk to right neighbor):
  GPU 0 sends c3_0 to GPU 1  --> GPU 1 sums: c3_1 + c3_0
  GPU 1 sends c0_1 to GPU 2  --> GPU 2 sums: c0_2 + c0_1
  GPU 2 sends c1_2 to GPU 3  --> GPU 3 sums: c1_3 + c1_2
  GPU 3 sends c2_3 to GPU 0  --> GPU 0 sums: c2_0 + c2_3

STEP 2 (GPU sends accumulated chunk to right neighbor):
  GPU 0 sends (c2_0 + c2_3) to GPU 1  --> GPU 1 sums: c2_1 + c2_0 + c2_3
  GPU 1 sends (c3_1 + c3_0) to GPU 2  --> GPU 2 sums: c3_2 + c3_1 + c3_0
  GPU 2 sends (c0_2 + c0_1) to GPU 3  --> GPU 3 sums: c0_3 + c0_2 + c0_1
  GPU 3 sends (c1_3 + c1_2) to GPU 0  --> GPU 0 sums: c1_0 + c1_3 + c1_2

STEP 3 (Scatter-Reduce Final Step):
  GPU 0 sends partially summed c1 to GPU 1  --> GPU 1 now holds FULL SUM: Σ c1
  GPU 1 sends partially summed c2 to GPU 2  --> GPU 2 now holds FULL SUM: Σ c2
  GPU 2 sends partially summed c3 to GPU 3  --> GPU 3 now holds FULL SUM: Σ c3
  GPU 3 sends partially summed c0 to GPU 0  --> GPU 0 now holds FULL SUM: Σ c0
```

**End of Scatter-Reduce**: Exactly after `N-1 = 3` steps, each GPU holds the **fully reduced sum of exactly ONE chunk**:
- GPU 0 holds the full sum of chunk 0: `sum(i=0)^3 c[0,i]`
- GPU 1 holds the full sum of chunk 1: `sum(i=0)^3 c[1,i]`
- GPU 2 holds the full sum of chunk 2: `sum(i=0)^3 c[2,i]`
- GPU 3 holds the full sum of chunk 3: `sum(i=0)^3 c[3,i]`

Data transferred per rank during Scatter-Reduce:
```text
Data_scatter-reduce = (N - 1) * (S / N)
```

---

#### 1.4.3.3 Phase 2: All-Gather (N-1 Steps)
Now, each GPU has one fully reduced chunk, but needs the other N-1 fully reduced chunks from the other ranks.
In the All-Gather phase, the exact same ring communication pattern occurs, but **instead of summing, each rank simply overwrites its local buffer with the received fully-reduced chunk**:

- Step 1: GPU 0 sends sum c_0 to GPU 1; GPU 1 sends sum c_1 to GPU 2; etc.
- Step 2: GPU 1 forwards sum c_0 to GPU 2; GPU 2 forwards sum c_1 to GPU 3; etc.
- Step 3: GPU 2 forwards sum c_0 to GPU 3; GPU 3 forwards sum c_1 to GPU 0; etc.

After `N-1 = 3` steps of All-Gather, **all 4 GPUs hold the identical, fully reduced tensor** `[ sum c_0, sum c_1, sum c_2, sum c_3 ]`!

Data transferred per rank during All-Gather:
```text
Data_all-gather = (N - 1) * (S / N)
```

---

#### 1.4.3.4 The Master Volume Equation and Bandwidth Bound
Summing both phases, the total data sent (and received) by each GPU is:
```text
Total Transferred Volume per Rank = 2 * (((N - 1) / N)) * S
```

```
As cluster size N grows:
  N = 2:    Total Data = 2 * (1/2) * S = 1.0 * S
  N = 4:    Total Data = 2 * (3/4) * S = 1.5 * S
  N = 8:    Total Data = 2 * (7/8) * S = 1.75 * S
  N = 64:   Total Data = 2 * (63/64) * S = 1.968 * S
  N -> ∞:   Total Data -> 2 * S
```

### 1.4.4 The Profound Architectural Consequence:
> **The communication bandwidth demand per GPU is constant (`< 2S`). You can scale from 8 GPUs to 16,384 GPUs, and the network load on any individual GPU link does NOT explode! It approaches asymptotically `2 * S`.**

---

#### 1.4.4.1 Network Latency vs Bandwidth: The alpha-beta Model
The time taken to run Ring All-Reduce is formally expressed by the Hockney communication model:

```text
Time_Ring = 2(N - 1)alpha + 2(((N - 1) / N))Sbeta
```

Where:
- alpha: **Latency / Network Message Setup Time** (time to negotiate and initiate a packet transfer, typically 1 us on NVLink, 5 - 10 us on InfiniBand).
- beta: **Inverse Bandwidth** (`1 / Bandwidth`, seconds per byte).
- S: **Payload Size** (bytes).

Notice what this equation reveals:
1. **For small tensors (S is tiny)**: The `2(N-1)alpha` term dominates. Ring All-Reduce suffers because the message must hop `2(N-1)` sequential times around the ring! If `N = 1,024`, you pay `2,046` message latency delays!
2. **For large tensors (S is massive, e.g., `>100 MB`)**: The beta term completely dwarfs the latency term. The link is 100% saturated with raw payload throughput.

### 1.4.5 Tree All-Reduce vs Ring All-Reduce in NCCL:
Because of the latency term `2(N-1)alpha`, NVIDIA's collective library (NCCL) does **not** always use a Ring:
- **Double Binary Tree All-Reduce**: Arranges GPUs into two binary trees. The latency scales logarithmically: `O(\log N * alpha)` rather than `O(N * alpha)`. NCCL uses Tree algorithms for **small payloads** or **very high node counts**.
- **Ring All-Reduce**: Reaches optimal bandwidth utilization (`((N-1) / N) -> 1`). NCCL uses Ring for **large tensor payloads** (e.g., gradient buckets `>25 MB`).
- **NVLS (NVLink SHARP)**: In modern H100/B200 servers, the NVSwitch hardware contains an on-chip arithmetic logic unit (ALU). The switch itself performs the addition at line-rate in hardware, bypassing the ring completely!

---

---

### 1.4.6 Reduce-Scatter & All-Gather (The Dual Primitives)

Notice that Ring All-Reduce is literally:
```text
All-Reduce(X) = All-Gather(Reduce-Scatter(X))
```

- **`reduce_scatter`**:
  Takes an unreduced tensor of size S on each rank, sums them across all ranks, and scatters the result so rank i holds a reduced slice of size `S/N`.
  - Communication volume: `(((N - 1) / N)) S`

- **`all_gather`**:
  Takes a local slice of size `S/N` on each rank and concatenates them across all ranks so every rank holds the full tensor of size S.
  - Communication volume: `(((N - 1) / N)) S`

> [!IMPORTANT]
> In [Sequence Parallelism](/sequence-parallelism/), you will see how **Megatron-LM Sequence Parallelism** achieves breakthrough efficiency by decomposing the All-Reduce into a Reduce-Scatter before LayerNorm and an All-Gather before GEMM, introducing **zero extra communication** while drastically reducing activation memory!

---

