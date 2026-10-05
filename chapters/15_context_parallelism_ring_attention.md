# Context Parallelism & Expert Parallelism (MoE)
> **Ring Attention with Safe Online Softmax, Zigzag Balancing, and Top-K Routing**

> **Reference Papers**:
> - *RingAttention with Block Pipelining for Millions of Tokens* (Liu et al., 2023, [arXiv:2310.01889](https://arxiv.org/abs/2310.01889))
> - *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
> - *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (Fedus et al., 2021, [arXiv:2101.03961](https://arxiv.org/abs/2101.03961))
> - *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (Lepikhin et al., 2020, [arXiv:2006.16668](https://arxiv.org/abs/2006.16668))
> - *Megatron Core Context & Expert Parallelism Architecture* (NVIDIA, 2024)

> [!NOTE]
> **Prerequisites Refresher**:
> Before reading [Context Parallelism & MoE](/context-parallelism/), ensure familiarity with:
> 1. **Tensor Parallelism ([1D Tensor Parallelism](/tensor-parallelism/))**: Sharding weights along hidden dimensions H.
> 2. **Sequence Parallelism ([Sequence Parallelism](/sequence-parallelism/))**: Sharding non-tensor-parallel layers along sequence length S within an NVLink node (`N <= 8`).
> 3. **The Online Softmax Concept**: How FlashAttention incrementally computes `Softmax(Q K^T) V` without materializing the `S * S` attention matrix in HBM.

---

## 1.1. Quadratic Attention Complexity & NVLink Bisection Bounds in Long-Context LLMs

In [Sequence Parallelism](/sequence-parallelism/), we examined **Sequence Parallelism (SP)**. While SP splits LayerNorm and Dropout activations across the sequence dimension, it is structurally coupled to the Tensor Parallelism process group:

> `SP Shard Count == TP Group Size = N <= 8`

Because Tensor Parallelism requires ultra-low-latency intra-node interconnects (`900 GB/s` NVLink), TP cannot scale across multiple physical server nodes without collapsing MFU. Consequently, **Sequence Parallelism is limited to sharding by a factor of 8**.

Now consider training or fine-tuning modern frontier models on **128k, 512k, or 1M token sequences**:
- Let sequence length `S = 131,072` (128k).
- Head dimension `d = 128`, number of query heads `h = 32`.
- Hidden dimension `H = h * d = 4,096`.

### 1.1.1 The Attention Memory Explosion

Even if weights and optimizer states are fully sharded, the intermediate attention matrix for a single sequence layer consumes:

> `Q in shape [B * S * H], K in shape [B * S * H]`

> `Attention Matrix S_attn = Q K^T in shape [B * h * S * S]`

For batch size `B = 1` in FP16 precision (2 bytes):

> `Memory(S_attn) = 1 * 32 * (131,072)^2 * 2 bytes ≈ 1,099.5 GB (1.1 Terabytes!)`

A single attention layer for a single 128k token sequence requires **`1.1 TB` of HBM**, exceeding the capacity of an 80GB H100 GPU by **`13.7 *`**!

Even if we apply FlashAttention (which tiles attention on-chip to avoid materializing the full `S * S` matrix in HBM), the input `Q, K, V` tensors and the feed-forward MLP activations for 128k tokens still consume tens of gigabytes per GPU.

We need a method to shard the sequence dimension **across arbitrary numbers of GPUs (e.g., `C = 16, 32, 64` GPUs across multiple server nodes)** independently of Tensor Parallelism. This is **Context Parallelism (CP)**.

---

## 1.2. Mathematical Foundation: The Online Softmax Recurrence

How can an attention layer be computed across distributed GPUs if the sequence is split into disjoint blocks? Standard softmax requires computing the maximum value across the *entire* sequence to prevent exponential overflow:

> `Softmax(x)_i = ((e^{x_i - max_j x_j}) / (sum_k e^{x_k - max_j x_j}))`

If GPU A holds tokens `[0 ... 4095]` and GPU B holds tokens `[4096 ... 8191]`, how can GPU A compute its softmax denominator before it has seen the tokens on GPU B?

The answer is the **Online Softmax** (Milakov & Gimelshein 2018; Dao et al., 2022).

---

### 1.2.1 Derivation of the Online Softmax Update Equations

Let a query vector `q in shape [1 * d]` attend to key vectors `K in shape [S * d]` and values `V in shape [S * d]`.
Suppose the key-value sequence is partitioned into sequential blocks `K^{(1)}, K^{(2)}, ..., K^{(C)}`, each of size `S_block * d`.

Let the attention scores for block k be:

> `S^{(k)} = (q (K^{(k)})^T / sqrt(d)) in shape [1 * S_block]`

#### 1.2.1.1 Step 1: Running Row Maximum
Let `m^{(k-1)}` be the maximum score observed up through block k-1:

> `m^{(0)} = -infinity`

> `m^{(k)}_local = max(j) S_j^{(k)}`

> `m^{(k)} = max(m^{(k-1)}, m^{(k)}_local)`

#### 1.2.1.2 Step 2: The Exponential Rescaling Factor
When transitioning from old maximum `m^{(k-1)}` to new maximum `m^{(k)}`, all previously computed unnormalized exponentials must be corrected by a factor `alpha^{(k)}`:

> `alpha^{(k)} = e^{m^{(k-1)} - m^{(k)}} <= 1.0`

#### 1.2.1.3 Step 3: Running Normalization Denominator (l)
Let `l^{(k-1)}` be the sum of unnormalized exponentials up to block k-1, scaled relative to `m^{(k-1)}`:

> `l^{(0)} = 0`

> `l^{(k)} = l^{(k-1)} * alpha^{(k)} + sum(j=1 to S_{block)} e^{S_j^{(k)} - m^{(k)}}`

#### 1.2.1.4 Step 4: Running Output Accumulator (O)
Let `O^{(k-1)}` be the true, normalized attention output up to block k-1:

> `O^{(k-1)} = ((sum(j in blocks 1 ... k-1) e^{S_j - m^{(k-1)}} V_j) / (l^{(k-1)}))`

To incorporate block k without recomputing past blocks:
1. Rescale the previous numerator: `Num^{(k-1)} * alpha^{(k)} = (O^{(k-1)} * l^{(k-1)}) * alpha^{(k)}`.
2. Add the new block's contribution: `sum(j=1 to S_{block)} e^{S_j^{(k)} - m^{(k)}} V_j^{(k)}`.
3. Divide by the new denominator `l^{(k)}`:

> `O^{(k)} = O^{(k-1)} ( ((l^{(k-1)} * alpha^{(k)}) / l^{(k)}) ) + ((sum(j=1 to S_{block)} e^{S_j^{(k)} - m^{(k)}} V_j^{(k)}) / l^{(k)})`

> `O^{(C)} == Softmax(q K^T / sqrt(d)) V (Exact mathematical identity!)`

This recurrence allows a GPU to maintain a running accumulator O, updating it incrementally whenever a new `K, V` block arrives, with **zero loss of numerical precision**.

---

## 1.3. The Ring Attention Algorithm

Liu et al. (2023) combined Online Softmax with a **circular peer-to-peer (P2P) ring topology** across C Context Parallel GPUs:

1. The sequence of length S is divided into C equal chunks of size `S_local = S / C`.
2. Rank `i in {0, 1, ..., C-1}` permanently stores its local query block:

> `Q_i = Q[ i * S_local : (i+1) * S_local ]`

3. Rank i initializes its key and value buffers with its local chunk:

> `K_i^{(0)} = K[ i * S_local : (i+1) * S_local ], V_i^{(0)} = V[ i * S_local : (i+1) * S_local ]`

4. **The Bucket Brigade**: Ranks arrange themselves in a ring:

> `Rank i \xrightarrow{sends (K, V)} Rank (i + 1) % C`

> `Rank i \xleftarrow{receives (K, V)} Rank (i - 1) % C`

---

### 1.3.1 Step-by-Step 4-GPU Execution Trace

Let `C = 4`. Total sequence length `S = 16,384` (`4,096` tokens per GPU):

```
                   RING ATTENTION EXECUTION TIMELINE (C = 4)
                   
Round 0:
  GPU 0 holds Q0. Computes Attn(Q0, K0, V0). Simultaneously sends (K0, V0) ──> GPU 1
  GPU 1 holds Q1. Computes Attn(Q1, K1, V1). Simultaneously sends (K1, V1) ──> GPU 2
  GPU 2 holds Q2. Computes Attn(Q2, K2, V2). Simultaneously sends (K2, V2) ──> GPU 3
  GPU 3 holds Q3. Computes Attn(Q3, K3, V3). Simultaneously sends (K3, V3) ──> GPU 0

Round 1:
  GPU 0 holds Q0. Receives (K3, V3). Updates Attn(Q0, K3, V3). Sends (K3, V3) ──> GPU 1
  GPU 1 holds Q1. Receives (K0, V0). Updates Attn(Q1, K0, V0). Sends (K0, V0) ──> GPU 2
  GPU 2 holds Q2. Receives (K1, V1). Updates Attn(Q2, K1, V1). Sends (K1, V1) ──> GPU 3
  GPU 3 holds Q3. Receives (K2, V2). Updates Attn(Q3, K2, V2). Sends (K2, V2) ──> GPU 0

Round 2:
  GPU 0 holds Q0. Receives (K2, V2). Updates Attn(Q0, K2, V2). Sends (K2, V2) ──> GPU 1
  GPU 1 holds Q1. Receives (K3, V3). Updates Attn(Q1, K3, V3). Sends (K3, V3) ──> GPU 2
  GPU 2 holds Q2. Receives (K0, V0). Updates Attn(Q2, K0, V0). Sends (K0, V0) ──> GPU 3
  GPU 3 holds Q3. Receives (K1, V1). Updates Attn(Q3, K1, V1). Sends (K1, V1) ──> GPU 0

Round 3:
  GPU 0 holds Q0. Receives (K1, V1). Updates Attn(Q0, K1, V1). (Final Step: no send)
  GPU 1 holds Q1. Receives (K2, V2). Updates Attn(Q1, K2, V2). (Final Step: no send)
  GPU 2 holds Q2. Receives (K3, V3). Updates Attn(Q2, K3, V3). (Final Step: no send)
  GPU 3 holds Q3. Receives (K0, V0). Updates Attn(Q3, K0, V0). (Final Step: no send)

Result: After C = 4 rounds, every Q_i has attended to all K_0..K_3 and V_0..V_3!
```

---

### 1.3.2 Comm-Compute Overlap: 100% Communication Hiding

In Ring Attention, while the GPU's Tensor Cores execute the FlashAttention kernel on the current `(K_curr, V_curr)` block, the GPU's Network Interface Card (NIC) transmits `(K_curr, V_curr)` and receives `(K_next, V_next)` asynchronously:

```
Compute Stream (Tensor Cores):    [ FlashAttn(Q, K_0, V_0) ]  [ FlashAttn(Q, K_3, V_3) ]
                                              │                           │
Sync Point:                                   ▼ Wait Recv                 ▼ Wait Recv
Comm Stream (NIC / InfiniBand):   [  P2P Send/Recv K_0, V_0  ] [  P2P Send/Recv K_3, V_3  ]
```

#### 1.3.2.1 The Hiding Condition: Forward vs Backward Pass

Let `S_local = 4,096`, `d = 128`, `h = 32`.
- **Forward-Only Compute Arithmetic**:
  Attention forward pass involves two primary GEMMs (`Q K^T` attention scores and `Softmax(S) * V` projection), each taking `2 B h S_local^2 d` floating-point operations:

> `FLOPs_fwd = 4 * B * h * S_local^2 * d = 4 * 1 * 32 * (4,096)^2 * 128 ≈ 2.75 * 10^11 FLOPs (275 GFLOPs)`

  On an NVIDIA H100 SXM5 running at 989 TFLOPS (dense BF16 Tensor Cores):

> `T[compute, fwd] ≈ ((275 * 10^9) / (989 * 10^12)) ≈ 0.278 ms`

- **Full Iteration (Forward + Backward) Compute Arithmetic**:
  In backpropagation, computing gradients `(d Loss / d Q)`, `(d Loss / d K)`, and `(d Loss / d V)` requires approximately `2 *` forward FLOPs (4 backward GEMMs):

> `FLOPs_{fwd+bwd} ≈ 3 * FLOPs_fwd ≈ 8.25 * 10^11 FLOPs (825 GFLOPs)`

> `T[compute, bwd] ≈ 0.556 ms`

- **Communication Volume** (sending K and V in BF16):

> `Bytes Transmitted = 2 * (2 bytes) * B * S_local * (h * d) = 4 * 1 * 4,096 * 4,096 = 67.1 MB`

  Over standard 400 Gbps (`50 GB/s`) InfiniBand links:

> `T_comm ≈ ((67.1 * 10^6) / (50 * 10^9)) ≈ 1.34 ms`

  Over intra-node NVLink (`900 GB/s`):

> `T_comm ≈ ((67.1 * 10^6) / (900 * 10^9)) ≈ 0.074 ms`

**Conclusion**: Inside an 8-GPU node (`900 GB/s` NVLink), `T_comm << T_compute`, achieving **100% communication hiding** in both forward and backward passes. Across multi-node clusters over InfiniBand, backpropagation provides a **`3 *` higher compute-to-communication ratio**, and pairing with larger local chunks (`S_local >= 8,192`) guarantees that communication latency is completely masked behind attention arithmetic!

---

### 1.3.3 The Causal Masking Problem & Zigzag Attention

In autoregressive language models, attention is strictly causal (`S_ij = -infinity` for `j > i`).
In naive Ring Attention:
- GPU 0 holds tokens `[0 ... 4095]`: it can only attend to tokens `[0 ... 4095]` (its own block). In rounds 1, 2, and 3, all incoming keys have indices `j > 4095`, meaning GPU 0 performs **zero compute and sits completely idle!**
- GPU 3 holds tokens `[12288 ... 16383]`: it must attend to all blocks from rounds 0, 1, 2, and 3.
- **Bubble Inefficiency**: A naive causal ring creates a 50% bubble overhead because half of the lower-triangular blocks are masked out!

#### 1.3.3.1 Megatron Core's Solution: Zigzag / Striped Ring Attention

Megatron Core eliminates this 50% idle bubble through **Zigzag / Striped Ring Attention**:
Instead of assigning a single contiguous chunk of `S/C` tokens to each GPU, the sequence is split into 2C smaller chunks (`2 * 4 = 8` chunks for `C=4`).
Each physical GPU is assigned **two symmetrically paired chunks**: an early chunk i and a late chunk 2C - 1 - i:

```
                 Zigzag Token Assignment (C = 4 GPUs, 8 Chunks)
                 
Sequence Chunks:   [ Chunk 0 ] [ Chunk 1 ] [ Chunk 2 ] [ Chunk 3 ] [ Chunk 4 ] [ Chunk 5 ] [ Chunk 6 ] [ Chunk 7 ]
Tokens:            0 .. B-1    B .. 2B-1   2B .. 3B-1  3B .. 4B-1  4B .. 5B-1  5B .. 6B-1  6B .. 7B-1  7B .. 8B-1
                   └───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
GPU Assignment:       GPU 0       GPU 1       GPU 2       GPU 3       GPU 3       GPU 2       GPU 1       GPU 0
```

#### 1.3.3.2 Why This Restores Perfect Causal Balance

In a causal matrix, Chunk k can only attend to Chunks `j <= k`:

| GPU | Early Chunk (i) | Late Chunk (2C - 1 - i) | Valid Key Chunks for Early | Valid Key Chunks for Late | Total Active Compute Blocks |
|---|---|---|---|---|---|
| **GPU 0** | Chunk 0 | Chunk 7 | Chunk 0 (1 block) | Chunks 0..7 (8 blocks) | **`1 + 8 = 9` blocks** |
| **GPU 1** | Chunk 1 | Chunk 6 | Chunks 0..1 (2 blocks) | Chunks 0..6 (7 blocks) | **`2 + 7 = 9` blocks** |
| **GPU 2** | Chunk 2 | Chunk 5 | Chunks 0..2 (3 blocks) | Chunks 0..5 (6 blocks) | **`3 + 6 = 9` blocks** |
| **GPU 3** | Chunk 3 | Chunk 4 | Chunks 0..3 (4 blocks) | Chunks 0..4 (5 blocks) | **`4 + 5 = 9` blocks** |

> `Active Compute Blocks per GPU == 9 blocks (100% Perfect Load Balance!)`

> [!TIP]
> **Why This Matters**:
> In naive causal Ring Attention, GPU 0 does 1 block while GPU 3 does 4 blocks, wasting 50% of your cluster's FLOP capacity. Zigzag Attention guarantees that **every single GPU executes the exact same number of floating-point operations**, entirely reclaiming the causal bubble!

---

