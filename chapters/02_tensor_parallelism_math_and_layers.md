# Chapter 02: 1D Tensor Parallelism & Linear Operator Sharding
> **Column/Row GEMMs, Parallel Cross-Entropy, and Tensor Core Boundary Alignment**

> **Reference Paper**: *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (Shoeybi et al., NVIDIA 2019, [arXiv:1909.08053](https://arxiv.org/abs/1909.08053))

---

## 1. The Core Problem: Why Model Parallelism Failed Before Megatron

Before Megatron-LM was introduced by NVIDIA in 2019, model parallelism was widely considered impractical for training large neural networks.

Consider a standard 2-layer Multi-Layer Perceptron (MLP) found in every Transformer block:
$$Y = \text{GELU}(X W_1) W_2$$

Where:
- $X \in \mathbb{R}^{B \times H}$ ($B$ is batch size $\times$ sequence length, $H$ is hidden dimension)
- $W_1 \in \mathbb{R}^{H \times 4H}$
- $W_2 \in \mathbb{R}^{4H \times H}$

### The Naive Row-Parallel Trap:
Suppose you naively try to split $W_1$ across $N = 2$ GPUs along its rows (input dimension):
$$W_1 = \begin{bmatrix} W_{1,1} \\ W_{1,2} \end{bmatrix}, \quad \text{where } W_{1,1}, W_{1,2} \in \mathbb{R}^{\frac{H}{2} \times 4H}$$

To multiply $X$ by this row-split $W_1$, the input $X$ must also be split along its columns:
$$X = \begin{bmatrix} X_1 & X_2 \end{bmatrix}, \quad \text{where } X_1, X_2 \in \mathbb{R}^{B \times \frac{H}{2}}$$

Each GPU computes its local matrix product:
- GPU 0 computes: $Z_1 = X_1 W_{1,1} \in \mathbb{R}^{B \times 4H}$
- GPU 1 computes: $Z_2 = X_2 W_{1,2} \in \mathbb{R}^{B \times 4H}$

The true intermediate activation is the sum of these outer products:
$$Z = Z_1 + Z_2$$

Now comes the fatal flaw: **We must apply the non-linear activation $\text{GELU}(Z)$**.
Because non-linear functions do not distribute over addition:
$$\text{GELU}(Z_1 + Z_2) \neq \text{GELU}(Z_1) + \text{GELU}(Z_2)$$

**The Catastrophic Result**:
GPU 0 and GPU 1 **cannot evaluate GELU locally**! They must pause, synchronize, and execute an expensive **All-Reduce communication** across the network just to compute $Z = Z_1 + Z_2$ before either GPU can compute GELU!
Then, to execute the second layer $W_2$, another communication synchronization is required.

In an 80-layer model, communicating across the network on every single matrix multiply caused the GPU's compute cores to spend $>75\%$ of their time waiting for network packets. Model parallelism was considered dead on arrival.

---

## 2. Megatron's Breakthrough: Complementary GEMM Factoring

Shoeybi et al. (2019) solved this with a brilliant algebraic insight:
> **If you pair a Column-Parallel linear layer with a Row-Parallel linear layer, the non-linear activation is completely trapped between them, requiring ZERO communication!**

### 2.1 The Algebraic and Geometric Proof

Let us write out the exact block-matrix multiplication for the two splitting strategies:

#### Step 1: Column-Parallel GEMM (Layer 1)
Slice $W_1 \in \mathbb{R}^{H \times 4H}$ along its **columns** into $N = 2$ blocks:
$$W_1 = \begin{bmatrix} W_{1,1} & W_{1,2} \end{bmatrix}, \quad \text{where each } W_{1,i} \in \mathbb{R}^{H \times \frac{4H}{2}}$$

Every GPU holds the **full, identical input** $X \in \mathbb{R}^{B \times H}$.
Multiplying $X$ by the column-sliced matrix yields:
$$X W_1 = X \begin{bmatrix} W_{1,1} & W_{1,2} \end{bmatrix} = \begin{bmatrix} X W_{1,1} & X W_{1,2} \end{bmatrix} = \begin{bmatrix} Z_{1,1} & Z_{1,2} \end{bmatrix}$$

Notice what just happened:
- GPU 0 computes $Z_{1,1} = X W_{1,1}$ locally.
- GPU 1 computes $Z_{1,2} = X W_{1,2}$ locally.
- **Communication Required: EXACTLY ZERO!**

---

#### Step 2: The Non-Linearity Property
Now we apply $\text{GELU}$ to the partitioned output $\begin{bmatrix} Z_{1,1} & Z_{1,2} \end{bmatrix}$.
Because GELU is an **elementwise function** (it operates independently on each individual number without mixing elements across columns):

$$\text{GELU}\left(\begin{bmatrix} Z_{1,1} & Z_{1,2} \end{bmatrix}\right) = \begin{bmatrix} \text{GELU}(Z_{1,1}) & \text{GELU}(Z_{1,2}) \end{bmatrix} = \begin{bmatrix} A_{1,1} & A_{1,2} \end{bmatrix}$$

- GPU 0 evaluates $\text{GELU}(Z_{1,1})$ on its local memory.
- GPU 1 evaluates $\text{GELU}(Z_{1,2})$ on its local memory.
- **Communication Required: EXACTLY ZERO!**

---

#### Step 3: Row-Parallel GEMM (Layer 2)
Now we must multiply the intermediate activations $A = \begin{bmatrix} A_{1,1} & A_{1,2} \end{bmatrix}$ by the second weight matrix $W_2 \in \mathbb{R}^{4H \times H}$.
Notice that $A$ is naturally partitioned **column-wise** across the two GPUs!

Therefore, we slice $W_2$ along its **rows**:
$$W_2 = \begin{bmatrix} W_{2,1} \\ W_{2,2} \end{bmatrix}, \quad \text{where each } W_{2,i} \in \mathbb{R}^{\frac{4H}{2} \times H}$$

Now perform the block matrix multiplication:
$$Y = A W_2 = \begin{bmatrix} A_{1,1} & A_{1,2} \end{bmatrix} \begin{bmatrix} W_{2,1} \\ W_{2,2} \end{bmatrix} = A_{1,1} W_{2,1} + A_{1,2} W_{2,2}$$

Look at the symmetry:
- GPU 0 already holds $A_{1,1}$. It multiplies it by its local row slice $W_{2,1}$ to get partial output $Y_1 = A_{1,1} W_{2,1}$.
- GPU 1 already holds $A_{1,2}$. It multiplies it by its local row slice $W_{2,2}$ to get partial output $Y_2 = A_{1,2} W_{2,2}$.
- Both $Y_1$ and $Y_2$ have shape $[B, H]$.

To obtain the true final output $Y = Y_1 + Y_2$, we execute **ONE All-Reduce (SUM)** across the GPUs!
$$Y = \text{All-Reduce}(Y_1 + Y_2)$$

### The Revolutionary Result:
In an entire 2-layer MLP block with massive matrix multiplications and non-linearities:
- Forward pass communication: **Exactly ONE All-Reduce!**
- Backward pass communication: **Exactly ONE All-Reduce!**

---

```
                      Megatron MLP Tensor Flow
                      
               Input X [Batch, Seq, Hidden]
                  (Replicated on all ranks)
                             │
                  ┌──────────┴──────────┐
                  │ Operator f (Identity)
                  ▼                     ▼
               Rank 0                Rank 1
          W1_1 [H, 4H/2]        W1_2 [H, 4H/2]
                  │                     │
             (Local GEMM)          (Local GEMM)
                  │                     │
                  ▼                     ▼
          Z1_1 [B, S, 2H]       Z1_2 [B, S, 2H]
                  │                     │
             (Local GELU)          (Local GELU)
                  │                     │
                  ▼                     ▼
          A1_1 [B, S, 2H]       A1_2 [B, S, 2H]
                  │                     │
          W2_1 [2H, H]          W2_2 [2H, H]
                  │                     │
             (Local GEMM)          (Local GEMM)
                  │                     │
                  ▼                     ▼
          Z2_1 [B, S, H]        Z2_2 [B, S, H]
                  │                     │
                  └──────────┬──────────┘
                             │
                Operator g (All-Reduce SUM)
                             ▼
             Output Y = Z2_1 + Z2_2 [B, S, H]
                  (Replicated on all ranks)
```

### Why this is mathematically optimal:
1. $W_1$ is **Column-Parallel** ($H \to 4H/N$).
2. The intermediate activation $Z_1$ is partitioned: rank $i$ holds $Z_{1,i}$.
3. $\text{GELU}$ is an **elementwise function**:
   $$\text{GELU}\Big([Z_{1,1}, Z_{1,2}]\Big) = \Big[\text{GELU}(Z_{1,1}), \text{GELU}(Z_{1,2})\Big]$$
   Each rank evaluates GELU on its own slice **without any network communication**!
4. $W_2$ is **Row-Parallel** ($4H/N \to H$). It directly consumes the partitioned activations $A_{1,i}$.
> [!TIP]
> **Why This Matters**:
> If you reversed the order (Row-Parallel followed by Column-Parallel), the non-linear GELU activation would be forced to operate on unsummed partial dot-products. To fix this, you would have to insert an extra All-Reduce *before* GELU, doubling your communication cost per layer from 2 All-Reduces to 4 All-Reduces! The Column-then-Row factoring is the only mathematical permutation that allows non-linear activations to execute locally without communication.

---

## 4. The Megatron Multi-Head Attention Block

In Multi-Head Attention (MHA), the hidden dimension $H$ is split across $h$ attention heads:
$$d_{head} = \frac{H}{h}$$

Megatron partitions the heads across the $N$ tensor parallel GPUs:
$$h_{local} = \frac{h}{N}$$

```
                Megatron Causal Self-Attention
                
               Input X [Batch, Seq, Hidden]
                  (Replicated on all ranks)
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
               Rank 0                Rank 1
           (Heads 0 .. h/2-1)    (Heads h/2 .. h-1)
                  │                     │
          ColumnParallel QKV    ColumnParallel QKV
                  │                     │
                  ▼                     ▼
            Q1, K1, V1            Q2, K2, V2
                  │                     │
          Compute Attention     Compute Attention
          (Local Heads ONLY)    (Local Heads ONLY)
                  │                     │
                  ▼                     ▼
             Head Out 1            Head Out 2
                  │                     │
                  └──────────┬──────────┘
                             │
                    RowParallel Proj
                             │
                Operator g (All-Reduce SUM)
                             ▼
               Output [Batch, Seq, Hidden]
                  (Replicated on all ranks)
```

### Detailed Execution Steps:
1. **QKV Projection (Column Parallel)**:
   - A single weight matrix $W_{QKV} \in \mathbb{R}^{H \times 3H}$ is split column-wise.
   - Each rank projects its input to $\left[Q_i, K_i, V_i\right]$ of size $B \times S \times \left(3 \times h_{local} \times d_{head}\right)$.
   - **Communication: 0.**

2. **Self-Attention Computation (Local)**:
   - Each rank computes scaled dot-product attention for its own heads:
     $$\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_{head}}}\right) V_i$$
   - Since attention heads are completely independent, **communication is 0!**

3. **Output Projection (Row Parallel)**:
   - The concatenated outputs of all heads are projected back to $H$ using Row Parallel linear layer $W_{proj}$.
   - Operator $g$ executes **one All-Reduce** to sum the projected vectors.

### Total Forward Communications Per Transformer Block:
$$\text{Attention All-Reduce (1)} + \text{MLP All-Reduce (1)} = \mathbf{2 \text{ All-Reduces per Block}}$$

---

## 5. The Autograd Conjugate Operators: $f$ and $g$

Megatron formalized the communication using two symbolic operators in the computation graph:

$$\text{Block}(X) = X + \mathbf{g}\Big(\text{RowProj}\big(\text{Attn}(\text{ColQKV}(\mathbf{f}(X)))\big)\Big)$$

### Mathematical Derivation of Gradients:

#### Operator $f$ (Identity in Forward):
$$f(X) = X$$
During backward propagation, by chain rule:
$$\frac{\partial L}{\partial X} = \sum_{i=1}^N \frac{\partial L}{\partial Y_i}$$
Since rank $i$ holds the local gradient $\frac{\partial L}{\partial Y_i}$, to compute $\frac{\partial L}{\partial X}$, we **must All-Reduce (SUM) the incoming gradients**:
$$f^*(\text{grad}) = \text{All-Reduce}(\text{grad})$$

#### Operator $g$ (All-Reduce in Forward):
$$g(Y_1, \dots, Y_N) = \sum_{i=1}^N Y_i$$
Since every rank receives the same total output $Y$, the gradient with respect to each input slice is identical:
$$\frac{\partial L}{\partial Y_i} = \frac{\partial L}{\partial Y}$$
Thus, no communication is required during the backward pass:
$$g^*(\text{grad}) = \text{grad}$$

---

## 6. Vocabulary Parallelism & Parallel Cross Entropy

When training language models with massive vocabularies ($V = 32{,}000$ to $256{,}000$), the token embedding table and final language modeling head become major memory bottlenecks:
$$\text{Params} = V \times H$$
For $V = 128{,}000$ and $H = 8{,}192$, the embedding matrix alone consumes **$2\text{ GB}$ in FP16**. More critically, computing logits produces a massive activation tensor:
$$\text{Logits Memory} = B \times T \times V \times 2\text{ bytes}$$
For $B = 8, T = 4{,}096, V = 128{,}000$, the logits tensor is **$8.4\text{ GB}$ per microbatch**, often triggering Out-Of-Memory (OOM) errors!

### 6.1 `VocabParallelEmbedding`
We partition the vocabulary dimension across the $N$ ranks:
$$\text{Vocab Partition Size} = \left\lceil \frac{V}{N} \right\rceil$$

Rank $i$ holds token IDs in range:
$$[\text{start\_idx}_i, \text{end\_idx}_i) = \left[i \times \frac{V}{N}, (i+1) \times \frac{V}{N}\right)$$

```
Token Input Tensor: [ token_id = 45 ]
Rank 0 holds vocab [0 .. 31]:   Lookup gives ZERO vector
Rank 1 holds vocab [32 .. 63]:  Lookup gives Embedding Vector E_45
Rank 2 holds vocab [64 .. 95]:  Lookup gives ZERO vector
Rank 3 holds vocab [96 .. 127]: Lookup gives ZERO vector

All-Reduce (SUM) across all ranks -> All ranks obtain E_45!
```

```python
"""
VocabParallelEmbedding Implementation.
Shards embedding table along vocab dimension across tensor parallel ranks.
"""
import torch
import torch.nn as nn
import torch.distributed as dist

class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, tp_group=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_group = tp_group
        
        self.tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0
        
        # Partition vocab size
        self.vocab_start_index, self.vocab_end_index, self.padded_vocab_size = self._get_vocab_range()
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        
        # Local embedding weight slice (aligned to Tensor Core boundaries)
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def _get_vocab_range(self):
        # NVIDIA Tensor Core alignment: GEMM outer dimensions must be multiples
        # of 64 or 128 elements to maximize systolic array throughput and avoid unaligned P2P/All-Reduce.
        alignment = self.tp_world_size * 64
        padded_vocab_size = int(math.ceil(self.num_embeddings / alignment)) * alignment
        per_partition = padded_vocab_size // self.tp_world_size
        start = self.tp_rank * per_partition
        end = start + per_partition
        return start, end, padded_vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Build mask for tokens belonging to this rank's partition
        input_mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
        # Shift input to local index range [0, num_embeddings_per_partition)
        local_ids = input_ids - self.vocab_start_index
        local_ids[input_mask] = 0  # Dummy index to prevent out-of-bounds error
        
        # Local lookup
        output_parallel = nn.functional.embedding(local_ids, self.weight)
        # Mask out embeddings for tokens that do NOT belong to this rank
        output_parallel[input_mask] = 0.0
        
        # All-Reduce (SUM) across TP ranks to reassemble full embeddings
        if self.tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(output_parallel, group=self.tp_group, op=dist.ReduceOp.SUM)
            
        return output_parallel
```

---

### 6.2 `ParallelCrossEntropyLoss`: Softmax Without Gathering Logits

In standard PyTorch, cross-entropy is:
$$\mathcal{L} = -\log \left(\frac{e^{z_{target}}}{\sum_{j=1}^V e^{z_j}}\right) = -z_{target} + \log \left(\sum_{j=1}^V e^{z_j}\right)$$

In Megatron, each rank $i$ only computes logits for its local slice of vocab $V_i$:
$$z_{local} \in \mathbb{R}^{B \times T \times \frac{V}{N}}$$

Megatron computes the loss in parallel using **three lightweight All-Reduces on scalars**:

```
Step 1: Local Max
  Each rank finds: m_i = max(z_local)
  All-Reduce (MAX) gives global max: M = max_i(m_i)

Step 2: Local Exp Sum
  Each rank computes: s_i = sum(exp(z_local - M))
  All-Reduce (SUM) gives global denominator: S = sum_i(s_i)

Step 3: Target Logit Extraction
  Rank holding target token extracts z_target (others have 0).
  All-Reduce (SUM) shares z_target across ranks.

Final Loss:
  Loss = - (z_target - M) + log(S)
```

**Memory Saved**: The full $(B \times T \times V)$ tensor is **NEVER materialized in GPU memory**!

```python
"""
ParallelCrossEntropyLoss Implementation.
Computes cross-entropy loss over partitioned logits without gathering the full [B, S, V] tensor.
"""
class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group=None):
        # vocab_parallel_logits: [B, S, V/N]
        # target: [B, S]
        tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0

        # Step 1: Global Maximum Logit for numerical stability
        local_max = torch.max(vocab_parallel_logits, dim=-1, keepdim=True)[0]
        global_max = local_max.clone()
        if tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(global_max, group=tp_group, op=dist.ReduceOp.MAX)

        # Step 2: Sum of Exponentials
        shifted_logits = vocab_parallel_logits - global_max
        local_sum_exp = torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)
        global_sum_exp = local_sum_exp.clone()
        if tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(global_sum_exp, group=tp_group, op=dist.ReduceOp.SUM)

        # Step 3: Extract target logit
        per_partition = vocab_parallel_logits.size(-1)
        vocab_start = tp_rank * per_partition
        vocab_end = vocab_start + per_partition
        
        target_mask = (target >= vocab_start) & (target < vocab_end)
        local_target = target.clone() - vocab_start
        local_target[~target_mask] = 0

        # Gather target logit from local partition
        target_logit = torch.gather(vocab_parallel_logits, dim=-1, index=local_target.unsqueeze(-1)).squeeze(-1)
        target_logit[~target_mask] = 0.0

        if tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(target_logit, group=tp_group, op=dist.ReduceOp.SUM)

        # Loss = log(sum(exp(z - max))) + max - target_logit
        loss = torch.log(global_sum_exp.squeeze(-1)) + global_max.squeeze(-1) - target_logit

        # Save tensors for backward pass
        ctx.save_for_backward(shifted_logits, global_sum_exp, target_mask, local_target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        shifted_logits, global_sum_exp, target_mask, local_target = ctx.saved_tensors
        # Softmax probabilities: exp(z - max) / sum(exp(z - max))
        softmax = torch.exp(shifted_logits) / global_sum_exp
        
        # dL / dz = softmax - 1 (for target token)
        grad_input = softmax
        if target_mask.any():
            grad_input.scatter_add_(-1, local_target.unsqueeze(-1), -torch.ones_like(grad_input))
            
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None, None


class ParallelCrossEntropyLoss(nn.Module):
    def __init__(self, tp_group=None):
        super().__init__()
        self.tp_group = tp_group

    def forward(self, vocab_parallel_logits, target):
        return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, self.tp_group).mean()
```

---

## 7. Numerical Subtleties & Bias Handling in `RowParallelLinear`

A subtle bug in tensor-parallel implementations occurs in bias handling:

$$Y = X W + b$$

In RowParallelLinear:
$$Y = \left(\sum_{i=1}^N X_i W_i\right) + b$$

If each rank computes $Z_i = X_i W_i + b$, and then performs `All-Reduce(SUM)`, the resulting tensor will be:
$$\sum_{i=1}^N (X_i W_i + b) = \left(\sum_{i=1}^N X_i W_i\right) + \mathbf{N \times b}$$
The bias is added $N$ times!

### The Two Correct Fixes:
1. **Divide bias by $N$**: Add $b/N$ before All-Reduce (introduces floating-point roundoff error).
2. **Add bias AFTER All-Reduce (Megatron's method)**:
   ```python
   # Compute GEMM without bias
   output_parallel = F.linear(input_parallel, self.weight, None)
   # All-reduce partial sums
   output = reduce_from_model_parallel_region(output_parallel)
   # Add bias once to the final sum
   if self.bias is not None:
       output = output + self.bias
   ```

---

## 8. Grouped Query Attention (GQA) & Multi-Query Attention (MQA) with TP

Modern LLMs (Llama 3, Mistral, Qwen) use **Grouped Query Attention (GQA)** instead of standard Multi-Head Attention (MHA).

In GQA, query heads and key/value heads have different counts:
- Query heads: $h_Q$ (e.g., 32)
- Key/Value heads: $h_{KV}$ (e.g., 8)
- **Group size**: $G = h_Q / h_{KV}$ (e.g., 4)

Each KV head is shared across $G$ query heads.

### Why GQA Changes the Tensor Parallel Constraint:

In standard MHA, any $N$ that evenly divides $h$ works:
$$h \% N = 0$$

In GQA, we must also ensure KV heads can be partitioned evenly:
$$h_{KV} \% N = 0$$

If $h_{KV} = 8$ and $N = 4$, each rank holds $h_Q / N = 8$ query heads and $h_{KV} / N = 2$ KV heads.

### The Constraint Matrix:

| Config | $h_Q$ | $h_{KV}$ | Max TP | Per-Rank Q Heads | Per-Rank KV Heads |
|---|---|---|---|---|---|
| **MHA** | 32 | 32 | 8 (if node has 8) | 4 | 4 |
| **GQA (G=4)** | 32 | 8 | 8 | 4 | 1 |
| **MQA** | 32 | 1 | **1** (Cannot TP!) | 32 | 1 |

> [!IMPORTANT]
> **Multi-Query Attention (MQA)** has exactly **1 KV head**. You **cannot** apply Tensor Parallelism to MQA attention without special all-gather tricks, because there's no KV head to shard! This is why modern production models use GQA (minimum $h_{KV} = N$) rather than MQA.

---

## 9. Weight Initialization Scaling for Tensor Parallelism

A subtle but critical detail: **weight initialization must be adjusted for TP.**

In a standard single-GPU model, residual projections (attention output projection and MLP FC2) are typically initialized with reduced variance to prevent residual activation blow-up:
$$\text{Std}(W_{out}) = \frac{0.02}{\sqrt{2 \times L}}$$

Where $L$ is the number of transformer layers (the $2L$ accounts for $2$ residual connections per block).

In Megatron with Tensor Parallelism, `RowParallelLinear` projects from $(4H / N) \to H$ instead of $4H \to H$.
The **fan-in** is $4H/N$ rather than $4H$, which changes the He/Xavier initialization normalization.

Megatron handles this by scaling with respect to the **full un-partitioned fan-in**:
```python
# In ColumnParallelLinear / RowParallelLinear __init__:
# WRONG (uses partitioned fan-in — gives wrong variance):
std = math.sqrt(2.0 / in_features_per_partition)

# CORRECT (Megatron always scales w.r.t. the FULL global model fan-in):
std = math.sqrt(2.0 / (in_features_global * num_layers_factor))
```

Not doing this correctly causes the activation variance to diverge by a factor of $\sqrt{N}$ across GPUs!

---

## 10. Worked Numerical Example: Exact Tensor Shapes at Every Step (TP=2)

Let's trace the complete forward pass of one Transformer block with exact tensor shapes:

**Model Config**: $B=2, S=4, H=8, h=4, d_{head}=2, \text{TP}=2$

```
INPUT: X [B=2, S=4, H=8]  <-- Identical on both Rank 0 and Rank 1
         │
         ├── Operator f (Identity)
         │
         ▼ RANK 0                               ▼ RANK 1
   W_QKV [8, 12] (heads 0,1)            W_QKV [8, 12] (heads 2,3)
   Output QKV [2, 4, 12]                Output QKV [2, 4, 12]
   Q0 [2, 4, 4], K0 [2, 4, 4]          Q1 [2, 4, 4], K1 [2, 4, 4]
   V0 [2, 4, 4]                         V1 [2, 4, 4]
         │                                      │
   AttnScores0 = Q0 @ K0.T              AttnScores1 = Q1 @ K1.T
   [2, h_local=2, S=4, S=4]             [2, h_local=2, S=4, S=4]
         │                                      │
   AttnOut0 = Softmax @ V0              AttnOut1 = Softmax @ V1
   [2, h_local=2, 4, 2]                 [2, h_local=2, 4, 2]
         │                                      │
   AttnOut0.reshape [2, 4, 4]           AttnOut1.reshape [2, 4, 4]
         │                                      │
   W_proj0 [4, 8]                       W_proj1 [4, 8]
   Z_proj0 = AttnOut0 @ W_proj0         Z_proj1 = AttnOut1 @ W_proj1
   [2, 4, 8]                            [2, 4, 8]
         │                                      │
         └──────── Operator g: All-Reduce (SUM) ┘
                             │
              Y_attn = Z_proj0 + Z_proj1 [2, 4, 8]  <-- Full H=8 on both ranks!
```

This shows concretely how the RowParallelLinear output projections `Z_proj0 [2,4,8]` and `Z_proj1 [2,4,8]` are **summed** via All-Reduce to yield the final output. Neither rank holds a partial slice of $H$; both receive the full $H=8$ dimensional vector.

---

## 11. Summary & Checklist

| Component | Parallelism Type | Local Dimension | Forward Communication | Backward Communication |
|---|---|---|---|---|
| **Token Embedding** | Vocab Parallel | $V/N \times H$ | All-Reduce (SUM) | Identity |
| **QKV Projection** | Column Parallel | $H \times (3H/N)$ | 0 | All-Reduce (SUM) |
| **Self-Attention** | Head Partitioned | $h/N \text{ heads}$ | 0 | 0 |
| **Attention Proj** | Row Parallel | $(H/N) \times H$ | All-Reduce (SUM) | Identity |
| **MLP FC1** | Column Parallel | $H \times (4H/N)$ | 0 | All-Reduce (SUM) |
| **MLP GELU** | Local Elementwise | $4H/N$ | 0 | 0 |
| **MLP FC2** | Row Parallel | $(4H/N) \times H$ | All-Reduce (SUM) | Identity |
| **LM Head** | Column Parallel | $H \times V/N$ | 0 (or All-Gather) | All-Reduce (SUM) |

---

## 12. Complete Standalone PyTorch Implementation: Megatron TP Layers from Scratch

Here is the complete, self-contained implementation of Megatron's Tensor Parallel building blocks using pure PyTorch. You can study, copy, and run this code directly:

```python
"""
Megatron 1D Tensor Parallelism: Complete PyTorch Implementation.
Includes:
- Autograd Conjugate Operators f and g
- ColumnParallelLinear
- RowParallelLinear
- Parallel Causal Self-Attention
- Parallel MLP
- Full MegatronTransformerBlock
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ── 1. Conjugate Autograd Operators f and g ──

class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    Operator f:
    Forward: Pass input unmodified (Identity).
    Backward: All-Reduce SUM across TP ranks (aggregates partitioned gradients).
    """
    @staticmethod
    def forward(ctx, input_, group=None):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized() and dist.get_world_size(ctx.group) > 1:
            dist.all_reduce(grad_output, group=ctx.group, op=dist.ReduceOp.SUM)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    Operator g:
    Forward: All-Reduce SUM across TP ranks (combines partial row-parallel outputs).
    Backward: Pass upstream gradient unmodified (Identity).
    """
    @staticmethod
    def forward(ctx, input_, group=None):
        if dist.is_initialized() and dist.get_world_size(group) > 1:
            dist.all_reduce(input_, group=group, op=dist.ReduceOp.SUM)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def copy_to_model_parallel_region(input_, group=None):
    return _CopyToModelParallelRegion.apply(input_, group)

def reduce_from_model_parallel_region(input_, group=None):
    return _ReduceFromModelParallelRegion.apply(input_, group)


# ── 2. ColumnParallelLinear ──

class ColumnParallelLinear(nn.Module):
    """
    Splits weight along columns (out_features dimension):
    W = [W_1, W_2, ..., W_N], where each W_i has shape [in_features, out_features / N].
    No forward communication. Backward applies All-Reduce SUM via operator f.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, tp_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        
        self.tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0
        
        assert out_features % self.tp_world_size == 0, "out_features must be divisible by TP size"
        self.out_features_per_partition = out_features // self.tp_world_size
        
        # Local weight shard: [out_features_per_partition, in_features]
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Megatron initialization scales w.r.t unpartitioned dimensions
        std = math.sqrt(2.0 / self.in_features)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Operator f: identity forward, all-reduce backward
        input_parallel = copy_to_model_parallel_region(x, self.tp_group)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        return output_parallel


# ── 3. RowParallelLinear ──

class RowParallelLinear(nn.Module):
    """
    Splits weight along rows (in_features dimension):
    W = [W_1^T, W_2^T, ..., W_N^T]^T, where each W_i has shape [in_features / N, out_features].
    Forward applies All-Reduce SUM via operator g.
    Crucial: bias is added AFTER All-Reduce to prevent N-times duplication!
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_scale: float = 1.0, tp_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale
        self.tp_group = tp_group
        
        self.tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0
        
        assert in_features % self.tp_world_size == 0, "in_features must be divisible by TP size"
        self.in_features_per_partition = in_features // self.tp_world_size
        
        # Local weight shard: [out_features, in_features_per_partition]
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        
        if bias:
            # Bias is replicated across all ranks
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Scaled w.r.t global full fan-in and residual depth scaling factor:
        # std = sqrt(2 / d_model) * init_scale (where init_scale = 1 / sqrt(2 * num_layers))
        std = math.sqrt(2.0 / self.in_features) * self.init_scale
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matrix multiply local slice WITHOUT bias
        output_parallel = F.linear(x, self.weight, None)
        # Operator g: All-Reduce partial sums
        output_ = reduce_from_model_parallel_region(output_parallel, self.tp_group)
        # Add bias ONCE to the final gathered sum
        if self.bias is not None:
            output_ = output_ + self.bias
        return output_


# ── 4. Megatron Parallel MLP ──

class ParallelMLP(nn.Module):
    """
    ColumnParallelLinear (H -> 4H) -> GELU -> RowParallelLinear (4H -> H).
    Only 1 All-Reduce in forward pass!
    """
    def __init__(self, hidden_size: int, ffn_hidden_size: int, init_scale: float = 1.0, tp_group=None):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, ffn_hidden_size, bias=True, tp_group=tp_group)
        self.activation = nn.GELU()
        self.dense_4h_to_h = RowParallelLinear(ffn_hidden_size, hidden_size, bias=True, init_scale=init_scale, tp_group=tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H]
        h = self.dense_h_to_4h(x)    # [B, S, 4H/N] (no comm)
        h = self.activation(h)        # [B, S, 4H/N] (local)
        out = self.dense_4h_to_h(h)   # [B, S, H] (1 All-Reduce)
        return out


# ── 5. Megatron Parallel Self-Attention ──

class ParallelSelfAttention(nn.Module):
    """
    Multi-Head Attention with partitioned heads.
    QKV projection is ColumnParallel (3H/N).
    Output projection is RowParallel (H/N -> H) with residual scaling init_scale.
    Only 1 All-Reduce in forward pass!
    """
    def __init__(self, hidden_size: int, num_heads: int, init_scale: float = 1.0, attention_dropout_prob: float = 0.0, tp_group=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_group = tp_group
        
        tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        assert num_heads % tp_world_size == 0, f"num_heads ({num_heads}) must be divisible by TP size ({tp_world_size})"
        self.num_local_heads = num_heads // tp_world_size
        self.head_dim = hidden_size // num_heads

        # ColumnParallel project to Q, K, V simultaneously: 3 * (num_local_heads * head_dim)
        self.query_key_value = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, bias=True, tp_group=tp_group
        )
        # Optional attention dropout
        self.attention_dropout = nn.Dropout(attention_dropout_prob) if attention_dropout_prob > 0.0 else None

        # RowParallel project back to hidden_size with 1/sqrt(2L) residual scaling
        self.dense = RowParallelLinear(
            hidden_size, hidden_size, bias=True, init_scale=init_scale, tp_group=tp_group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        # [B, S, 3 * num_local_heads * head_dim]
        qkv = self.query_key_value(x)
        
        # Real Megatron Core QKV Layout:
        # Reshape to separate the 3 projections and unbind along dim=2:
        qkv = qkv.reshape(B, S, 3, self.num_local_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each is [B, S, num_local_heads, head_dim]
        
        # Transpose to multi-head batch format: [B, num_local_heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention on local heads (zero comm!)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Robust Causal Mask: use torch.finfo(dtype).min to guarantee no NaN/underflow across FP16/BF16/FP32
        mask_value = torch.finfo(scores.dtype).min
        causal_mask = torch.triu(
            torch.full((S, S), mask_value, dtype=scores.dtype, device=x.device), diagonal=1
        )
        scores = scores + causal_mask
        probs = F.softmax(scores, dim=-1)
        if self.attention_dropout is not None:
            probs = self.attention_dropout(probs)
        
        # Context: [B, num_local_heads, S, head_dim] -> [B, S, num_local_heads * head_dim]
        context = torch.matmul(probs, v).transpose(1, 2).reshape(B, S, -1)
        
        # RowParallel output projection (1 All-Reduce across TP ranks)
        output = self.dense(context)
        return output


# ── 6. Full Megatron Transformer Block ──

class MegatronTransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer Block with Megatron 1D TP.
    Total communications: exactly 2 All-Reduces per layer!
    Residual scaling 1 / sqrt(2 * num_layers) applied to both residual projections.
    """
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, num_layers: int = 1, tp_group=None):
        super().__init__()
        # Megatron-LM scaling rule: variance of residual stream preserved by scaling output projections
        init_scale = 1.0 / math.sqrt(2.0 * num_layers)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attention = ParallelSelfAttention(hidden_size, num_heads, init_scale=init_scale, tp_group=tp_group)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = ParallelMLP(hidden_size, ffn_hidden_size, init_scale=init_scale, tp_group=tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention with residual connection
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attention(norm_x)
        x = x + attn_out

        # Pre-LN MLP with residual connection
        norm_x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        return x
```

### Deep Line-by-Line Pedagogical Breakdown of the TP Architecture:

#### 1. Why `F.linear` Shape is `[out_features, in_features]`
In PyTorch, the linear layer function `torch.nn.functional.linear(input, weight, bias)` implements:
$$y = x W^T + b$$
Because PyTorch transposes the weight matrix during computation, the memory layout of `weight` is stored as `[out_features, in_features]`:
- In **`ColumnParallelLinear`**: We partition `out_features`. Therefore, local weight has shape:
  $$\left[\frac{\text{out\_features}}{N}, \text{in\_features}\right]$$
- In **`RowParallelLinear`**: We partition `in_features`. Therefore, local weight has shape:
  $$\left[\text{out\_features}, \frac{\text{in\_features}}{N}\right]$$

---

#### 2. The Autograd Conjugate Pairing in Code
Look at where the autograd functions are placed:
- In **`ColumnParallelLinear.forward`**:
  ```python
  input_parallel = copy_to_model_parallel_region(x, self.tp_group) # Operator f
  output_parallel = F.linear(input_parallel, self.weight, self.bias)
  ```
  `copy_to_model_parallel_region` is an **identity** in forward. But it plants a hook in PyTorch's backward graph! When PyTorch runs backpropagation, it automatically triggers `dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)`. You never have to manually write backward communication logic!

- In **`RowParallelLinear.forward`**:
  ```python
  output_parallel = F.linear(x, self.weight, None) # NO bias!
  output_ = reduce_from_model_parallel_region(output_parallel, self.tp_group) # Operator g
  if self.bias is not None:
      output_ = output_ + self.bias
  ```
  `reduce_from_model_parallel_region` triggers an immediate **forward All-Reduce** to sum the partial dot-products across GPUs. During backpropagation, it returns the incoming gradient without modification.

---

#### 3. Why the Bias Timing is a Life-or-Death Detail
Notice that in `RowParallelLinear`, we explicitly pass `None` as the bias to `F.linear`, and only add `self.bias` **after** the All-Reduce:
```python
# CRITICAL:
output_parallel = F.linear(x, self.weight, None)  # 1. Multiply without bias
output_ = reduce_from_model_parallel_region(...)  # 2. All-Reduce partial sums
if self.bias is not None:
    output_ = output_ + self.bias                 # 3. Add bias ONCE to the true sum
```

What would happen if you added the bias during `F.linear(x, self.weight, self.bias)` before the All-Reduce?
Each of the $N$ GPUs would add its local bias $b$. Then, the All-Reduce would sum all $N$ outputs:
$$\text{Summed Output} = \left(\sum_{i=1}^N x_i W_i\right) + \mathbf{N \times b}$$
The bias would be added **$N$ times!** For $N = 8$, the bias term is multiplied by 8, instantly blowing up activations, destabilizing LayerNorm, and causing gradient explosion (NaN loss) within 10 iterations!

---

#### 4. The Megatron Attention Factoring Efficiency
Notice how `ParallelSelfAttention` fuses the Q, K, and V projections:
```python
self.query_key_value = ColumnParallelLinear(
    hidden_size, 3 * hidden_size, bias=True, tp_group=tp_group
)
```
- In naive attention, you would call `linear_q(x)`, `linear_k(x)`, and `linear_v(x)` separately.
- In Megatron, a single Column-Parallel GEMM produces the concatenated $[Q, K, V]$ tensor of shape $[B, S, 3 \times h_{local} \times d_{head}]$.
- Because all heads are partitioned evenly across ranks ($h_{local} = h / N$), each GPU computes attention for its assigned heads using standard scaled dot-product attention in local HBM.
- **Zero communication occurs during the entire attention score calculation and softmax!** Communication only occurs once, at the very end of the output projection `RowParallelLinear`.

---

## 13. Common Bugs & Gotchas in Tensor Parallelism

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **$N \times b$ Bias Failure** | Loss explodes to NaN in $<10$ steps | Passing `bias` directly to `F.linear` in `RowParallelLinear` before the All-Reduce | Set `bias=None` in `RowParallelLinear`'s GEMM; add `bias` explicitly **after** `dist.all_reduce()` |
| **GQA Head Imbalance** | `AssertionError: num_heads % TP != 0` | Attempting to shard 8 KV heads across $\text{TP}=16$ | In Grouped-Query Attention, $h_{\text{KV}}$ must be an integer multiple of $N$; use $\text{TP} \le h_{\text{KV}}$ |
| **Vocab Parallel Striding Error** | `RuntimeError: input and target shapes do not match` | Target class indices not offset to local partition $[0, V/N)$ | Mask unowned targets with `-100` and subtract `vocab_start_index` before cross-entropy |
| **Duplicate RNG Seeds** | All TP ranks sample identical dropout masks | Using the same random seed across all TP ranks | Use Megatron's `CudaRNGStatesTracker` with distinct per-rank seeds for dropout |
| **Inconsistent Weight Init** | Output differs across TP ranks on identical inputs | Column/Row parallel weights initialized with different random states | Seed identically for replicated weights, use rank-dependent offsets for partitioned weights |

---

## 14. Runnable Checklist & Verification

To verify the complete 1D Tensor Parallelism module locally on 2 CPU workers:

```bash
# Verify ColumnParallelLinear, RowParallelLinear, and VocabParallelEmbedding
torchrun --nproc_per_node=2 demo_megatron.py
```

**Key Execution Checks**:
1. Confirm that `tp_size=2` partitions `hidden_size=1024` into two local slices of `512`.
2. Confirm that forward loss matches the unpartitioned reference model within floating-point tolerance ($\Delta < 1 \times 10^{-5}$).
3. Verify that backward gradients across both ranks are synchronized via the backward All-Reduce of operator $f$.

In **Book 3**, we will study **Sequence Parallelism (SP)** from Megatron-LM v3: how to eliminate activation memory redundancy in LayerNorm and Dropout with zero extra communication!

