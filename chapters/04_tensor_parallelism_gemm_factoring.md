# 1D Tensor Parallelism & Linear Operator Sharding
> **Column/Row GEMMs, Parallel Cross-Entropy, and Tensor Core Boundary Alignment**

> **Reference Paper**: *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (Shoeybi et al., NVIDIA 2019, [arXiv:1909.08053](https://arxiv.org/abs/1909.08053))

---

## 1.1. The Core Problem: Why Model Parallelism Failed Before Megatron

Before Megatron-LM was introduced by NVIDIA in 2019, model parallelism was widely considered impractical for training large neural networks.

Consider a standard 2-layer Multi-Layer Perceptron (MLP) found in every Transformer block:

> `Y = GELU(X W_1) W_2`

Where:
- `X in shape [B * H]` (B is batch size `*` sequence length, H is hidden dimension)
- `W_1 in shape [H * 4H]`
- `W_2 in shape [4H * H]`

### 1.1.1 The Naive Row-Parallel Trap:
Suppose you naively try to split W_1 across `N = 2` GPUs along its rows (input dimension):

> `W_1 = [W[1,1]; W[1,2]], where W[1,1], W[1,2] in shape [(H / 2) * 4H]`

To multiply X by this row-split W_1, the input X must also be split along its columns:

> `X = [X_1 , X_2], where X_1, X_2 in shape [B * (H / 2)]`

Each GPU computes its local matrix product:
- GPU 0 computes: `Z_1 = X_1 W[1,1] in shape [B * 4H]`
- GPU 1 computes: `Z_2 = X_2 W[1,2] in shape [B * 4H]`

The true intermediate activation is the sum of these outer products:

> `Z = Z_1 + Z_2`

Now comes the fatal flaw: **We must apply the non-linear activation `GELU(Z)`**.
Because non-linear functions do not distribute over addition:

> `GELU(Z_1 + Z_2) != GELU(Z_1) + GELU(Z_2)`

**The Catastrophic Result**:
GPU 0 and GPU 1 **cannot evaluate GELU locally**! They must pause, synchronize, and execute an expensive **All-Reduce communication** across the network just to compute `Z = Z_1 + Z_2` before either GPU can compute GELU!
Then, to execute the second layer W_2, another communication synchronization is required.

In an 80-layer model, communicating across the network on every single matrix multiply caused the GPU's compute cores to spend >75% of their time waiting for network packets. Model parallelism was considered dead on arrival.

---

## 1.2. Megatron's Breakthrough: Complementary GEMM Factoring

Shoeybi et al. (2019) solved this with a brilliant algebraic insight:
> **If you pair a Column-Parallel linear layer with a Row-Parallel linear layer, the non-linear activation is completely trapped between them, requiring ZERO communication!**

### 1.2.1 The Algebraic and Geometric Proof

Let us write out the exact block-matrix multiplication for the two splitting strategies:

#### 1.2.1.1 Step 1: Column-Parallel GEMM (Layer 1)
Slice `W_1 in shape [H * 4H]` along its **columns** into `N = 2` blocks:

> `W_1 = [W[1,1] , W[1,2]], where each W[1,i] in shape [H * (4H / 2)]`

Every GPU holds the **full, identical input** `X in shape [B * H]`.
Multiplying X by the column-sliced matrix yields:

> `X W_1 = X [W[1,1] , W[1,2]] = [X W[1,1] , X W[1,2]] = [Z[1,1] , Z[1,2]]`

Notice what just happened:
- GPU 0 computes `Z[1,1] = X W[1,1]` locally.
- GPU 1 computes `Z[1,2] = X W[1,2]` locally.
- **Communication Required: EXACTLY ZERO!**

---

#### 1.2.1.2 Step 2: The Non-Linearity Property
Now we apply GELU to the partitioned output `[Z[1,1] , Z[1,2]]`.
Because GELU is an **elementwise function** (it operates independently on each individual number without mixing elements across columns):

> `GELU([Z[1,1] , Z[1,2]]) = [GELU(Z[1,1]) , GELU(Z[1,2])] = [A[1,1] , A[1,2]]`

- GPU 0 evaluates `GELU(Z[1,1])` on its local memory.
- GPU 1 evaluates `GELU(Z[1,2])` on its local memory.
- **Communication Required: EXACTLY ZERO!**

---

#### 1.2.1.3 Step 3: Row-Parallel GEMM (Layer 2)
Now we must multiply the intermediate activations `A = [A[1,1] , A[1,2]]` by the second weight matrix `W_2 in shape [4H * H]`.
Notice that A is naturally partitioned **column-wise** across the two GPUs!

Therefore, we slice W_2 along its **rows**:

> `W_2 = [W[2,1]; W[2,2]], where each W[2,i] in shape [(4H / 2) * H]`

Now perform the block matrix multiplication:

> `Y = A W_2 = [A[1,1] , A[1,2]] [W[2,1]; W[2,2]] = A[1,1] W[2,1] + A[1,2] W[2,2]`

Look at the symmetry:
- GPU 0 already holds `A[1,1]`. It multiplies it by its local row slice `W[2,1]` to get partial output `Y_1 = A[1,1] W[2,1]`.
- GPU 1 already holds `A[1,2]`. It multiplies it by its local row slice `W[2,2]` to get partial output `Y_2 = A[1,2] W[2,2]`.
- Both Y_1 and Y_2 have shape `[B, H]`.

To obtain the true final output `Y = Y_1 + Y_2`, we execute **ONE All-Reduce (SUM)** across the GPUs!

> `Y = All-Reduce(Y_1 + Y_2)`

### 1.2.2 The Revolutionary Result:
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

### 1.2.3 Why this is mathematically optimal:
1. W_1 is **Column-Parallel** (`H -> 4H/N`).
2. The intermediate activation Z_1 is partitioned: rank i holds `Z[1,i]`.
3. GELU is an **elementwise function**:

> `GELU([Z[1,1], Z[1,2]]) = [GELU(Z[1,1]), GELU(Z[1,2])]`

   Each rank evaluates GELU on its own slice **without any network communication**!
4. W_2 is **Row-Parallel** (`4H/N -> H`). It directly consumes the partitioned activations `A[1,i]`.
> [!TIP]
> **Why This Matters**:
> If you reversed the order (Row-Parallel followed by Column-Parallel), the non-linear GELU activation would be forced to operate on unsummed partial dot-products. To fix this, you would have to insert an extra All-Reduce *before* GELU, doubling your communication cost per layer from 2 All-Reduces to 4 All-Reduces! The Column-then-Row factoring is the only mathematical permutation that allows non-linear activations to execute locally without communication.

---

## 1.3. The Megatron Multi-Head Attention Block

In Multi-Head Attention (MHA), the hidden dimension H is split across h attention heads:

> `d_head = (H / h)`

Megatron partitions the heads across the N tensor parallel GPUs:

> `h_local = (h / N)`

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

### 1.3.1 Detailed Execution Steps:
1. **QKV Projection (Column Parallel)**:
   - A single weight matrix `W_QKV in shape [H * 3H]` is split column-wise.
   - Each rank projects its input to `[Q_i, K_i, V_i]` of size `B * S * (3 * h_local * d_head)`.
   - **Communication: 0.**

2. **Self-Attention Computation (Local)**:
   - Each rank computes scaled dot-product attention for its own heads:

> `Attention(Q_i, K_i, V_i) = softmax(Q_i K_i^T / sqrt(d_head)) V_i`

   - Since attention heads are completely independent, **communication is 0!**

3. **Output Projection (Row Parallel)**:
   - The concatenated outputs of all heads are projected back to H using Row Parallel linear layer W_proj.
   - Operator g executes **one All-Reduce** to sum the projected vectors.

### 1.3.2 Total Forward Communications Per Transformer Block:

> `Attention All-Reduce (1) + MLP All-Reduce (1) = 2 All-Reduces per Block`

---

## 1.4. The Autograd Conjugate Operators: f and g

Megatron formalized the communication using two symbolic operators in the computation graph:

> `Block(X) = X + g(RowProj(Attn(ColQKV(f(X)))))`

### 1.4.1 Mathematical Derivation of Gradients:

#### 1.4.1.1 Operator f (Identity in Forward):

> `f(X) = X`

During backward propagation, by chain rule:

> `d(L) / d(X) = sum(i=1)^N d(L) / d(Y_i)`

Since rank i holds the local gradient `d(L) / d(Y_i)`, to compute `d(L) / d(X)`, we **must All-Reduce (SUM) the incoming gradients**:

> `f^*(grad) = All-Reduce(grad)`

#### 1.4.1.2 Operator g (All-Reduce in Forward):

> `g(Y_1, ..., Y_N) = sum(i=1)^N Y_i`

Since every rank receives the same total output Y, the gradient with respect to each input slice is identical:

> `d(L) / d(Y_i) = d(L) / d(Y)`

Thus, no communication is required during the backward pass:

> `g^*(grad) = grad`

---

