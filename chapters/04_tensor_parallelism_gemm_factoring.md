# 1D Tensor Parallelism & Linear Operator Sharding
> **Column/Row GEMMs, Parallel Cross-Entropy, and Tensor Core Boundary Alignment**

> **Reference Paper**: *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (Shoeybi et al., NVIDIA 2019, [arXiv:1909.08053](https://arxiv.org/abs/1909.08053))

---

## 1.1. The Core Problem: Why Model Parallelism Failed Before Megatron

Before Megatron-LM was introduced by NVIDIA in 2019, model parallelism was widely considered impractical for training large neural networks.

Consider a standard 2-layer Multi-Layer Perceptron (MLP) found in every Transformer block:
$$Y = \text{GELU}(X W_1) W_2$$

Where:
- $X \in \mathbb{R}^{B \times H}$ ($B$ is batch size $\times$ sequence length, $H$ is hidden dimension)
- $W_1 \in \mathbb{R}^{H \times 4H}$
- $W_2 \in \mathbb{R}^{4H \times H}$

### 1.1.1 The Naive Row-Parallel Trap:
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

## 1.2. Megatron's Breakthrough: Complementary GEMM Factoring

Shoeybi et al. (2019) solved this with a brilliant algebraic insight:
> **If you pair a Column-Parallel linear layer with a Row-Parallel linear layer, the non-linear activation is completely trapped between them, requiring ZERO communication!**

### 1.2.1 The Algebraic and Geometric Proof

Let us write out the exact block-matrix multiplication for the two splitting strategies:

#### 1.2.1.1 Step 1: Column-Parallel GEMM (Layer 1)
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

#### 1.2.1.2 Step 2: The Non-Linearity Property
Now we apply $\text{GELU}$ to the partitioned output $\begin{bmatrix} Z_{1,1} & Z_{1,2} \end{bmatrix}$.
Because GELU is an **elementwise function** (it operates independently on each individual number without mixing elements across columns):

$$\text{GELU}\left(\begin{bmatrix} Z_{1,1} & Z_{1,2} \end{bmatrix}\right) = \begin{bmatrix} \text{GELU}(Z_{1,1}) & \text{GELU}(Z_{1,2}) \end{bmatrix} = \begin{bmatrix} A_{1,1} & A_{1,2} \end{bmatrix}$$

- GPU 0 evaluates $\text{GELU}(Z_{1,1})$ on its local memory.
- GPU 1 evaluates $\text{GELU}(Z_{1,2})$ on its local memory.
- **Communication Required: EXACTLY ZERO!**

---

#### 1.2.1.3 Step 3: Row-Parallel GEMM (Layer 2)
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

## 1.3. The Megatron Multi-Head Attention Block

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

### 1.3.1 Detailed Execution Steps:
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

### 1.3.2 Total Forward Communications Per Transformer Block:
$$\text{Attention All-Reduce (1)} + \text{MLP All-Reduce (1)} = \mathbf{2 \text{ All-Reduces per Block}}$$

---

## 1.4. The Autograd Conjugate Operators: $f$ and $g$

Megatron formalized the communication using two symbolic operators in the computation graph:

$$\text{Block}(X) = X + \mathbf{g}\Big(\text{RowProj}\big(\text{Attn}(\text{ColQKV}(\mathbf{f}(X)))\big)\Big)$$

### 1.4.1 Mathematical Derivation of Gradients:

#### 1.4.1.1 Operator $f$ (Identity in Forward):
$$f(X) = X$$
During backward propagation, by chain rule:
$$\frac{\partial L}{\partial X} = \sum_{i=1}^N \frac{\partial L}{\partial Y_i}$$
Since rank $i$ holds the local gradient $\frac{\partial L}{\partial Y_i}$, to compute $\frac{\partial L}{\partial X}$, we **must All-Reduce (SUM) the incoming gradients**:
$$f^*(\text{grad}) = \text{All-Reduce}(\text{grad})$$

#### 1.4.1.2 Operator $g$ (All-Reduce in Forward):
$$g(Y_1, \dots, Y_N) = \sum_{i=1}^N Y_i$$
Since every rank receives the same total output $Y$, the gradient with respect to each input slice is identical:
$$\frac{\partial L}{\partial Y_i} = \frac{\partial L}{\partial Y}$$
Thus, no communication is required during the backward pass:
$$g^*(\text{grad}) = \text{grad}$$

---

