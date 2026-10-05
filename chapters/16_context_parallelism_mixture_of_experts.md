# Mixture of Experts (MoE) & Expert Parallelism
> **Switch Transformers, Top-K Routing, Capacity Factor, and All-to-All Token Dispatch**

---

## 2.1. Mixture of Experts (MoE) & Expert Parallelism (EP)

Dense Large Language Models become computationally prohibitive to scale beyond a few hundred billion parameters because **every single token must execute matrix multiplications across every single weight in the network**.

**Mixture of Experts (MoE)** decouples total parameter capacity from FLOPs per token:
- The standard dense MLP block is replaced by E distinct, independent expert MLPs:
  ```text
  Expert_set = \{ MLP_1, MLP_2, ..., MLP_E \}
  ```
- A parameterized **Router (Gating Network)** routes each token to a sparse subset of k experts (typically `k = 1` or `k = 2`, out of `E = 8, 64, or 256` experts).

```
Dense Model:  Token x ─────────────────────────> [ Full Dense MLP ] ─────────────────────────> Output
                                                       (All params active)

MoE Model:    Token x ───┬─────────────────────> [ Router Gate W_g ] ───> Select Expert 2 & 7
                         │                                                        │
                         ├─────────────────────────────────────────┐              │
                         ▼                                         ▼              ▼
                  [ Expert MLP 2 ]                          [ Expert MLP 7 ]      │
                         │                                         │              │
                         └─────────────────┬───────────────────────┘              │
                                           ▼                                      ▼
                               Output = g_2 · MLP_2(x)      +      g_7 · MLP_7(x)
```

For a model with `E = 64` experts and `k = 2`:
- **Parameter Capacity**: `64 *` the parameter capacity of a single MLP.
- **Compute Cost (FLOPs)**: Only `2 *` the compute cost of a single MLP!

---

### 2.1.1 The Mathematical Routing Formulation

Given token embedding `x in shape [H]`:
1. The router computes raw routing logits via a linear projection `W_g in shape [H * E]`:
   ```text
   h(x) = x * W_g in shape [E]
   ```
2. Softmax normalization over all experts:
   ```text
   P(x) = Softmax(h(x)), P_i(x) = (e^{h_i(x)} / sum(j=1)^E e^{h_j(x)})
   ```
3. Select the Top-k expert indices:
   ```text
   TopK_set = Top-k(P(x), k)
   ```
4. Renormalize the routing gates among the chosen k experts so their weights sum to 1.0:
   ```text
   g_i(x) = (P_i(x) / sum(j in TopK_set) P_j(x)) for i in TopK_set
   ```
5. Final MoE output:
   ```text
   y = sum(i in TopK_set) g_i(x) * MLP_i(x)
   ```

---

### 2.1.2 The Routing Collapse Trap & Auxiliary Load-Balancing Loss

A naive router trained via backpropagation quickly collapses into a pathological local minimum:
- Early in training, due to random initialization, Expert 3 might produce slightly better gradients than other experts.
- The router routes more tokens to Expert 3.
- Because Expert 3 receives more tokens, it updates faster and becomes better at minimizing loss.
- Soon, the router routes **100% of all tokens to Expert 3**, leaving the remaining E-1 experts completely unutilized!
- The model degenerates into a tiny dense model, wasting 99% of its parameters.

To prevent routing collapse, Fedus et al. (2021) introduced the **Auxiliary Load-Balancing Loss** (Loss_aux).

#### 2.1.2.1 Mathematical Derivation of Loss_aux

Let T be the total number of tokens in a training microbatch.
Define two probability distributions over the E experts:

1. **Fraction of Dispatched Tokens (f_i)**: The empirical ratio of tokens assigned to expert i:
   ```text
   f_i = (1 / T) sum(t=1)^T Indicator(Expert i in TopK_set_t)
   ```
   *(Note: Because the argmax indicator function Indicator is non-differentiable, backpropagation cannot pass gradients through f_i directly.)*

2. **Average Routing Probability (P_i)**: The mean continuous probability assigned to expert i by the router:
   ```text
   P_i = (1 / T) sum(t=1)^T P_i(x_t)
   ```
   *(This quantity is fully differentiable with respect to router weights W_g.)*

The auxiliary loss is defined as:
```text
Loss_aux = alpha * E sum(i=1)^E f_i * P_i
```

Where:
- E is the number of experts (scaling factor).
- alpha is a hyperparameter (typically 0.01).

#### 2.1.2.2 Proof of Minimum at Uniform Load

By the Cauchy-Schwarz inequality, for positive vectors f and P constrained by `sum f_i = 1` and `sum P_i = 1`:
```text
sum(i=1)^E f_i P_i >= (1 / E) ( sum(i=1)^E sqrt(f_i P_i) )^2 >= (1 / E)
```

The minimum occurs **if and only if all experts receive an equal share of tokens**:
```text
f_i = (1 / E), P_i = (1 / E) for all i in \{1, ..., E\}
```

At this point of perfect balance:
```text
Loss_aux = alpha * E sum(i=1)^E ((1 / E) * (1 / E)) = alpha * E (E * (1 / E^2)) = alpha
```

Any imbalance increases `sum f_i P_i`, penalizing the router and forcing it to distribute tokens uniformly across the entire expert pool.

---

## 2.2. Expert Parallelism (EP) & All-to-All Token Dispatch

When an MoE model has `E = 64` or 256 experts, the expert weights cannot fit on a single GPU.
We partition the experts across P_EP GPUs:
```text
Experts per GPU = (E / P_EP)
```

- GPU 0 hosts Experts `0 ... 7`.
- GPU 1 hosts Experts `8 ... 15`.
- GPU 2 hosts Experts `16 ... 23`, etc.

### 2.2.1 The All-to-All Shuffling Architecture

A token originating on GPU 0 might be routed to Expert 14 (located on GPU 1) and Expert 22 (located on GPU 2).
To execute this, tokens must be physically transmitted across the cluster using **`All-to-All` communication**:

```
                       EXPERT PARALLEL ALL-TO-ALL DATAFLOW
                       
GPU 0: Tokens [T0, T1, T2, T3]               GPU 1: Tokens [T4, T5, T6, T7]
  Router maps:                                 Router maps:
  T0 -> Exp 1 (Rank 0)                         T4 -> Exp 2 (Rank 0)
  T1 -> Exp 9 (Rank 1)                         T5 -> Exp 11 (Rank 1)
  T2 -> Exp 3 (Rank 0)                         T6 -> Exp 1 (Rank 0)
  T3 -> Exp 12 (Rank 1)                        T7 -> Exp 10 (Rank 1)
       │                                            │
       ▼                                            ▼
Pack by Destination:                         Pack by Destination:
  To Rank 0: [T0, T2]                          To Rank 0: [T4, T6]
  To Rank 1: [T1, T3]                          To Rank 1: [T5, T7]
       │                                            │
       └────────────────────┬───────────────────────┘
                            ▼
           PHASE 1: DISPATCH ALL-TO-ALL (dist.all_to_all)
                            │
       ┌────────────────────┴───────────────────────┐
       ▼                                            ▼
GPU 0 Receives: [T0, T2, T4, T6]             GPU 1 Receives: [T1, T3, T5, T7]
(All tokens destined for Experts 0..7)       (All tokens destined for Experts 8..15)
       │                                            │
       ▼                                            ▼
Execute Local Experts 0..7 on GPU 0          Execute Local Experts 8..15 on GPU 1
       │                                            │
       └────────────────────┬───────────────────────┘
                            ▼
           PHASE 2: COMBINE ALL-TO-ALL (dist.all_to_all)
                            │
       ┌────────────────────┴───────────────────────┐
       ▼                                            ▼
GPU 0 Receives Outputs for: [T0, T1, T2, T3] GPU 1 Receives Outputs for: [T4, T5, T6, T7]
       │                                            │
       ▼                                            ▼
Weighted Sum: y = g_1·O_1 + g_2·O_2          Weighted Sum: y = g_1·O_1 + g_2·O_2
```

### 2.2.2 Interaction of Expert Parallelism (EP) + Context Parallelism (CP)

In modern architectures like DeepSeek-V3 or Mixtral trained on 128k contexts, **Context Parallelism and Expert Parallelism operate simultaneously**:

1. **Attention Phase (CP Domain)**:
   - The sequence S is sharded across C Context Parallel ranks.
   - Each GPU computes Ring Attention on its local slice of `S / C` tokens.
   - Output of the attention layer on rank c is a tensor of shape `[B, S / C, H]`.

2. **Routing & Dispatch Phase (EP Domain)**:
   - Rank c passes its local `T = B * (S / C)` tokens into the MoE Router.
   - The router assigns each token to global experts.
   - Now, **the All-to-All collective operates across the Expert Parallel process group**:
     Tokens originating from rank c's context shard are shipped directly to the GPU hosting their chosen expert!
3. **Capacity & Buffer Management**:
   - Because `T = B * S / C`, the number of tokens each GPU routes is reduced by factor C.
   - This prevents All-to-All communication buffer explosion, keeping per-GPU dispatch memory strictly bounded even at `128k+` sequence lengths!

---


---

## 2.3. Summary: The Complete 5D Parallelism Matrix

With Context Parallelism and Expert Parallelism added to the Megatron architectural stack, training frontier AI systems spans **5 orthogonal parallelism dimensions**:

```text
Total Cluster GPUs = TP * CP * EP * PP * DP
```

```
┌─────────────────────────┬──────────────────────┬──────────────────────┬───────────────────────────────┐
│ Parallelism Dimension   │ Target Component     │ Sharding Domain      │ Optimal Interconnect          │
├─────────────────────────┼──────────────────────┼──────────────────────┼───────────────────────────────┤
│ **TP (Tensor Parallel)**│ Hidden Dimension (H) │ Intra-Layer Matrix   │ Intra-Node NVLink (900 GB/s)  │
│ **CP (Context Parallel)│ Sequence Length (S)  │ Ring Attention       │ NVLink or High-Bandwidth IB   │
│ **EP (Expert Parallel)**│ MoE Expert MLPs      │ All-to-All Shuffling │ Low-Latency Bisection IB      │
│ **PP (Pipeline)**       │ Layer Depth (L)      │ Inter-Stage P2P      │ Cross-Node InfiniBand P2P     │
│ **DP (Data Parallel)**  │ Batch Size (B)       │ Distributed Optimizer│ Inter-Node Cluster InfiniBand │
└─────────────────────────┴──────────────────────┴──────────────────────┴───────────────────────────────┘
```

---

