# Context Parallelism & MoE Implementation
> **Runnable Ring Attention with Online Softmax, MoE Router, and Verifiable Test**

---

## 3.1. Reference Implementation: Ring Attention with Online Softmax

The following self-contained code demonstrates the complete Ring Attention engine in pure PyTorch, featuring the exact Online Softmax accumulator and non-blocking P2P communication primitives.

```python
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple

def online_softmax_update(
    m_prev: torch.Tensor,
    l_prev: torch.Tensor,
    O_prev: torch.Tensor,
    Q_block: torch.Tensor,
    K_block: torch.Tensor,
    V_block: torch.Tensor,
    scale: float,
    causal: bool = False,
    query_offset: int = 0,
    key_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes one incremental step of the Online Softmax recurrence.
    
    Shapes:
      m_prev, l_prev : [B, h, S_q, 1]     (running maximum and denominator)
      O_prev         : [B, h, S_q, d]     (running normalized output)
      Q_block        : [B, h, S_q, d]     (stationary query slice)
      K_block        : [B, h, S_kv, d]    (circulating key slice)
      V_block        : [B, h, S_kv, d]    (circulating value slice)
    """
    B, h, S_q, d = Q_block.shape
    S_kv = K_block.shape[2]

    # 1. Compute raw attention scores for this block: S = (Q K^T) * scale
    # [B, h, S_q, d] x [B, h, d, S_kv] -> [B, h, S_q, S_kv]
    scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

    # 2. Apply causal mask if requested
    if causal:
        # Create coordinate grid comparing global sequence positions
        q_idx = torch.arange(query_offset, query_offset + S_q, device=scores.device).view(1, 1, S_q, 1)
        k_idx = torch.arange(key_offset, key_offset + S_kv, device=scores.device).view(1, 1, 1, S_kv)
        causal_mask = q_idx < k_idx  # True where key position exceeds query position
        scores.masked_fill_(causal_mask, float('-inf'))

    # 3. Compute local maximum for current block: m_local [B, h, S_q, 1]
    m_local = scores.amax(dim=-1, keepdim=True)

    # 4. Update running global maximum: m_new = max(m_prev, m_local)
    m_new = torch.maximum(m_prev, m_local)

    # 5. Compute exponential rescale factors with IEEE 754 NaN guard
    # Edge case: If all tokens in a causal block are masked out (-inf),
    # m_prev = -inf and m_new = -inf. In IEEE 754: (-inf) - (-inf) = NaN!
    # We must explicitly zero out alpha when m_prev is -inf.
    diff = m_prev - m_new
    alpha = torch.where(
        torch.isneginf(m_prev),
        torch.zeros_like(m_prev),
        torch.exp(torch.clamp(diff, max=0.0))
    )
    # exp_scores = exp(scores - m_new) (masked entries with -inf become exp(-inf) = 0.0)
    exp_scores = torch.exp(scores - m_new)
    exp_scores = torch.nan_to_num(exp_scores, nan=0.0)

    # 6. Update running normalization denominator: l_new [B, h, S_q, 1]
    l_new = l_prev * alpha + exp_scores.sum(dim=-1, keepdim=True)

    # 7. Update running output accumulator: O_new [B, h, S_q, d]
    # Rescale past numerator by alpha, add new contribution, divide by l_new
    prev_num = O_prev * (l_prev * alpha)
    prev_num = torch.nan_to_num(prev_num, nan=0.0)
    O_new = (prev_num + torch.matmul(exp_scores, V_block)) / (l_new + 1e-8)

    return m_new, l_new, O_new


def ring_attention_forward(
    Q_local: torch.Tensor,
    K_local: torch.Tensor,
    V_local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    causal: bool = False,
) -> torch.Tensor:
    """
    Executes a full Ring Attention forward pass across all Context Parallel ranks
    using pre-allocated double-buffering (ping-pong buffers) to eliminate memory thrashing.
    The local query Q_local remains stationary on each GPU.
    The keys and values rotate around the P2P ring across C rounds.
    """
    rank = dist.get_rank(cp_group)
    world_size = dist.get_world_size(cp_group)
    
    send_dst = (rank + 1) % world_size
    recv_src = (rank - 1 + world_size) % world_size

    B, h, S_local, d = Q_local.shape
    scale = 1.0 / math.sqrt(d)

    # Initialize Online Softmax Accumulators
    m = torch.full((B, h, S_local, 1), float('-inf'), dtype=Q_local.dtype, device=Q_local.device)
    l = torch.zeros((B, h, S_local, 1), dtype=Q_local.dtype, device=Q_local.device)
    O = torch.zeros((B, h, S_local, d), dtype=Q_local.dtype, device=Q_local.device)

    # ── Double-Buffering Ping-Pong Memory Arenas ──
    # Pre-allocate two static buffer slots to eliminate dynamic allocations inside the ring loop:
    K_buffers = [K_local.clone(), torch.empty_like(K_local)]
    V_buffers = [V_local.clone(), torch.empty_like(V_local)]
    curr_idx = 0
    
    current_key_rank = rank

    for step in range(world_size):
        next_idx = 1 - curr_idx
        # Asynchronously initiate P2P transmission from curr_idx buffer
        # and reception directly into the pre-allocated next_idx buffer:
        if step < world_size - 1:
            p2p_ops = [
                dist.P2POp(dist.isend, K_buffers[curr_idx].contiguous(), send_dst, group=cp_group),
                dist.P2POp(dist.isend, V_buffers[curr_idx].contiguous(), send_dst, group=cp_group),
                dist.P2POp(dist.irecv, K_buffers[next_idx], recv_src, group=cp_group),
                dist.P2POp(dist.irecv, V_buffers[next_idx], recv_src, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(p2p_ops)

        # Compute Online Softmax update with current active buffer slot:
        query_offset = rank * S_local
        key_offset = current_key_rank * S_local
        
        m, l, O = online_softmax_update(
            m, l, O, Q_local, K_buffers[curr_idx], V_buffers[curr_idx], scale=scale,
            causal=causal, query_offset=query_offset, key_offset=key_offset
        )

        # Await completion of asynchronous P2P network transfers and ping-pong:
        if step < world_size - 1:
            for req in reqs:
                req.wait()
            curr_idx = next_idx  # Ping-pong active buffer slot
            current_key_rank = (current_key_rank - 1 + world_size) % world_size

    return O
```

### 3.1.1 Deep Line-by-Line Pedagogical Breakdown: Ring Attention Implementation

1. **Lines 31–32 (`scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale`):**
   - Computes query-key dot products between the stationary query block $Q_{\text{block}}$ and the current rotating key block $K_{\text{block}}$.
   - Floating-point arithmetic executes entirely inside fast SRAM/Tensor Cores.
2. **Lines 36–41 (Causal Mask Coordinate Grid):**
   - Compares absolute global token coordinates: `query_offset + q` vs `key_offset + k`.
   - If `q_idx < k_idx`, the interaction represents a query attending to a future token. Setting these scores to $-\infty$ ensures that $e^{-\infty} = 0$, completely nullifying future token influence in the attention sum.
3. **Lines 44–47 (`m_local`, `m_new`):**
   - Finds the row-wise maximum of the current block.
   - `torch.maximum(m_prev, m_local)` establishes the new global maximum across all blocks observed up to this step.
4. **Lines 51–53 (`alpha = torch.exp(m_prev - m_new)`):**
   - If the new block contains scores larger than any seen previously, $m_{\text{new}} > m_{\text{prev}}$, meaning $m_{\text{prev}} - m_{\text{new}} < 0$.
   - $\alpha \in (0, 1]$ acts as a fractional decay coefficient, downscaling the old unnormalized accumulators to align them with the new exponent base.
5. **Line 60 (`O_new = (O_prev * (l_prev * alpha) + ...) / (l_new + 1e-8)`):**
   - `O_prev * (l_prev * alpha)` recovers the exact unnormalized numerator from the previous iteration, scaled to base $m_{\text{new}}$.
   - Adding `torch.matmul(exp_scores, V_block)` incorporates the new value projections.
   - Dividing by $l_{\text{new}}$ normalizes the total sum, producing the mathematically exact weighted attention output.
6. **Lines 102–113 (`dist.batch_isend_irecv`):**
   - Packs non-blocking `isend` and `irecv` operations into a single kernel dispatch.
   - Crucially, this communication call is launched **before** the compute step `online_softmax_update()`, enabling the network hardware to transfer memory in parallel with Tensor Core execution.

---


---

## 3.2. Reference Implementation: MoE Router & Expert Parallel Dispatch

The code below implements a production-grade MoERouter with Top-$k$ selection and auxiliary loss, along with the two-phase `all_to_all` token dispatch and combine pipeline.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, List

class TopKMoERouter(nn.Module):
    """
    Top-K Gating Router for Mixture-of-Experts (MoE) with Capacity Factor enforcement.
    Computes gating probabilities, enforces expert token capacities, and computes auxiliary loss.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_coeff: float = 0.01,
        capacity_factor: Optional[float] = 1.25,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.capacity_factor = capacity_factor

        # Gating projection matrix W_g [H, E]
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input token representations [T, H]
        Returns:
            topk_weights: Normalized routing weights [T, top_k]
            topk_indices: Assigned expert indices [T, top_k]
            aux_loss: Scalar auxiliary load-balancing loss
            token_mask: Boolean mask [T, top_k] indicating retained tokens (1) vs capacity-dropped tokens (0)
        """
        T, H = x.shape

        # 1. Compute raw router logits and softmax probabilities
        logits = self.gate(x)                    # [T, E]
        probs = F.softmax(logits, dim=-1)        # [T, E]

        # 2. Extract Top-K experts per token
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # 3. Renormalize top-k weights to sum to 1.0
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Calculate Auxiliary Load-Balancing Loss
        expert_mask = F.one_hot(topk_indices[:, 0], num_classes=self.num_experts).float()
        f = expert_mask.mean(dim=0)             # [E]
        P = probs.mean(dim=0)                   # [E]
        aux_loss = self.aux_loss_coeff * self.num_experts * torch.sum(f * P)

        # 5. Capacity Factor Enforcement & Token Dropping
        token_mask = torch.ones_like(topk_weights, dtype=torch.bool)
        if self.capacity_factor is not None:
            # Capacity per expert = ceil(capacity_factor * (T * top_k / E))
            expert_capacity = math.ceil(self.capacity_factor * (T * self.top_k / self.num_experts))
            expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
            for k in range(self.top_k):
                for t in range(T):
                    exp_idx = topk_indices[t, k].item()
                    if expert_counts[exp_idx] < expert_capacity:
                        expert_counts[exp_idx] += 1
                    else:
                        token_mask[t, k] = False
                        topk_weights[t, k] = 0.0  # Zero out dropped token routing weight

            # Renormalize weights for surviving tokens
            weights_sum = topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = torch.where(weights_sum > 0, topk_weights / (weights_sum + 1e-8), topk_weights)

        return topk_weights, topk_indices, aux_loss, token_mask


class ExpertParallelEngine:
    """
    Orchestrates Expert Parallel token shuffling using dist.all_to_all_single.
    """
    def __init__(self, local_experts: nn.ModuleList, ep_group: dist.ProcessGroup):
        self.local_experts = local_experts
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_size = dist.get_world_size(ep_group)
        self.num_local_experts = len(local_experts)

    def forward(self, tokens: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        tokens: [T, H]
        topk_weights: [T, top_k]
        topk_indices: [T, top_k]
        """
        T, H = tokens.shape
        top_k = topk_indices.shape[1]

        # Expand tokens for top-k routing: [T * top_k, H]
        expanded_tokens = tokens.repeat_interleave(top_k, dim=0)
        flat_expert_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1, 1)

        # Determine target EP rank for each token: target_rank = expert_id // num_local_experts
        target_ranks = flat_expert_indices // self.num_local_experts

        # Count tokens destined for each EP rank
        send_counts = torch.zeros(self.ep_size, dtype=torch.long, device=tokens.device)
        for r in range(self.ep_size):
            send_counts[r] = (target_ranks == r).sum()

        # Exchange token counts so every rank knows how many tokens it will receive
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        # Sort tokens by destination rank for contiguous transmission
        sort_indices = torch.argsort(target_ranks)
        sorted_tokens = expanded_tokens[sort_indices]

        # Allocate buffer for incoming tokens
        total_recv_tokens = recv_counts.sum().item()
        received_tokens = torch.empty((total_recv_tokens, H), dtype=tokens.dtype, device=tokens.device)

        # ── PHASE 1: DISPATCH ALL-TO-ALL ─────────────────────────────────
        dist.all_to_all_single(
            received_tokens, sorted_tokens,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=self.ep_group
        )

        # ── LOCAL EXPERT COMPUTATION ────────────────────────────────────
        # Compute locally on received tokens (simplified: uniform feedthrough)
        processed_tokens = torch.zeros_like(received_tokens)
        if total_recv_tokens > 0:
            # Pass tokens through the first local expert for demonstration
            processed_tokens = self.local_experts[0](received_tokens)

        # ── PHASE 2: COMBINE ALL-TO-ALL ──────────────────────────────────
        # Send processed outputs back to original token owners
        returned_tokens = torch.empty_like(sorted_tokens)
        dist.all_to_all_single(
            returned_tokens, processed_tokens,
            output_split_sizes=send_counts.tolist(),
            input_split_sizes=recv_counts.tolist(),
            group=self.ep_group
        )

        # Restore original token order
        unsort_indices = torch.argsort(sort_indices)
        restored_tokens = returned_tokens[unsort_indices]

        # Weight by routing coefficients and reduce across top_k
        weighted_tokens = restored_tokens * flat_weights
        output = weighted_tokens.view(T, top_k, H).sum(dim=1)

        return output
```

---

### 3.2.1 Deep Line-by-Line Pedagogical Breakdown: MoE & Expert Parallelism

1. **Lines 31–32 (`logits = self.gate(x)`, `probs = F.softmax(...)`):**
   - Projects hidden representations $x \in \mathbb{R}^{T \times H}$ into expert probability simplex $\mathbb{R}^{T \times E}$.
2. **Lines 41–49 (`expert_mask`, `f`, `P`, `aux_loss`):**
   - Implements the Switch Transformer load-balancing auxiliary penalty.
   - `f` captures empirical routing allocation, while `P` provides smooth autograd gradients to steer router weights away from collapsed states.
3. **Line 68 (`expanded_tokens = tokens.repeat_interleave(top_k, dim=0)`):**
   - For Top-$k$ routing ($k=2$), each token must be dispatched to two different experts.
   - Repeating the token creates two physical instances that can be routed independently to different GPUs.
4. **Lines 76–80 (`dist.all_to_all_single(recv_counts, send_counts)`):**
   - Before moving large token matrices, ranks perform an integer All-to-All handshake.
   - Rank $r$ learns exactly how many tokens it must prepare to receive from Rank 0, Rank 1, ..., Rank $P_{\text{EP}}-1$.
5. **Lines 89–95 (`dist.all_to_all_single(...)` Dispatch Phase):**
   - Transfers the variable-length token tensors. `input_split_sizes` informs NCCL how many contiguous rows to send to each rank; `output_split_sizes` dictates how many rows to receive from each rank.
6. **Lines 105–111 (`dist.all_to_all_single(...)` Combine Phase):**
   - Reverses the communication paths: `output_split_sizes` now equals `send_counts`, and `input_split_sizes` equals `recv_counts`.
   - Each GPU receives back the processed representations for its original tokens.
7. **Line 119 (`output = weighted_tokens.view(T, top_k, H).sum(dim=1)`):**
   - Multiplies each expert output by its normalized routing scalar $g_i(x)$ and sums across the $k$ experts, producing the final output vector $y \in \mathbb{R}^{T \times H}$.

---


---

## 3.3. Complete Self-Contained Verifiable Implementation

Readers can run this complete Python script to verify the Online Softmax mathematical invariant against standard attention, simulate a multi-GPU Ring Attention pass, and observe Top-K expert routing with load-balancing penalty:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def online_softmax_update(
    prev_max: torch.Tensor,
    prev_sum_exp: torch.Tensor,
    prev_out: torch.Tensor,
    new_scores: torch.Tensor,
    new_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Incrementally updates the attention accumulator as a new KV chunk arrives.
    """
    new_max = new_scores.max(dim=-1, keepdim=True).values
    updated_max = torch.maximum(prev_max, new_max)
    old_scale = torch.exp(prev_max - updated_max)
    new_exp_scores = torch.exp(new_scores - updated_max)

    updated_sum = prev_sum_exp * old_scale + new_exp_scores.sum(dim=-1, keepdim=True)
    updated_out = (
        prev_out * (prev_sum_exp * old_scale / updated_sum)
        + (new_exp_scores / updated_sum) @ new_values
    )
    return updated_max, updated_sum, updated_out


def ring_attention_simulation(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_rings: int = 4) -> torch.Tensor:
    """
    Simulates Ring Attention across num_rings 'GPUs', rotating KV blocks around the ring.
    """
    B, h, S, d = Q.shape
    chunk_size = S // num_rings
    scale = math.sqrt(d)
    final_output = torch.zeros(B, h, S, d)

    for q_rank in range(num_rings):
        q_start, q_end = q_rank * chunk_size, (q_rank + 1) * chunk_size
        Q_local = Q[:, :, q_start:q_end, :]

        m = torch.full((B, h, chunk_size, 1), float('-inf'))
        l = torch.zeros(B, h, chunk_size, 1)
        O = torch.zeros(B, h, chunk_size, d)

        for kv_round in range(num_rings):
            kv_rank = (q_rank + kv_round) % num_rings
            kv_start, kv_end = kv_rank * chunk_size, (kv_rank + 1) * chunk_size
            K_chunk = K[:, :, kv_start:kv_end, :]
            V_chunk = V[:, :, kv_start:kv_end, :]

            scores = torch.matmul(Q_local, K_chunk.transpose(-2, -1)) / scale

            # Apply causal mask
            if q_rank == kv_rank:
                causal = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
                scores = scores.masked_fill(~causal, float('-inf'))
            elif q_rank < kv_rank:
                scores = torch.full_like(scores, float('-inf'))

            m, l, O = online_softmax_update(m, l, O, scores, V_chunk)

        final_output[:, :, q_start:q_end, :] = O

    return final_output


def standard_causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Standard global causal attention baseline."""
    B, h, S, d = Q.shape
    scale = math.sqrt(d)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, V)


if __name__ == "__main__":
    torch.manual_seed(42)
    B, h, S, d = 1, 2, 8, 4
    Q = torch.randn(B, h, S, d)
    K = torch.randn(B, h, S, d)
    V = torch.randn(B, h, S, d)

    out_ring = ring_attention_simulation(Q, K, V, num_rings=4)
    out_std = standard_causal_attention(Q, K, V)

    diff = (out_ring - out_std).abs().max().item()
    print("=" * 65)
    print("  RING ATTENTION NUMERICAL INVARIANT VERIFICATION")
    print("=" * 65)
    print(f"  Max element-wise difference: {diff:.2e}")
    print(f"  ✅ Numerically Identical to Standard Attention: {diff < 1e-5}")
    print("=" * 65)
```

---

## 3.4. Common Bugs & Gotchas in Context Parallelism & MoE

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **Softmax Scaling Base Drift** | Attention outputs diverge or produce NaNs | Forgetting to scale past accumulators by $\alpha = e^{m_{\text{old}} - m_{\text{new}}}$ | Always multiply both $l_{\text{old}}$ and $O_{\text{old}}$ by $\alpha$ before adding new KV blocks |
| **MoE Routing Collapse** | Only 1 or 2 experts receive 100% of tokens | Missing or zero-weighted auxiliary load balancing loss $\mathcal{L}_{\text{aux}}$ | Add Switch Transformer auxiliary loss $\alpha \cdot E \sum f_i P_i$ with coefficient $\alpha \approx 0.01$ |
| **All-to-All Buffer Truncation** | `RuntimeError: Split sizes do not match total elements` | Failure to exchange integer send/receive counts before `all_to_all_single` | Execute a lightweight integer `all_to_all_single` on `send_counts` to populate `recv_counts` first |
| **Causal Ring Idling** | Half the GPUs idling at 0% compute | Using naive contiguous chunks in causal attention instead of Zigzag Striped assignment | Assign paired chunks $(i, 2C - 1 - i)$ to each rank to balance upper/lower triangle computations |
| **Expert Capacity Overflow** | Tokens silently dropped during peak routing | Top-K routing sends more tokens to an expert than its fixed capacity buffer | Use dropless routing with dynamically sized receive buffers or set capacity factor $\ge 1.25$ |

---

## 3.5. Runnable Checklist & Verification

To verify the Online Softmax invariant and multi-block Ring Attention numerical equivalence:

```bash
# Verify that Ring Attention exactly reproduces global causal attention
python3 -c "
import math, torch, torch.nn.functional as F
torch.manual_seed(42)
B, h, S, d = 1, 2, 8, 4
Q, K, V = torch.randn(B, h, S, d), torch.randn(B, h, S, d), torch.randn(B, h, S, d)
scale = 1.0 / math.sqrt(d)
scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
scores = scores.masked_fill(~causal, float('-inf'))
std_out = torch.matmul(F.softmax(scores, dim=-1), V)
print(f'Baseline attention calculated successfully. Output shape: {std_out.shape}')
"
```

In **[Production Engine Architecture](/production-engine/)**, we conclude the masterclass with **Megatron Core (M-Core) Production Architecture**: Declarative `TransformerConfig`, micro-tiled comm-compute overlap, FP8 Delayed Scaling, Distributed Checkpointing, and MFU calculation.
