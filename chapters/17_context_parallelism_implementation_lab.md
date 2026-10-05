# Context Parallelism & MoE: API Reference
> **Megatron-Core Ring Attention, FlashAttention Integration, MoE Router, and Expert Parallel Dispatch**

---

## 3.1. Context Parallelism in Megatron-Core

Context Parallelism (CP) is enabled via `TransformerConfig` and managed through `megatron.core.parallel_state`:

```python
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    context_parallel_size=8,        # Split sequence across 8 CP ranks
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    ...
)
```

The CP process group is initialized in `megatron.core.parallel_state.initialize_model_parallel`:

```python
from megatron.core import parallel_state

# After initialize_model_parallel():
cp_group = parallel_state.get_context_parallel_group()
cp_rank  = parallel_state.get_context_parallel_rank()
cp_size  = parallel_state.get_context_parallel_world_size()
```

Each CP rank holds a sequence shard of shape `[B, S/CP, H]`. The full sequence is only reconstructed inside the attention kernel via ring communication.

---

## 3.2. Ring Attention with Online Softmax: Core Operators

The Ring Attention forward pass rotates KV blocks across CP ranks while each rank's queries remain stationary. The key primitives used in `megatron/core/transformer/dot_product_attention.py`:

### 3.2.1 Online Softmax Recurrence

Each ring step updates three running accumulators without materializing the full `S * S` attention matrix:

```python
# Incremental update for one rotating KV block

# m_prev, l_prev: [B, h, S_local, 1]  — running max and normalization sum
# O_prev:         [B, h, S_local, d]  — running output accumulator
# K_block, V_block: arriving KV shard from neighboring CP rank

scores = torch.matmul(Q_local, K_block.transpose(-2, -1)) * scale   # [B, h, S_local, S_kv]
m_local = scores.amax(dim=-1, keepdim=True)                          # local block max
m_new   = torch.maximum(m_prev, m_local)                             # global running max

alpha   = torch.exp(m_prev - m_new)                                  # rescale factor ∈ (0, 1]
exp_s   = torch.exp(scores - m_new)                                  # re-normalized exp scores

l_new = l_prev * alpha + exp_s.sum(dim=-1, keepdim=True)            # update denominator
O_new = (O_prev * (l_prev * alpha) + torch.matmul(exp_s, V_block)) / (l_new + 1e-8)
```

> [!IMPORTANT]
> The `alpha = exp(m_prev - m_new)` rescaling step is **mandatory**. If the new KV block contains scores larger than any seen previously, the old accumulator `O_prev` was normalized to a stale maximum and must be rescaled before adding the new contribution. Omitting this step causes attention outputs to diverge from standard attention by an unbounded factor.

---

### 3.2.2 P2P Ring Communication Pattern

Each ring step fires non-blocking bidirectional sends and receives for K and V tensors:

```python
# megatron/core/transformer/dot_product_attention.py (CP ring loop)

send_dst = (cp_rank + 1) % cp_size
recv_src = (cp_rank - 1 + cp_size) % cp_size

for step in range(cp_size):
    # Asynchronously launch K/V transfer to next rank while computing on current buffers
    if step < cp_size - 1:
        p2p_ops = [
            dist.P2POp(dist.isend, K_buf[curr].contiguous(), send_dst, group=cp_group),
            dist.P2POp(dist.isend, V_buf[curr].contiguous(), send_dst, group=cp_group),
            dist.P2POp(dist.irecv, K_buf[next_],             recv_src, group=cp_group),
            dist.P2POp(dist.irecv, V_buf[next_],             recv_src, group=cp_group),
        ]
        reqs = dist.batch_isend_irecv(p2p_ops)

    # Compute on current buffers (overlaps with network transfer above)
    m, l, O = online_softmax_update(m, l, O, Q_local, K_buf[curr], V_buf[curr], ...)

    if step < cp_size - 1:
        for req in reqs: req.wait()   # Sync before next step
        curr, next_ = next_, curr     # Ping-pong buffer swap
```

The double-buffering (ping-pong) pattern ensures that compute and communication execute concurrently — Tensor Cores process the current KV block while NVLink/IB transfers the next block.

---

### 3.2.3 FlashAttention Integration

In production, Megatron replaces the manual `torch.matmul` attention with `flash_attn_varlen_func` for O(1) HBM footprint:

```python
from flash_attn.flash_attn_interface import flash_attn_varlen_func

# Called inside each ring step instead of manual QK^T + softmax:
output = flash_attn_varlen_func(
    q=Q_local,                        # [total_tokens, h, d]
    k=K_block,                        # [total_tokens, h, d]
    v=V_block,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    dropout_p=0.0,
    causal=causal,
    softmax_scale=scale,
)
```

The flash kernel fuses QK^T, online softmax, and AV into a single CUDA kernel, eliminating the intermediate `[B, h, S, S]` score matrix from HBM entirely.

---

## 3.3. Mixture of Experts: Router & Expert Parallel Dispatch

MoE in Megatron-Core lives in `megatron.core.transformer.moe`. The two-phase All-to-All dispatch pattern is the core communication primitive:

### 3.3.1 TopK Router (megatron.core.transformer.moe.router)

```python
# megatron/core/transformer/moe/router.py

class TopKRouter(Router):
    """
    Top-K gating with auxiliary load-balancing loss.
    Computes: logits → softmax → top-k selection → capacity enforcement.
    """

    def routing(self, logits: torch.Tensor):
        """
        Args:
            logits: [S*B, E] — raw router logits for E experts
        Returns:
            scores:  [S*B, top_k] — normalized routing weights
            indices: [S*B, top_k] — selected expert indices
        """
        if self.config.moe_router_score_function == 'softmax':
            scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        else:
            scores = torch.sigmoid(logits)

        # Top-K selection
        top_k_scores, top_k_indices = torch.topk(scores, k=self.topk, dim=1)

        # Auxiliary load-balancing loss (Switch Transformer formulation)
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability to expert i
        # L_aux = num_experts * sum(f_i * P_i)
        if self.training:
            self.aux_loss = self._compute_load_balancing_loss(scores, top_k_indices)

        return top_k_scores, top_k_indices
```

---

### 3.3.2 Expert Parallel Token Dispatch (All-to-All)

```python
# megatron/core/transformer/moe/token_dispatcher.py

class MoEAlltoAllTokenDispatcher:
    """
    Two-phase token routing using dist.all_to_all_single for Expert Parallelism.
    Phase 1 (Dispatch): Send tokens to their assigned expert ranks.
    Phase 2 (Combine): Receive processed outputs back from expert ranks.
    """

    def token_permutation(self, hidden_states, max_prob, max_ind):
        """
        PHASE 1: DISPATCH
        Sorts tokens by destination EP rank and fires All-to-All.
        """
        # Exchange token counts: every rank learns recv volumes
        dist.all_to_all_single(
            self.output_splits,    # recv counts from each EP rank
            self.input_splits,     # send counts to each EP rank
            group=self.ep_group,
        )

        # Sort tokens by destination rank for contiguous send buffers
        self.hidden_shape = hidden_states.shape
        permuted_input = hidden_states[self.sort_order]

        # Dispatch: variable-length All-to-All of token embeddings
        dist.all_to_all_single(
            output=self.dispatched_input,        # Pre-allocated recv buffer
            input=permuted_input,
            output_split_sizes=self.output_splits.tolist(),
            input_split_sizes=self.input_splits.tolist(),
            group=self.ep_group,
        )
        return self.dispatched_input

    def token_unpermutation(self, expert_output, bias=None):
        """
        PHASE 2: COMBINE
        Returns processed expert outputs back to token origin ranks.
        """
        dist.all_to_all_single(
            output=self.restored_output,
            input=expert_output,
            output_split_sizes=self.input_splits.tolist(),   # Reversed
            input_split_sizes=self.output_splits.tolist(),   # Reversed
            group=self.ep_group,
        )

        # Restore original token order
        output = self.restored_output[self.restore_order]
        return output
```

> [!NOTE]
> **Why two separate All-to-All calls?** The first lightweight integer All-to-All (`output_splits`, `input_splits`) is a mandatory handshake — each rank must know how large a receive buffer to allocate before moving the large token tensors. Skipping this leads to `RuntimeError: Split sizes do not match total elements`.

---

### 3.3.3 Capacity Factor & Token Dropping

```python
# Expert capacity: maximum tokens per expert per batch
expert_capacity = math.ceil(
    config.moe_expert_capacity_factor * tokens_per_expert_if_balanced
)
# tokens_per_expert_if_balanced = (S * B * top_k) / num_experts

# Tokens exceeding expert_capacity are dropped (masked to zero weight)
# Megatron supports dropless routing via dynamic buffer sizing:
config = TransformerConfig(
    moe_expert_capacity_factor=1.25,    # 25% overflow buffer
    moe_pad_expert_input_to_capacity=False,  # Dropless: dynamic buffer
    moe_token_drop_policy='probs',      # Drop lowest-probability tokens first
    ...
)
```

---

## 3.4. Common Bugs & Gotchas in Context Parallelism & MoE

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **Softmax Scaling Base Drift** | Attention outputs diverge or produce NaNs | Forgetting to scale past accumulators by `alpha = e^m_old - m_new` | Always multiply both l_old and O_old by alpha before adding new KV blocks |
| **MoE Routing Collapse** | Only 1 or 2 experts receive 100% of tokens | Missing or zero-weighted auxiliary load balancing loss Loss_aux | Add Switch Transformer auxiliary loss `alpha * E sum f_i P_i` with coefficient `alpha ≈ 0.01` |
| **All-to-All Buffer Truncation** | `RuntimeError: Split sizes do not match total elements` | Failure to exchange integer send/receive counts before `all_to_all_single` | Execute a lightweight integer `all_to_all_single` on `send_counts` to populate `recv_counts` first |
| **Causal Ring Idling** | Half the GPUs idling at 0% compute | Using naive contiguous chunks in causal attention instead of Zigzag Striped assignment | Assign paired chunks `(i, 2C - 1 - i)` to each rank to balance upper/lower triangle computations |
| **Expert Capacity Overflow** | Tokens silently dropped during peak routing | Top-K routing sends more tokens to an expert than its fixed capacity buffer | Use dropless routing with `moe_pad_expert_input_to_capacity=False` or set capacity factor `>= 1.25` |

---

## 3.5. Summary & What's Next

In **[Production Engine Architecture](/production-engine/)**, we conclude the masterclass with **Megatron Core (M-Core) Production Architecture**: Declarative `TransformerConfig`, micro-tiled comm-compute overlap, FP8 Delayed Scaling, Distributed Checkpointing, and MFU calculation.
