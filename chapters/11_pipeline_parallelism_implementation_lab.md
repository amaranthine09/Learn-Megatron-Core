# Pipeline Parallelism: P2P API Reference & Failure Diagnostics
> **Megatron-Core batch_isend_irecv Patterns and Pipeline Bug Diagnostics**

---

## 3.1. Megatron-Core P2P Communication API

Pipeline parallelism in Megatron-Core is implemented via non-blocking bidirectional P2P communication using `dist.batch_isend_irecv`. This is the exact pattern from `megatron/core/pipeline_parallel/p2p_communication.py`:

### 3.1.1 Core P2P Send/Recv Pattern

```python
# megatron/core/pipeline_parallel/p2p_communication.py

def _communicate(
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sends/receives tensors to/from the next/previous pipeline stage.
    Uses dist.batch_isend_irecv for non-blocking bidirectional communication
    to avoid deadlocks between adjacent stages.
    """
    ops = []

    if tensor_send_prev is not None:
        send_prev_op = dist.P2POp(
            dist.isend,
            tensor_send_prev,
            get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        ops.append(send_prev_op)

    if recv_prev:
        tensor_recv_prev = torch.empty(
            tensor_shape,
            requires_grad=recv_prev,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
        recv_prev_op = dist.P2POp(
            dist.irecv,
            tensor_recv_prev,
            get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        ops.append(recv_prev_op)

    if tensor_send_next is not None:
        send_next_op = dist.P2POp(
            dist.isend,
            tensor_send_next,
            get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        ops.append(send_next_op)

    if recv_next:
        tensor_recv_next = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
        recv_next_op = dist.P2POp(
            dist.irecv,
            tensor_recv_next,
            get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        ops.append(recv_next_op)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return tensor_recv_prev, tensor_recv_next
```

> [!IMPORTANT]
> **Why `batch_isend_irecv` and not `send`/`recv`?** Two adjacent stages simultaneously calling synchronous `dist.send()` to each other creates a deadlock — both are blocked waiting for the other to post a receive. `batch_isend_irecv` atomically posts all non-blocking operations together, letting the NCCL engine resolve ordering.

---

### 3.1.2 1F1B Steady-State Schedule (Forward/Backward Interleaving)

Megatron's `forward_backward_pipelining_without_interleaving` in `megatron/core/pipeline_parallel/schedules.py` implements 1F1B:

```python
# megatron/core/pipeline_parallel/schedules.py (simplified structure)

# ── Warmup Phase: push p-1 forward microbatches ──────────────────────
for k in range(num_warmup_microbatches):
    input_tensor = recv_forward(tensor_shape, config)
    output_tensor = forward_step(input_tensor, microbatch_k)
    send_forward(output_tensor, config)
    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)

# ── 1F1B Steady State ─────────────────────────────────────────────────
for k in range(num_microbatches - num_warmup_microbatches):
    # Simultaneous recv-forward + send-backward (non-blocking)
    input_tensor, output_tensor_grad = send_forward_recv_backward(
        output_tensors[-1], tensor_shape, config
    )

    # Backward pass on oldest stored activation (frees memory)
    input_tensor_grad = backward_step(
        input_tensors.pop(0),
        output_tensors.pop(0),
        output_tensor_grad,
    )

    # Simultaneous send-backward + recv-forward
    input_tensor = send_backward_recv_forward(input_tensor_grad, tensor_shape, config)
    output_tensor = forward_step(input_tensor, microbatch_k)
    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)

# ── Cooldown Phase: drain remaining backward passes ───────────────────
for k in range(num_warmup_microbatches):
    output_tensor_grad = recv_backward(tensor_shape, config)
    input_tensor_grad = backward_step(
        input_tensors.pop(0),
        output_tensors.pop(0),
        output_tensor_grad,
    )
    send_backward(input_tensor_grad, config)
```

**Memory invariant**: At any point during steady-state, at most $p - 1$ microbatch activations are live in memory simultaneously (where $p$ is the pipeline depth). This is the defining property of 1F1B vs. GPipe's $m$ simultaneous activations.

---

### 3.1.3 Pipeline Bubble Fraction

$$\text{Bubble Fraction} = \frac{p - 1}{m + p - 1}$$

Where $p$ = pipeline stages, $m$ = number of microbatches per global batch.

| Microbatches $m$ | Bubble % ($p=8$) | Efficiency |
|---|---|---|
| 8 | 46.7% | 53.3% |
| 16 | 31.8% | 68.2% |
| 32 | 18.4% | 81.6% |
| 64 | 10.1% | 89.9% |
| 128 | 5.4% | 94.6% |

Megatron-Core's interleaved 1F1B schedule (`num_model_chunks > 1`) reduces the bubble to:
$$\text{Bubble Fraction (Interleaved)} = \frac{1}{m} \cdot \frac{p-1}{V}$$
where $V$ is the number of virtual pipeline stages per physical rank.

---

## 3.2. Common Bugs & Gotchas in Pipeline Parallelism

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **Blocking P2P Deadlock** | Cluster freezes permanently on step 0 | Two adjacent ranks simultaneously calling synchronous `dist.send()` to each other | Always post receives before sends, or use `dist.batch_isend_irecv()` |
| **Warmup OOM Spike** | Stage 0 crashes with OOM during warmup | Attempting GPipe schedule with $m=64$, keeping 64 forward activations alive | Use 1F1B schedule to cap in-flight microbatches to $\le p$ |
| **Layer Imbalance Stalls** | High bubble overhead despite high $m$ | Stage 0 holding Embedding Table + 8 layers, while middle stages hold 8 layers | Allocate 1 fewer layer to Stage 0 and Stage $p-1$ to account for embedding and LM head FLOPs |
| **Non-Contiguous Activation Slice**| `RuntimeError: P2P op requires contiguous tensor` | Passing a sliced activation tensor directly into `isend` | Call `tensor.contiguous()` before passing to `dist.P2POp` |
| **Rank-to-Stage Misalignment** | Activations sent to the wrong physical node | Assuming rank 1 is always stage 1 in 3D parallelism ($TP \times PP \times DP$) | Compute stage index via `get_pipeline_model_parallel_rank()` from `megatron.core.parallel_state` |

---

## 3.3. Summary & What's Next

In **[Distributed Optimizer & ZeRO-2](/distributed-optimizer/)**, we explore **Memory Accounting** and the **Megatron Distributed Optimizer (ZeRO-1 / ZeRO-2)**: how to eliminate parameter and optimizer state redundancy across the Data Parallel dimension.
