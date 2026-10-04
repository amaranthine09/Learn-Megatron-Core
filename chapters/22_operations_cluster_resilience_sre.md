# Production Cluster Resilience & Fault Recovery
> **MTBF Economics, Elastic Rendezvous (c10d), NCCL Watchdogs, and Async Storage Prefetch**

> **Reference**: *NVIDIA NeMo Framework & PyTorch Distributed Elastic (c10d) Fault Tolerance Specification (2023–2025)*

---

## 1. The Physics of Frontier Training Failures: Mean Time Between Failures (MTBF)

Training a 70B–405B parameter model takes weeks or months across thousands of GPUs. In this regime, **hardware failure is not an anomaly — it is a mathematical certainty**:
- An individual GPU accelerator has an annual failure rate of $\sim 3\% - 5\%$ (due to HBM3 thermal cycling, voltage regulator drift, or SRAM bit flips).
- In a cluster of $16{,}384$ accelerators:
  $$\text{Expected Failures per Day} = \frac{16{,}384 \times 0.04}{365} \approx \mathbf{1.8 \text{ failures per day!}}$$
- Every 12 to 24 hours, an InfiniBand cable drops packets, a GPU throws an uncorrectable double-bit ECC error, or an NVLink bridge fails.

If your training harness crashes and requires manual human intervention to restart, **effective cluster training time drops below 50%**. Production Megatron Core deployments rely on **Elastic Rendezvous**, **Heartbeat Watchdogs**, and **Non-Blocking Distributed Checkpointing**.

---

## 2. `torchrun` Elastic Rendezvous Architecture (`c10d`)

Traditional MPI jobs require an immutable list of static IP addresses. If Node 47 dies, the entire MPI communicator collapses.

PyTorch Elastic (`torchrun`) uses a dynamic **Rendezvous Backend** (typically based on `c10d` using etcd or a designated master rank):

```
                       PyTorch Elastic Rendezvous Architecture
                       
                   ┌────────────────────────────────────────┐
                   │  ETCD / C10D Master Rendezvous Server  │
                   │    (Tracks Active Ranks & Heartbeats)  │
                   └───────────────────┬────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  Node 0 (Ranks 0..7)            Node 1 (Ranks 8..15)          Node 2 (Ranks 16..23)
  [Local Worker Agent]           [Local Worker Agent]          [Local Worker Agent]
        │                              │                              │
        │ Heartbeat OK                 │ Heartbeat OK                 │ Crash / ECC Error (X)
        ▼                              ▼                              ▼
  Active Training                Active Training               Node Dies!
                                                                      │
  ◄────── Trigger Re-Rendezvous & Reload Latest Checkpoint ───────────┘
```

### 2.1 Elastic Launch Command & Parameters
```bash
torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=master-node-01:29500 \
  --rdzv_id=megatron_70b_pretraining_job \
  --min_nodes=60 \
  --max_nodes=64 \
  --max_restarts=10 \
  pretrain_gpt.py
```
- `--min_nodes=60`: If 4 nodes crash, the cluster does NOT abort; it re-rendezvous on the surviving 60 nodes!
- `--max_restarts=10`: Automatically orchestrates up to 10 automatic failure recoveries before alerting engineering on-call.

---

## 3. NCCL Watchdogs, Heartbeats, and Deadlock Prevention

The most insidious failure in distributed deep learning is the **Silent Deadlock**:
- GPU 12 experiences an internal kernel freeze or hardware lockup.
- GPU 13 waits indefinitely for an incoming P2P `irecv`.
- All $16{,}384$ GPUs sit at 100% power consumption while performing zero work!

To prevent multi-hour deadlocks, production clusters configure **NCCL Heartbeat Watchdogs**:

```bash
# 1. Enable asynchronous error handling so exceptions propagate to Python
export NCCL_ASYNC_ERROR_HANDLING=1

# 2. Timeout threshold for blocking collective operations (e.g. 1,800 seconds = 30 minutes)
export NCCL_TIMEOUT=1800

# 3. Heartbeat watchdog interval for detecting hung CUDA streams
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

# 4. Dump detailed NCCL ring state on failure for automated root-cause analysis
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## 4. Signal Trapping (`SIGTERM`, `SIGUSR1`) & Slurm Preemption

On enterprise SLURM clusters or cloud spot instances, jobs are periodically preempted to make room for higher-priority workloads.
SLURM sends a warning signal (typically `SIGUSR1` or `SIGTERM`) 10–15 minutes before forcefully terminating the job:

```python
import signal
import sys

def register_graceful_shutdown_handlers(save_checkpoint_fn):
    """
    Registers OS signal handlers for graceful preemption under Slurm.
    """
    def handle_preemption(signum, frame):
        print(f"[Rank {dist.get_rank()}] Received preemption signal {signum}! Triggering emergency checkpoint...")
        save_checkpoint_fn()
        print("[Rank {dist.get_rank()}] Emergency checkpoint successfully saved. Exiting gracefully.")
        sys.exit(0)

    # Trap SLURM preemption signals
    signal.signal(signal.SIGUSR1, handle_preemption)
    signal.signal(signal.SIGTERM, handle_preemption)
```

---

## 5. Non-Blocking Asynchronous Checkpointing

Saving a 70B parameter model checkpoint (over $140\text{ GB}$ of weights and $280\text{ GB}$ of optimizer states) synchronously stalls GPU compute for 3–5 minutes. If checkpoints are saved every 500 steps, **$15\% - 25\%$ of total training time is lost to disk I/O**.

Megatron Core uses **Non-Blocking Asynchronous Checkpointing**:
1. **Device-to-Host Asynchronous DMA**: Updated parameter and optimizer shards are copied from GPU HBM into pinned Host CPU RAM in $< 1.5\text{ seconds}$ via PCIe Gen 5 ($64\text{ GB/s}$).
2. **Immediate Compute Resumption**: GPU Tensor Cores immediately resume forward passes for the next training iteration!
3. **Background Host I/O Thread**: A background Python multiprocessing thread streams the pinned host RAM buffers to parallel distributed storage (Lustre / Ceph / Amazon S3) asynchronously without blocking CUDA streams.

---

## 6. Megatron-Core Checkpoint & Signal Handling Patterns

### 6.1 SIGTERM Handler Pattern (production torchrun)

In production Megatron runs, a SIGTERM handler is registered to flush an emergency checkpoint before the process is preempted:

```python
# megatron/training/training.py (signal handling pattern)

import signal
import torch

def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler, ...):
    """
    Called by SIGTERM handler and scheduled checkpoint intervals.
    Saves asynchronously to avoid blocking GPU compute.
    """
    from megatron.core import dist_checkpointing

    sharded_state_dict = {
        "model":     model.sharded_state_dict(prefix=''),
        "optimizer": optimizer.sharded_state_dict(model_sharded_state_dict=model.sharded_state_dict()),
    }

    dist_checkpointing.save(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=f"{args.save}/iter_{iteration:07d}",
        sharded_strategy=('zarr', 1),      # Parallel write to Lustre / S3
        async_sharded_save=True,           # Returns immediately; GPU resumes
    )

# Register handler
signal.signal(signal.SIGTERM, lambda *_: save_checkpoint_and_time(iteration, ...))
```

---

### 6.2 Distributed Checkpoint Load & Resharding

```python
from megatron.core import dist_checkpointing

# Load checkpoint — topology may differ from save time (TP/PP can change)
sharded_state_dict = {
    "model":     model.sharded_state_dict(prefix=''),
    "optimizer": optimizer.sharded_state_dict(...),
}

state_dict = dist_checkpointing.load(
    sharded_state_dict=sharded_state_dict,
    checkpoint_dir="/checkpoints/iter_0100000",
    sharded_strategy=('zarr', 1),
    validate_access_integrity=True,    # Verifies no shard is missing
)

model.load_state_dict(state_dict["model"])
optimizer.load_state_dict(state_dict["optimizer"])
```

> [!NOTE]
> `dist_checkpointing.load` dynamically recalculates slice intersections between the saved topology (e.g., TP=8, PP=4) and the current topology (e.g., TP=4, PP=2). No manual resharding is needed — this is the key advantage over raw `torch.save`/`torch.load`.

---

### 6.3 torchrun Elastic Launch Command

```bash
torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --max-restarts=3 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  pretrain_gpt.py \
  --num-layers 96 \
  --hidden-size 12288 \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --save /checkpoints/gpt4-175b \
  --load /checkpoints/gpt4-175b \
  --save-interval 1000
```

`--max-restarts=3` allows torchrun to respawn the entire process group up to 3 times on node failure, automatically reloading from the last checkpoint via `--load`.

