# Megatron Core: Architecture Manual & Distributed Systems Curriculum

> **A Comprehensive, First-Principles Specification for Extreme-Scale Deep Learning**
> Covers: 1D Tensor Parallelism, Sequence Parallelism, Pipeline Parallelism (1F1B & DualPipe), ZeRO-1/2 Distributed Optimizer, Context Parallelism (Ring Attention), MoE Expert Parallelism, and Megatron Core Production Architecture.
>
> All distributed algorithms run locally on **Mac CPU** via the PyTorch `gloo` backend.  
> Use `/opt/anaconda3/bin/python3` and `torchrun --nproc_per_node=N` throughout.

---

## 🏛️ Curriculum Structure & Architecture Parts

### Part I: Distributed Foundations & Intra-Node Scaling (The NVLink Domain)
| Ch | Module Document | Key Topics | Implementation Artifact |
|:---:|---|---|---|
| **01** | [Distributed Compute Foundations & Interconnect Topologies](./chapters/01_foundations_distributed_torch.md) | Hardware hierarchy, Ring/Tree/NVLS All-Reduce, process groups, autograd conjugate operators, MFU/HFU | Full autograd conjugate demo |
| **02** | [1D Tensor Parallelism & Linear Operator Sharding](./chapters/02_tensor_parallelism_math_and_layers.md) | Column/Row parallel, $f$ & $g$ operators, vocab parallel, ParallelCrossEntropy, Tensor Core tile alignment, residual scaling | Full TP Transformer Block & Vocab Parallel |
| **03** | [Sequence Parallelism & Dynamic Activation Management](./chapters/03_sequence_parallelism_and_activations.md) | RS+AG == AllReduce proof, SP LayerNorm, selective activation recomputation, memory economics | SP module + selective checkpointing |

### Part II: Inter-Node Scaling & Memory Paradigms (The Scale-Out Fabric)
| Ch | Module Document | Key Topics | Implementation Artifact |
|:---:|---|---|---|
| **04** | [Pipeline Parallelism, Distributed Schedules & DualPipe](./chapters/04_pipeline_parallelism_and_schedules.md) | GPipe vs 1F1B, exact bubble formulas, Virtual PP, DualPipe $B_{\text{input}} \perp B_{\text{weight}}$ overlap, P2P comm | 1F1B layer assignment & 2-stage P2P pipeline |
| **05** | [Memory Accounting & The Megatron Distributed Optimizer](./chapters/05_memory_accounting_and_distributed_optimizer.md) | 16 bytes/param breakdown, ZeRO-1/2 from scratch, IEEE 754 swamping theorem, 70B parameter memory math | Full ZeRO-2 Distributed Optimizer & swamping proof |

### Part III: Extreme-Scale Frontiers: Context Length & Sparsity
| Ch | Module Document | Key Topics | Implementation Artifact |
|:---:|---|---|---|
| **06** | [Context Parallelism & Expert Parallelism (MoE)](./chapters/06_context_parallelism_and_moe.md) | Online softmax with NaN guard, Ring Attention, double-buffering ping-pong, Top-K router with capacity factor | Online softmax recurrence, Ring P2P, MoE router |
| **07** | [Megatron Core Production Architecture & Acceleration Primitives](./chapters/07_megatron_core_production_architecture.md) | M-Core ModuleSpec, Comm-Compute Overlap, FP8 Delayed Scaling, Muon research optimizer, FSDP2 comparison | Complete M-Core production suite & Muon |

### Part IV: Production Runtime & Infrastructure Specifications
| Ref | Reference Document | Key Topics | Implementation Artifact |
|:---:|---|---|---|
| **A** | [High-Throughput Token Pipelines & Sequence Packing](./chapters/appendix_a_data_pipeline.md) | Binary indexed format (.bin/.idx), MMapIndexedDataset, blended datasets, unpadded sequence packing | Full memory-mapped reader & sequence packer |
| **B** | [Production Cluster Resilience & Fault Tolerance](./chapters/appendix_b_cluster_reliability.md) | Hardware MTBF, elastic rendezvous (c10d), NCCL watchdogs/heartbeats, async checkpointing | Resilient elastic training simulation & signal trap |
| **C** | [Accelerator Silicon Matrix & Superchip Topologies](./chapters/appendix_c_hardware_superchips.md) | Silicon matrix (A100/H100/B200/NVL72), InfiniBand NDR/XDR, NVLink 5, Megatron vs DeepSpeed vs FSDP2 | Complete architectural comparison tables |

---

## 🖥️ Running Code Directly From The Chapters

Every single chapter in this curriculum is **100% self-contained**:
- All algorithms, mathematical proofs, architectural traces, and PyTorch implementations are embedded directly within each Markdown document.
- Readers can copy and paste any snippet directly into a Python script, terminal, or Jupyter notebook.
- All distributed code is designed to run seamlessly on standard CPU hardware using PyTorch's `gloo` backend without requiring access to an NVIDIA GPU cluster.

For example, to execute the foundational distributed conjugate dual operator test:
```bash
torchrun --nproc_per_node=2 demo_megatron.py
```

---

## 🗺️ Concept Map

```
                    MEGATRON CORE PARALLELISM STACK
                    ═══════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │                    DATA PARALLELISM (DP)                 │
  │    Gradient sync: AllReduce (DDP) or RS+AG (ZeRO-2)     │
  │    Optimizer states sharded across D ranks               │
  └─────────────────────────────────────────────────────────┘
                              │
  ┌─────────────────────────────────────────────────────────┐
  │                 PIPELINE PARALLELISM (PP)                │
  │    Layers split across P stages (GPipe / 1F1B / VPP)    │
  │    Point-to-point activation tensors via batch_isend     │
  └─────────────────────────────────────────────────────────┘
                              │
  ┌─────────────────────────────────────────────────────────┐
  │              TENSOR PARALLELISM (TP) + SP               │
  │    Weights split: ColumnParallel → f-op → RowParallel   │
  │    f: Identity fwd / AllReduce bwd                       │
  │    g: AllReduce fwd / Identity bwd                       │
  │    SP: Replace AllReduce with ReduceScatter + AllGather  │
  └─────────────────────────────────────────────────────────┘
                              │
  ┌─────────────────────────────────────────────────────────┐
  │              CONTEXT PARALLELISM (CP)                    │
  │    Sequence split: Ring Attention across S/N tokens      │
  │    Online softmax: m,l,O recurrence across KV chunks     │
  └─────────────────────────────────────────────────────────┘
                              │
  ┌─────────────────────────────────────────────────────────┐
  │              EXPERT PARALLELISM (EP / MoE)               │
  │    Tokens routed via Top-K gate to E expert GPUs         │
  │    AlltoAll dispatch + combine; aux load-balance loss    │
  └─────────────────────────────────────────────────────────┘
```

---

## 📐 Key Formulas Cheatsheet

| Formula | Meaning |
|---|---|
| $\text{Volume}_{\text{AllReduce}} = 2\frac{N-1}{N}S$ | Ring All-Reduce comm per rank |
| $\text{Volume}_{\text{RS}} + \text{Volume}_{\text{AG}} = 2\frac{N-1}{N}S$ | Why SP has zero extra comm |
| $\text{Bubble} = \frac{p-1}{m+p-1}$ | 1F1B pipeline bubble fraction |
| $\text{Mem}_{\text{ZeRO-2}} = 2\Phi + \frac{14\Phi}{D}$ | Per-GPU memory with ZeRO-2 |
| $\text{FLOPs/token} \approx 6\Phi + 12Lhd_\text{head}S$ | Analytic training FLOPs |
| $\text{MFU} = \frac{\text{FLOPs/step}}{\text{Peak TFLOPS} \times t_\text{step} \times N_\text{GPU}}$ | Model FLOPs Utilization |

---

## 🔑 The Three Core Papers

1. **[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)** — Megatron-LM v1: Column/Row parallel, vocab parallel
2. **[arXiv:2104.04473](https://arxiv.org/abs/2104.04473)** — Megatron-LM v2: Pipeline parallelism, 3D parallelism, 1F1B schedule
3. **[arXiv:2205.05198](https://arxiv.org/abs/2205.05198)** — Megatron-LM v3: Sequence parallelism, selective activation recomputation
