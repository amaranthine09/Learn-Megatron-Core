# Megatron Core: Architecture Manual & Technical Documentation

> **A Modular, First-Principles Technical Documentation Suite for Extreme-Scale Deep Learning**  
> Covers: 1D Tensor Parallelism, Sequence Parallelism, Pipeline Parallelism (1F1B & Virtual Stages), ZeRO-1/2 Distributed Optimizer, Context Parallelism (Ring Attention), MoE Expert Parallelism, and Megatron Core Production Architecture.
>
> All distributed algorithms run locally on **Mac CPU** via the PyTorch `gloo` backend.  
> Use `/opt/anaconda3/bin/python3` and `torchrun --nproc_per_node=N` throughout.

---

## 🏛️ Modular Documentation Portal

Every major parallelism paradigm is organized into **focused, bite-sized articles** with rigorous **hierarchical headings (`1.1`, `1.1.1`, `1.1.1.1`)**, mathematical proofs, and runnable implementation labs.

### 1. Foundations & Interconnects
- [1. Foundations & Interconnect Topologies](./chapters/foundations/index.md): Scale-up vs. scale-out bandwidth, Ring/Tree/NVLS All-Reduce algorithms.
- [2. Autograd Mechanics & Memory Layouts](./chapters/foundations/autograd-and-memory.md): Multi-dimensional process groups, custom autograd conjugate hooks, and caching allocators.
- [3. Foundations Lab & Verification](./chapters/foundations/implementation.md): Runnable CPU verification scripts, NCCL selection heuristics, and MFU/HFU arithmetic.

### 2. 1D Tensor Parallelism (TP)
- [1. Complementary GEMM Factoring](./chapters/tensor-parallelism/index.md): Why naive model parallelism failed, Column/Row parallel pairing, and autograd $f$/$g$ operators.
- [2. Vocab Parallelism & Tensor Cores](./chapters/tensor-parallelism/vocab-and-advanced.md): `VocabParallelEmbedding`, Parallel Cross-Entropy, GQA/MQA sharding, and Tensor Core tile alignment.
- [3. Complete PyTorch Implementation](./chapters/tensor-parallelism/implementation.md): 350-line standalone PyTorch implementation, $N \times b$ bias gotchas, and production checklists.

### 3. Sequence Parallelism (SP)
- [1. Zero-Overhead Activation Sharding](./chapters/sequence-parallelism/index.md): Memory scaling limits, the $\text{RS} + \text{AG} \equiv \text{AllReduce}$ identity, and sharded LayerNorm/Dropout.
- [2. Memory Economics & PyTorch Lab](./chapters/sequence-parallelism/implementation.md): Selective activation recomputation vs checkpointing, autograd mappings, and runnable verification.

### 4. Pipeline Parallelism (PP)
- [1. 1F1B & Interleaved Virtual Stages](./chapters/pipeline-parallelism/index.md): Vertical layer partitioning, GPipe baseline, 1F1B steady-state, and virtual pipeline ($v$) bubble math.
- [2. DeepSeek DualPipe & P2P Comm](./chapters/pipeline-parallelism/dualpipe-and-p2p.md): Non-blocking batched P2P, deadlock avoidance, and DualPipe $B_{\text{input}} \perp B_{\text{weight}}$ overlap.
- [3. 2-Stage Pipeline Implementation](./chapters/pipeline-parallelism/implementation.md): Runnable 2-stage P2P pipeline, rank assignments, and failure diagnostics.

### 5. Memory & Distributed Optimizer (ZeRO)
- [1. The 16-Bytes Law & Swamping Proof](./chapters/distributed-optimizer/index.md): IEEE 754 mantissa swamping theorem, FP32 master weights, and the ZeRO-1/2/3 hierarchy.
- [2. Buffer Layouts & Overlap Math](./chapters/distributed-optimizer/buffers-and-overlap.md): `ParamAndGradBuffer` contiguous flattening, async Reduce-Scatter overlap, and 70B/175B memory math.
- [3. ZeRO-2 Optimizer Implementation](./chapters/distributed-optimizer/implementation.md): Complete Megatron `DistributedOptimizer` implementation and numerical validation.

### 6. Context Parallelism (CP) & MoE
- [1. Ring Attention & Online Softmax](./chapters/context-parallelism/index.md): Quadratic memory explosion, Online Softmax mathematical recurrence, and double-buffered Ring P2P.
- [2. Mixture of Experts & EP Dispatch](./chapters/context-parallelism/mixture-of-experts.md): Switch Transformers, Top-K routing, capacity factor, and All-to-All token dispatch.
- [3. CP & MoE Runnable Verification](./chapters/context-parallelism/implementation.md): Runnable Ring Attention, MoE router lab, and failure mode checklists.

### 7. Production Pretraining Engine
- [1. M-Core Declarative ModuleSpec](./chapters/production-engine/index.md): Legacy Megatron vs. M-Core, `TransformerConfig`, and `ModuleSpec` component trees.
- [2. FP8 Scaling & State Checkpointing](./chapters/production-engine/fp8-and-checkpointing.md): FP8 Delayed Scaling (`E4M3`/`E5M2`), Transformer Engine amax buffers, and sharded state dicts.
- [3. Dynamic-CP & FSDP2 Extensions](./chapters/production-engine/modern-innovations.md): Dynamic Context Parallelism solvers, Megatron-FSDP2 module sharding, and cluster checklists.

### 8. Cluster Operations & Silicon Matrix
- [Token Ingestion Pipelines & Sequence Packing](./chapters/operations/data-pipeline.md): Binary indexed datasets (`.bin`/`.idx`), `MMapIndexedDataset`, and unpadded sequence packing.
- [Cluster Resilience & SRE Diagnostics](./chapters/operations/cluster-resilience.md): Hardware MTBF, elastic rendezvous (`c10d`), NCCL watchdogs, and async checkpointing.
- [Accelerator Silicon & Systems Matrix](./chapters/operations/silicon-and-systems.md): H100 vs. Blackwell GB200 NVL72, InfiniBand fabrics, and Megatron vs. DeepSpeed vs. FSDP2.

---

## 🗺️ The Distributed Parallelism Stack

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

## 📄 License

This project and its documentation are open-source and licensed under the [Apache License 2.0](https://github.com/amaranthine09/Learn-Megatron-Core/blob/main/LICENSE).
