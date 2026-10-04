# Megatron Core: Architecture Manual & Technical Documentation

> **A Modular, First-Principles Technical Documentation Suite for Extreme-Scale Deep Learning**  
> Covers: 1D Tensor Parallelism, Sequence Parallelism, Pipeline Parallelism (1F1B & Virtual Stages), ZeRO-1/2 Distributed Optimizer, Context Parallelism (Ring Attention), MoE Expert Parallelism, and Megatron Core Production Architecture.
>
> All specifications correspond directly to production **NVIDIA Megatron-Core (`megatron.core`)** APIs and CUDA NCCL runtimes.

---

## 🏛️ Complete Reading Order (Chapters 01 to 23)

The raw Markdown specification is organized into sequentially ordered chapters in the [`chapters/`](./chapters/) directory:

| Ch | Document | Core System Focus | Key Concepts & Primitives |
|:---:|---|---|---|
| **01** | [01. Foundations & Interconnects](./chapters/01_foundations_interconnects.md) | Network hierarchy & collectives | Scale-up vs. scale-out bandwidth, Ring/Tree/NVLS All-Reduce algorithms |
| **02** | [02. Autograd & Memory Layouts](./chapters/02_foundations_autograd_and_memory.md) | Distributed autograd & process groups | Multi-dimensional process groups, autograd conjugate hooks ($f$/$g$), caching allocators |
| **03** | [03. Foundations Implementation Lab](./chapters/03_foundations_implementation_lab.md) | Verification scripts & MFU math | Multi-GPU verification scripts, NCCL selection heuristics, MFU/HFU arithmetic |
| **04** | [04. 1D Tensor Parallelism GEMMs](./chapters/04_tensor_parallelism_gemm_factoring.md) | Intra-node GEMM partitioning | Complementary Column/Row parallel linear pairing, non-linear activation trapping |
| **05** | [05. Vocab Parallelism & Tensor Cores](./chapters/05_tensor_parallelism_vocab_and_alignment.md) | Embedding & LM head scaling | `VocabParallelEmbedding`, Parallel Cross-Entropy, GQA/MQA sharding, tile alignment |
| **06** | [06. Tensor Parallelism Implementation](./chapters/06_tensor_parallelism_implementation_lab.md) | Production PyTorch TP block | Standalone PyTorch TP layers, $N \times b$ bias gotchas, production checklist |
| **07** | [07. Sequence Parallelism Sharding](./chapters/07_sequence_parallelism_activation_sharding.md) | Sequence-dimension sharding | $\text{RS} + \text{AG} \equiv \text{AllReduce}$ identity, sharded LayerNorm/Dropout |
| **08** | [08. Sequence Parallelism Lab](./chapters/08_sequence_parallelism_implementation_lab.md) | Memory economics & verification | Selective activation recomputation vs checkpointing, autograd mappings |
| **09** | [09. Pipeline Parallelism 1F1B](./chapters/09_pipeline_parallelism_1f1b_schedules.md) | Inter-node model pipelining | Vertical layer partitioning, GPipe baseline, 1F1B steady-state, virtual stages ($v$) |
| **10** | [10. P2P Communication Architecture](./chapters/10_pipeline_parallelism_p2p_communication.md) | Non-blocking transfers | Batched `isend`/`irecv`, distributed deadlock avoidance, CUDA stream synchronization |
| **11** | [11. Pipeline Parallelism Lab](./chapters/11_pipeline_parallelism_implementation_lab.md) | 2-Stage pipeline execution | 2-stage P2P pipeline implementation, bubble math, failure diagnostics |
| **12** | [12. Distributed Optimizer ZeRO Math](./chapters/12_distributed_optimizer_zero_math.md) | Static memory sharding | 16 bytes/param law, IEEE 754 mantissa swamping theorem, ZeRO-1/2/3 hierarchy |
| **13** | [13. Memory Buffers & Overlap](./chapters/13_distributed_optimizer_buffers_and_overlap.md) | Memory flattening & overlap | `ParamAndGradBuffer` layout, async Reduce-Scatter overlap, 70B/175B memory math |
| **14** | [14. Distributed Optimizer Lab](./chapters/14_distributed_optimizer_implementation_lab.md) | Native ZeRO-2 implementation | Megatron `DistributedOptimizer` from scratch, gradient clipping, swamping proofs |
| **15** | [15. Ring Attention & Online Softmax](./chapters/15_context_parallelism_ring_attention.md) | Million-token context scaling | Quadratic memory explosion, Online Softmax mathematical recurrence, Ring P2P |
| **16** | [16. Mixture of Experts (MoE)](./chapters/16_context_parallelism_mixture_of_experts.md) | Sparse model architectures | Switch Transformers, Top-K routing, capacity factor, All-to-All token dispatch |
| **17** | [17. Context Parallelism & MoE Lab](./chapters/17_context_parallelism_implementation_lab.md) | Runnable CP & MoE verification | Ring Attention with online softmax, MoE router lab, failure mode checklists |
| **18** | [18. M-Core Declarative ModuleSpec](./chapters/18_production_engine_modulespec.md) | Production engine architecture | Monolithic Megatron vs. M-Core, `TransformerConfig`, and `ModuleSpec` component trees |
| **19** | [19. FP8 Precision & State Checkpoints](./chapters/19_production_engine_fp8_and_checkpointing.md) | Low-precision & checkpointing | FP8 Delayed Scaling (`E4M3`/`E5M2`), Transformer Engine amax buffers, sharded state dicts |
| **20** | [20. Dynamic-CP & FSDP2 Extensions](./chapters/20_production_engine_dynamic_cp_and_fsdp2.md) | Cutting-edge M-Core additions | Dynamic Context Parallelism solver, Megatron-FSDP2 module sharding, cluster gotchas |
| **21** | [21. Token Ingestion Pipelines](./chapters/21_operations_token_data_pipeline.md) | High-throughput data loaders | Binary indexed datasets (`.bin`/`.idx`), `MMapIndexedDataset`, unpadded sequence packing |
| **22** | [22. Cluster Resilience & SRE](./chapters/22_operations_cluster_resilience_sre.md) | Production cluster fault tolerance | Hardware MTBF, elastic rendezvous (`c10d`), NCCL watchdogs, async checkpointing |
| **23** | [23. Accelerator Silicon Matrix](./chapters/23_operations_accelerator_silicon_matrix.md) | Hardware specs & framework trade-offs | H100 vs. Blackwell GB200 NVL72, InfiniBand fabrics, Megatron vs. DeepSpeed vs. FSDP2 |

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
