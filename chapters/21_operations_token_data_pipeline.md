# High-Throughput Token Pipelines & Sequence Packing
> **Memory-Mapped Datasets (.bin/.idx), Blended Samplers, and Unpadded Variable-Length Batching**

> **Reference**: *NVIDIA Megatron Core Dataset Architecture & DeepSpeed Data Preprocessing Standards (2023–2025)*

---

## 1. The Data Ingestion Bottleneck at Frontier Scale

In distributed pretraining clusters scaling from hundreds to thousands of GPUs, **the data pipeline is often an unexpected and catastrophic throughput bottleneck**:
1. **Host Memory Thrashing**: Loading multi-terabyte corpora of raw text or JSONL files into Python memory causes immediate Linux Out-Of-Memory (OOM) kills or extensive swapping.
2. **Network Storage Storms**: If `16,384` GPUs simultaneously issue synchronous read requests to raw text files on a shared network file system (NFS, Lustre, GPFS), metadata servers collapse under millions of IOPS.
3. **Non-Deterministic Multi-Epoch Shuffling**: In standard PyTorch, shuffling a multi-terabyte dataset requires materializing index permutations that exceed host RAM and cannot be deterministically resumed after a cluster crash.
4. **Padding Inefficiency**: Batches padded to `--seq-length` with pad tokens waste `20% - 40%` of GPU Tensor Core FLOPs on calculating self-attention over meaningless zeros.

Megatron Core solves these challenges through **`megatron.core.datasets`**, built upon the high-performance **Binary Indexed Dataset format (`MMapIndexedDataset`)**, **Blended Weighted Datasets**, and **Sequence Packing**.

---

## 2. The Binary Indexed Dataset Format (`.bin` and `.idx`)

Instead of parsing raw JSONL or CSV during training, text is tokenized offline and converted into two paired binary files:
- **`data.bin`**: A dense, contiguous array of raw token IDs (typically `uint16` for vocabularies `<= 65,536`, or `int32` for larger vocabs up to 1M tokens).
- **`data.idx`**: A compact metadata index recording document boundaries and byte offsets.

```
                    MMapIndexedDataset File Layout
                    
data.idx Header:
[ Magic Bytes (8B) | Version (8B) | Dtype Code (1B) | Num Docs (8B) | Num Tokens (8B) ]
Followed by:
[ Doc Pointers: offset_0, offset_1, offset_2, ... ] (int64 array)
[ Doc Lengths:  len_0,    len_1,    len_2, ... ]    (int32 array)

data.bin (Raw Memory-Mapped Token Stream):
[ Token_0, Token_1, Token_2, ..., Token_N ] (Contiguous uint16 / int32)
  │                                   │
  └──────── Document 0 ───────────────┘└─────── Document 1 ───────...
```

### 2.1 Constant-Time `O(1)` Zero-Copy Memory Mapping (`mmap`)
During training:
- Processes never load the whole `.bin` file into memory. Instead, they call POSIX `mmap()` to map the file into the virtual address space.
- The Linux kernel page cache transparently pages in only the 4 KB / 2 MB memory pages corresponding to the active microbatch.
- Millions of random access samples are fetched in **constant `O(1)` time** with zero memory allocation overhead!

---

## 3. Deterministic Sample Indexing & Epoch Shuffling

To guarantee exact reproducibility across node crashes and distributed resumptions, Megatron Core constructs a deterministic **Sample Index Mapping**:
Given:
- A sequence length S (e.g. `4,096`).
- An epoch count E.
- Document lengths `\{L_0, L_1, ..., L_D\}`.

Megatron pre-computes an array of sample coordinates:
```text
Sample_i = (doc\_idx_i, start\_offset_i)
```
When a document finishes mid-sequence, Megatron automatically concatenates the beginning of the next document, separated by an `<|endoftext|>` token, ensuring that **every single sequence fed to the model is exactly S tokens long with zero padding**.

---

## 4. Blended & Weighted Datasets

Frontier LLMs are trained on mixtures of diverse data sources (e.g., 50% Web Crawl, 20% Code, 15% Academic/ArXiv, 10% Books, 5% Math).

Naive mixing requires rewriting and copying petabytes of tokens into a single merged file. 
Megatron Core's **`BlendedDataset`** combines datasets virtually at the index level:
- You specify a list of dataset prefixes and target sampling probabilities:
```python
weights = [0.50, 0.20, 0.15, 0.10, 0.05]
```
- Megatron builds a single virtual index mapping where each global sample index i deterministically maps to a sub-dataset index based on a pseudo-random permutation weighted by the specified mixture ratios.
- **Zero data duplication**: Source `.bin` and `.idx` files remain completely untouched on disk!

---

## 5. Sequence Packing (Packing Without Padding)

In instruction fine-tuning (SFT) or multi-turn conversational data, sample lengths vary wildly (e.g., from 50 tokens to 4,000 tokens). Traditional batch padding pads every sequence to the max length, wasting up to 40% of GPU compute on pad tokens.

**Sequence Packing** (often called *Packing* or *Multi-pack*) packs multiple short sequences into a single continuous sequence of length S:

```
Padded Batch (Wasteful):
Sample 0: [ Token A, Token B, <PAD>, <PAD>, <PAD>, <PAD> ] (66% compute wasted!)
Sample 1: [ Token C, Token D, Token E, Token F, <PAD>, <PAD> ] (33% compute wasted!)

Packed Sequence (100% Compute Efficiency):
Packed:   [ Token A, Token B, <EOS>, Token C, Token D, Token E, Token F, <EOS> ]
Cumulative Seqlens (cu_seqlens): [ 0, 3, 8 ]
```

### 5.1 FlashAttention with `cu_seqlens`
To prevent tokens from Sample 0 attending to tokens in Sample 1 within the same packed window, Megatron Core passes an array of cumulative sequence lengths (`cu_seqlens`) directly to FlashAttention / Transformer Engine:
```text
cu\_seqlens = [0, S_1, S_1 + S_2, ..., S]
```
The FlashAttention CUDA kernel uses `cu_seqlens` to reset its softmax accumulators at document boundaries, completely preventing cross-document contamination while achieving **100% arithmetic throughput**!

---

## 6. Megatron-Core Data Pipeline API Reference

The production data pipeline classes live in `megatron/core/datasets/`:

```python
# megatron/core/datasets/gpt_dataset.py
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig

config = GPTDatasetConfig(
    is_built_on_rank=lambda: True,
    random_seed=42,
    sequence_length=4096,
    blend=[("/data/pile/train.bin", 0.7), ("/data/books/train.bin", 0.3)],
    split="949,50,1",                   # train/val/test split ratios
    tokenizer=tokenizer,
    reset_position_ids=True,            # Pack with position ID reset
    reset_attention_mask=True,          # Enforce cu_seqlens document boundaries
    eod_mask_loss=True,                 # Mask loss on EOD tokens
)

# Dataset is built once offline; memory-mapped at runtime
train_dataset = GPTDataset(
    file_prefix="/data/pile/train",     # Path to .bin/.idx pair
    documents=np.arange(0, num_train_docs),
    config=config,
)
```

The `GPTDataset.__getitem__` returns a packed dict:
```python
{
    "tokens":           torch.Tensor[S],        # Input token IDs
    "labels":           torch.Tensor[S],        # Shifted target token IDs
    "attention_mask":   torch.Tensor[1],        # Scalar: 1 if valid sample
    "loss_mask":        torch.Tensor[S],        # 0 on EOD/padding positions
    "position_ids":     torch.Tensor[S],        # Packed position offsets
}
```

For sequence packing with `cu_seqlens`, Megatron's collator calls:
```python
from megatron.core.packed_seq_params import PackedSeqParams

packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cu_seqlens,           # [num_docs+1] cumulative sequence offsets
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    qkv_format='thd',                  # (total_tokens, heads, head_dim)
)
```

This is passed directly to `flash_attn_varlen_func`, which uses `cu_seqlens` to reset softmax accumulators at document boundaries, preventing any cross-document attention contamination.




