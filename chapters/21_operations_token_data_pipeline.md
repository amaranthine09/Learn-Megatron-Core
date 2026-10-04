# High-Throughput Token Pipelines & Sequence Packing
> **Memory-Mapped Datasets (.bin/.idx), Blended Samplers, and Unpadded Variable-Length Batching**

> **Reference**: *NVIDIA Megatron Core Dataset Architecture & DeepSpeed Data Preprocessing Standards (2023–2025)*

---

## 1. The Data Ingestion Bottleneck at Frontier Scale

In distributed pretraining clusters scaling from hundreds to thousands of GPUs, **the data pipeline is often an unexpected and catastrophic throughput bottleneck**:
1. **Host Memory Thrashing**: Loading multi-terabyte corpora of raw text or JSONL files into Python memory causes immediate Linux Out-Of-Memory (OOM) kills or extensive swapping.
2. **Network Storage Storms**: If $16{,}384$ GPUs simultaneously issue synchronous read requests to raw text files on a shared network file system (NFS, Lustre, GPFS), metadata servers collapse under millions of IOPS.
3. **Non-Deterministic Multi-Epoch Shuffling**: In standard PyTorch, shuffling a multi-terabyte dataset requires materializing index permutations that exceed host RAM and cannot be deterministically resumed after a cluster crash.
4. **Padding Inefficiency**: Batches padded to `--seq-length` with pad tokens waste $20\% - 40\%$ of GPU Tensor Core FLOPs on calculating self-attention over meaningless zeros.

Megatron Core solves these challenges through **`megatron.core.datasets`**, built upon the high-performance **Binary Indexed Dataset format (`MMapIndexedDataset`)**, **Blended Weighted Datasets**, and **Sequence Packing**.

---

## 2. The Binary Indexed Dataset Format (`.bin` and `.idx`)

Instead of parsing raw JSONL or CSV during training, text is tokenized offline and converted into two paired binary files:
- **`data.bin`**: A dense, contiguous array of raw token IDs (typically `uint16` for vocabularies $\le 65{,}536$, or `int32` for larger vocabs up to $1\text{M}$ tokens).
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

### 2.1 Constant-Time $O(1)$ Zero-Copy Memory Mapping (`mmap`)
During training:
- Processes never load the whole `.bin` file into memory. Instead, they call POSIX `mmap()` to map the file into the virtual address space.
- The Linux kernel page cache transparently pages in only the 4 KB / 2 MB memory pages corresponding to the active microbatch.
- Millions of random access samples are fetched in **constant $O(1)$ time** with zero memory allocation overhead!

---

## 3. Deterministic Sample Indexing & Epoch Shuffling

To guarantee exact reproducibility across node crashes and distributed resumptions, Megatron Core constructs a deterministic **Sample Index Mapping**:
Given:
- A sequence length $S$ (e.g. $4{,}096$).
- An epoch count $E$.
- Document lengths $\{L_0, L_1, \dots, L_D\}$.

Megatron pre-computes an array of sample coordinates:
$$\text{Sample}_i = (\text{doc\_idx}_i, \text{start\_offset}_i)$$
When a document finishes mid-sequence, Megatron automatically concatenates the beginning of the next document, separated by an `<|endoftext|>` token, ensuring that **every single sequence fed to the model is exactly $S$ tokens long with zero padding**.

---

## 4. Blended & Weighted Datasets

Frontier LLMs are trained on mixtures of diverse data sources (e.g., $50\%$ Web Crawl, $20\%$ Code, $15\%$ Academic/ArXiv, $10\%$ Books, $5\%$ Math).

Naive mixing requires rewriting and copying petabytes of tokens into a single merged file. 
Megatron Core's **`BlendedDataset`** combines datasets virtually at the index level:
- You specify a list of dataset prefixes and target sampling probabilities:
```python
weights = [0.50, 0.20, 0.15, 0.10, 0.05]
```
- Megatron builds a single virtual index mapping where each global sample index $i$ deterministically maps to a sub-dataset index based on a pseudo-random permutation weighted by the specified mixture ratios.
- **Zero data duplication**: Source `.bin` and `.idx` files remain completely untouched on disk!

---

## 5. Sequence Packing (Packing Without Padding)

In instruction fine-tuning (SFT) or multi-turn conversational data, sample lengths vary wildly (e.g., from 50 tokens to 4,000 tokens). Traditional batch padding pads every sequence to the max length, wasting up to $40\%$ of GPU compute on pad tokens.

**Sequence Packing** (often called *Packing* or *Multi-pack*) packs multiple short sequences into a single continuous sequence of length $S$:

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
$$\text{cu\_seqlens} = [0, S_1, S_1 + S_2, \dots, S]$$
The FlashAttention CUDA kernel uses `cu_seqlens` to reset its softmax accumulators at document boundaries, completely preventing cross-document contamination while achieving **100% arithmetic throughput**!

---

## 6. Runnable Reference Implementation: Scaled Data Pipeline

Below is a self-contained, runnable Python implementation demonstrating:
1. Binary indexed dataset creation and memory-mapped reader (`MMapIndexedDataset`).
2. Virtual Blended Dataset with deterministic weighted sampling.
3. Sequence Packing with `cu_seqlens` calculation.

```python
import os
import struct
import math
import numpy as np
import torch
from typing import List, Tuple

class MMapIndexedDatasetBuilder:
    """
    Builds paired binary indexed dataset files (.bin and .idx) offline.
    """
    def __init__(self, bin_path: str, idx_path: str, dtype=np.uint16):
        self.bin_file = open(bin_path, 'wb')
        self.idx_file = open(idx_path, 'wb')
        self.dtype = dtype
        self.doc_offsets = [0]
        self.doc_lengths = []
        self.total_tokens = 0

    def add_document(self, token_ids: List[int]):
        arr = np.array(token_ids, dtype=self.dtype)
        self.bin_file.write(arr.tobytes())
        numel = len(token_ids)
        self.total_tokens += numel
        self.doc_lengths.append(numel)
        self.doc_offsets.append(self.total_tokens)

    def finalize(self):
        self.bin_file.close()
        # Write index header: magic (8B), version (8B), num_docs (8B), num_tokens (8B)
        header = struct.pack('<QQQQ', 0x4D45474154524F4E, 1, len(self.doc_lengths), self.total_tokens)
        self.idx_file.write(header)
        # Write doc offsets (int64) and doc lengths (int32)
        np.array(self.doc_offsets[:-1], dtype=np.int64).tofile(self.idx_file)
        np.array(self.doc_lengths, dtype=np.int32).tofile(self.idx_file)
        self.idx_file.close()


class MMapIndexedDataset:
    """
    Memory-mapped constant-time reader for large token corpora.
    Zero-copy read via OS mmap.
    """
    def __init__(self, bin_path: str, idx_path: str, dtype=np.uint16):
        self.bin_path = bin_path
        self.idx_path = idx_path
        self.dtype = dtype

        with open(idx_path, 'rb') as f:
            header = f.read(32)
            magic, version, self.num_docs, self.total_tokens = struct.unpack('<QQQQ', header)
            self.doc_offsets = np.fromfile(f, dtype=np.int64, count=self.num_docs)
            self.doc_lengths = np.fromfile(f, dtype=np.int32, count=self.num_docs)

        # Memory map the binary file (read-only, shared across workers)
        self.bin_mmap = np.memmap(bin_path, dtype=self.dtype, mode='r')

    def get_document(self, doc_idx: int) -> np.ndarray:
        offset = self.doc_offsets[doc_idx]
        length = self.doc_lengths[doc_idx]
        return self.bin_mmap[offset : offset + length]

    def __len__(self):
        return self.num_docs


class VirtualBlendedDataset:
    """
    Virtually blends multiple MMapIndexedDatasets according to sampling weights
    without copying or rewriting underlying binary files.
    """
    def __init__(self, datasets: List[MMapIndexedDataset], weights: List[float], total_samples: int = 1000):
        self.datasets = datasets
        norm_weights = np.array(weights) / sum(weights)
        self.weights = norm_weights
        self.total_samples = total_samples
        
        # Build deterministic global index mapping
        rng = np.random.RandomState(42)
        self.dataset_assignments = rng.choice(len(datasets), size=total_samples, p=norm_weights)
        self.sample_indices = [
            rng.randint(0, len(datasets[ds_idx])) for ds_idx in self.dataset_assignments
        ]

    def __getitem__(self, idx: int) -> np.ndarray:
        ds_idx = self.dataset_assignments[idx]
        sample_idx = self.sample_indices[idx]
        return self.datasets[ds_idx].get_document(sample_idx)

    def __len__(self):
        return self.total_samples


class SequencePacker:
    """
    Packs variable-length sequences into fixed windows of max_seq_len
    and generates cu_seqlens for unpadded FlashAttention execution.
    """
    def __init__(self, max_seq_len: int, eos_token_id: int = 2):
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id

    def pack(self, documents: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Packs documents into [max_seq_len] buffer and returns (tokens, cu_seqlens).
        """
        packed_tokens = []
        cu_seqlens = [0]

        for doc in documents:
            doc_with_eos = list(doc) + [self.eos_token_id]
            if len(packed_tokens) + len(doc_with_eos) <= self.max_seq_len:
                packed_tokens.extend(doc_with_eos)
                cu_seqlens.append(len(packed_tokens))
            else:
                # Truncate or spill to next packed chunk
                remaining = self.max_seq_len - len(packed_tokens)
                if remaining > 0:
                    packed_tokens.extend(doc_with_eos[:remaining])
                    cu_seqlens.append(len(packed_tokens))
                break

        tokens_tensor = torch.tensor(packed_tokens, dtype=torch.long)
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)
        return tokens_tensor, cu_seqlens_tensor


if __name__ == '__main__':
    print("=" * 65)
    print("  MEGATRON SCALED DATA PIPELINE VERIFICATION")
    print("=" * 65)

    bin_path = "/tmp/megatron_demo.bin"
    idx_path = "/tmp/megatron_demo.idx"

    # 1. Offline Indexing
    builder = MMapIndexedDatasetBuilder(bin_path, idx_path)
    sample_docs = [
        [101, 2054, 2003, 1037, 3231, 102],
        [101, 7592, 1010, 2026, 2171, 102],
        [101, 2129, 2024, 2115, 1029, 102],
    ]
    for doc in sample_docs:
        builder.add_document(doc)
    builder.finalize()

    # 2. Zero-Copy Memory-Mapped Reading
    ds = MMapIndexedDataset(bin_path, idx_path)
    print(f"  Indexed Corpus Documents:       {len(ds)}")
    print(f"  Total Indexed Tokens:          {ds.total_tokens}")
    sample_read = ds.get_document(1)
    print(f"  Document 1 Direct MMAP Slice:   {sample_read.tolist()}")

    # 3. Sequence Packing
    packer = SequencePacker(max_seq_len=16, eos_token_id=0)
    packed_toks, cu_lens = packer.pack(sample_docs)
    print(f"  Packed Tokens Length:          {len(packed_toks)}")
    print(f"  FlashAttention cu_seqlens:     {cu_lens.tolist()}")
    print("=" * 65)

    # Cleanup temporary demo files
    if os.path.exists(bin_path): os.remove(bin_path)
    if os.path.exists(idx_path): os.remove(idx_path)
```
