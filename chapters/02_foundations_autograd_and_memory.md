# Autograd Mechanics & Memory Layouts
> **Custom Distributed Autograd, Tensor Storage, and Multi-Dimensional Grid Topology**

---

## 2.1. Process Groups and Multi-Dimensional Grid Topology

In real LLM systems, we don't just have one flat communication ring. We use **3D Parallelism**:
- Some ranks communicate for Tensor Parallelism (TP)
- Some ranks communicate for Pipeline Parallelism (PP)
- Some ranks communicate for Data Parallelism (DP)

### 2.1.1 PyTorch Process Groups
A `ProcessGroup` in PyTorch defines a subset of ranks that can perform collective operations together.

For example, consider an 8-GPU setup with `TP = 2`, `DP = 4`:
```
Global Ranks: [0, 1, 2, 3, 4, 5, 6, 7]

TP Groups (size 2):
  Group 0: [0, 1]  (Ranks 0 and 1 split weights of Model Replica 0)
  Group 1: [2, 3]  (Ranks 2 and 3 split weights of Model Replica 1)
  Group 2: [4, 5]  (Ranks 4 and 5 split weights of Model Replica 2)
  Group 3: [6, 7]  (Ranks 6 and 7 split weights of Model Replica 3)

DP Groups (size 4):
  Group 0: [0, 2, 4, 6]  (Ranks holding the same TP partition 0 of different replicas)
  Group 1: [1, 3, 5, 7]  (Ranks holding the same TP partition 1 of different replicas)
```

When TP performs an All-Reduce, communication happens **only within the 2-rank TP group** over NVLink!
When DP synchronizes gradients, communication happens **only within the 4-rank DP group**!

---

## 2.2. PyTorch Custom Autograd Mechanics for Distributed Computing

In Megatron, communication is intimately tied to PyTorch's automatic differentiation graph (`torch.autograd`).

### 2.2.1 How `torch.autograd.Function` Works
A standard `nn.Module` forward pass records operations dynamically in a directed acyclic graph (DAG) of `Node` objects.
When defining custom distributed operations, we inherit from `torch.autograd.Function`:
```python
class DistributedOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, ...):
        # 1. Save any tensors needed for backward via ctx.save_for_backward(...)
        # 2. Perform forward computation and/or forward collective communication
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve saved tensors via ctx.saved_tensors
        # 2. Perform backward gradient computation and/or backward collective communication
        return grad_input, ...
```

### 2.2.2 The Conjugate Inversion Principle
Notice what happens during backpropagation:
- If a forward operation is an **identity** (data passed to N ranks without modification), the gradient accumulated at each rank must be **summed** across all N ranks:

> `Forward: Identity => Backward: All-Reduce (Sum)`

- If a forward operation is a **sum** across ranks (combining partial computations), each rank receives the identical upstream gradient during backprop:

> `Forward: All-Reduce (Sum) => Backward: Identity`

This is the exact mathematical foundation of Megatron's f and g operators!

---

---

## 2.3. Memory Mechanics: Tensors, Storage, and the Caching Allocator

Distributed collective operations directly touch memory buffers. Writing high-performance distributed code requires a mastery of how PyTorch and the GPU manage memory.

### 2.3.1 Tensor Metadata vs Underlying Storage Buffer
A PyTorch `Tensor` is not a block of memory. It is merely a **lightweight view (metadata)** pointing to an underlying `Storage` object:
- `tensor.data_ptr()`: The raw 64-bit virtual memory address of the first element in HBM/RAM.
- `tensor.shape`: The conceptual dimensions of the tensor (e.g., `[2, 3]`).
- `tensor.stride()`: The number of elements in physical memory you must skip to move by 1 step along each dimension.

```
Underlying Storage (Physical memory array):
Memory Address:  0x1000  0x1004  0x1008  0x100C  0x1010  0x1014
Values:         [  1.0,   2.0,    3.0,    4.0,    5.0,    6.0  ]

View A: x = torch.tensor([[1, 2, 3], [4, 5, 6]])
  Shape:  (2, 3)
  Stride: (3, 1)  -> Moving across rows skips 3 elements; moving across cols skips 1 element.
  Contiguous: YES (elements are consecutive in physical storage)

View B: y = x.t()  (Transpose)
  Shape:  (3, 2)
  Stride: (1, 3)  -> Moving across rows skips 1 element; moving across cols skips 3 elements!
  Contiguous: NO!
```

### 2.3.2 The Silent Collective Communication Trap:
Why does this distinction matter critically in distributed deep learning?
Because low-level collective libraries (**NCCL** and **Gloo**) are written in C/C++. They take a raw memory pointer `tensor.data_ptr()` and an element count N, and transmit raw consecutive bytes directly over PCIe/NVLink!

If you pass a **non-contiguous** tensor (like `y = x.t()`) into `dist.all_reduce(y)`:
1. NCCL reads N contiguous memory cells starting from `data_ptr()`.
2. It will read elements belonging to unrelated rows or even out-of-bounds memory!
3. PyTorch prevents this by either:
   - Crashing with `RuntimeError: Tensor must be contiguous`.
   - Or **silently allocating a temporary copy** via `tensor.contiguous()`, causing hidden GPU allocation stalls, memory spikes, and performance degradation!

> [!CAUTION]
> **Production Rule**: Every tensor passed to `dist.all_reduce()`, `dist.all_gather()`, or `dist.reduce_scatter()` must satisfy `tensor.is_contiguous() == True`. In Megatron, all GEMM activations and gradient buffers are kept strictly contiguous.

---

### 2.3.3 The PyTorch Caching Allocator & Memory Fragmentation
In standard C/C++, you allocate memory with `malloc()`, and in CUDA with `cudaMalloc()`.
However, `cudaMalloc()` is a **synchronous operating system kernel call**:
- It requires CPU-GPU driver synchronization.
- It takes 10 - 50 us per call.
- If a model allocated memory via `cudaMalloc` on every forward and backward pass, the training step time would increase by `3 *`!

To prevent this, PyTorch includes a custom **Caching Allocator**:
1. When your code first needs memory, PyTorch calls `cudaMalloc` to allocate a **large block (e.g., 20 MB to several GBs)** from the GPU driver.
2. It subdivides this block internally into two memory pools:
   - **Small allocation pool** (tensors `< 1 MB`)
   - **Large allocation pool** (tensors `>= 1 MB`)
3. When a tensor is deleted in Python (`del tensor`), PyTorch does **NOT** return the memory to the GPU driver! Instead, it retains the memory in its pool cache so that the next tensor allocation is instantaneous (`~ 0.1 us`, pure CPU pointer math).

```
                      The Danger of Memory Fragmentation
                      
  Physical VRAM Block: [  Tensor A (100MB)  ][  Tensor B (50MB)  ][  Tensor C (100MB)  ]
  
  Delete Tensor B:     [  Tensor A (100MB)  ][   FREE (50MB)    ][  Tensor C (100MB)  ]
  
  Now allocate Tensor D of size 80MB:
  Total free memory: 50MB + 500MB at end = 550MB free.
  Can Tensor D fit into the 50MB gap? NO!
  
  Result: PyTorch must allocate a NEW 80MB block at the end.
  The 50MB gap becomes "fragmented memory" — unusable for large tensors!
```

### 2.3.4 How Megatron Solves Fragmentation:
In long-running production training runs (spanning weeks or months), fragmentation can cause an **Out-Of-Memory (OOM) crash** even when 30 GB of VRAM appears free in monitoring dashboards!
Megatron Core prevents this by:
1. **Static Buffer Pre-Allocation**: Large communication buffers (for gradient reduction and sequence parallel gathering) are allocated once at initialization and reused forever using `torch.empty(..., out=static_buffer)`.
2. **PyTorch Allocator Configuration**: Setting the environment variable:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
   ```
   This allows PyTorch to dynamically expand existing virtual memory allocations without needing physical contiguous address space, eliminating fragmentation OOMs!

---

