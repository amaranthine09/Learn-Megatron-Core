import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Megatron Core',
  description: 'First-Principles Architecture Manual & Distributed Systems Specification',
  base: '/Learn-Megatron-Core/',
  ignoreDeadLinks: true,
  
  markdown: {
    math: true
  },

  themeConfig: {
    nav: [
      { text: 'System Overview', link: '/' },
      { 
        text: 'Core Parallelisms', 
        items: [
          { text: '1D Tensor Parallelism', link: '/tensor-parallelism/' },
          { text: 'Sequence Parallelism', link: '/sequence-parallelism/' },
          { text: 'Pipeline Parallelism & DualPipe', link: '/pipeline-parallelism/' },
          { text: 'Context Parallelism & MoE', link: '/context-parallelism/' }
        ]
      },
      { 
        text: 'Memory & Engine', 
        items: [
          { text: 'Distributed Optimizer (ZeRO)', link: '/distributed-optimizer/' },
          { text: 'M-Core Production Engine', link: '/production-engine/' },
          { text: 'Token Ingestion Pipelines', link: '/operations/data-pipeline' }
        ]
      },
      { 
        text: 'Operations & Silicon', 
        items: [
          { text: 'Cluster Fault Tolerance', link: '/operations/cluster-resilience' },
          { text: 'Accelerator Silicon Matrix', link: '/operations/silicon-and-systems' }
        ]
      },
      { text: 'GitHub', link: 'https://github.com/amaranthine09/Learn-Megatron-Core' }
    ],

    sidebar: [
      {
        text: 'Overview & Foundations',
        collapsed: false,
        items: [
          { text: 'Architecture Master Map', link: '/' },
          { text: '1. Foundations & Interconnects', link: '/foundations/' },
          { text: '2. Autograd & Memory Layouts', link: '/foundations/autograd-and-memory' },
          { text: '3. Foundations Lab & Verification', link: '/foundations/implementation' }
        ]
      },
      {
        text: '1D Tensor Parallelism (TP)',
        collapsed: false,
        items: [
          { text: '1. Complementary GEMM Factoring', link: '/tensor-parallelism/' },
          { text: '2. Vocab Parallelism & Tensor Cores', link: '/tensor-parallelism/vocab-and-advanced' },
          { text: '3. Complete PyTorch Implementation', link: '/tensor-parallelism/implementation' }
        ]
      },
      {
        text: 'Sequence Parallelism (SP)',
        collapsed: false,
        items: [
          { text: '1. Zero-Overhead Activation Sharding', link: '/sequence-parallelism/' },
          { text: '2. Memory Economics & PyTorch Lab', link: '/sequence-parallelism/implementation' }
        ]
      },
      {
        text: 'Pipeline Parallelism (PP)',
        collapsed: false,
        items: [
          { text: '1. 1F1B & Interleaved Virtual Stages', link: '/pipeline-parallelism/' },
          { text: '2. P2P Communication & Deadlock Avoidance', link: '/pipeline-parallelism/dualpipe-and-p2p' },
          { text: '3. 2-Stage Pipeline Implementation', link: '/pipeline-parallelism/implementation' }
        ]
      },
      {
        text: 'Memory & Distributed Optimizer (ZeRO)',
        collapsed: false,
        items: [
          { text: '1. The 16-Bytes Law & Swamping Proof', link: '/distributed-optimizer/' },
          { text: '2. Buffer Layouts & Overlap Math', link: '/distributed-optimizer/buffers-and-overlap' },
          { text: '3. ZeRO-2 Optimizer Implementation', link: '/distributed-optimizer/implementation' }
        ]
      },
      {
        text: 'Context Parallelism (CP) & MoE',
        collapsed: false,
        items: [
          { text: '1. Ring Attention & Online Softmax', link: '/context-parallelism/' },
          { text: '2. Mixture of Experts & EP Dispatch', link: '/context-parallelism/mixture-of-experts' },
          { text: '3. CP & MoE Runnable Verification', link: '/context-parallelism/implementation' }
        ]
      },
      {
        text: 'Production Pretraining Engine',
        collapsed: false,
        items: [
          { text: '1. M-Core Declarative ModuleSpec', link: '/production-engine/' },
          { text: '2. FP8 Scaling & State Checkpointing', link: '/production-engine/fp8-and-checkpointing' },
          { text: '3. Dynamic-CP & FSDP2 Extensions', link: '/production-engine/modern-innovations' }
        ]
      },
      {
        text: 'Cluster Infrastructure & Operations',
        collapsed: false,
        items: [
          { text: 'Token Pipelines & Sequence Packing', link: '/operations/data-pipeline' },
          { text: 'Cluster Resilience & SRE Diagnostics', link: '/operations/cluster-resilience' },
          { text: 'Accelerator Silicon & Systems Matrix', link: '/operations/silicon-and-systems' }
        ]
      }
    ],

    search: {
      provider: 'local'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/amaranthine09/Learn-Megatron-Core' }
    ],

    footer: {
      message: 'Released under the Apache 2.0 License.',
      copyright: 'Copyright © 2026 Ankush Thakur'
    }
  }
})
