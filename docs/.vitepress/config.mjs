import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Megatron Core',
  description: 'First-Principles Architecture Manual & Distributed Systems Curriculum',
  base: '/Learn-Megatron-Core/',
  ignoreDeadLinks: true,
  
  markdown: {
    math: true
  },

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Architecture Manual', link: '/chapters/01_foundations_distributed_torch' },
      { text: 'Hardware Matrix', link: '/chapters/appendix_c_hardware_superchips' },
      { text: 'GitHub', link: 'https://github.com/amaranthine09/Learn-Megatron-Core' }
    ],

    sidebar: [
      {
        text: 'Overview',
        items: [
          { text: 'Curriculum Index', link: '/' }
        ]
      },
      {
        text: 'Part I: Intra-Node Scaling (NVLink)',
        collapsed: false,
        items: [
          { text: '01. Foundations & Interconnects', link: '/chapters/01_foundations_distributed_torch' },
          { text: '02. 1D Tensor Parallelism', link: '/chapters/02_tensor_parallelism_math_and_layers' },
          { text: '03. Sequence Parallelism', link: '/chapters/03_sequence_parallelism_and_activations' }
        ]
      },
      {
        text: 'Part II: Scale-Out Fabric & Memory',
        collapsed: false,
        items: [
          { text: '04. Pipeline Parallelism & DualPipe', link: '/chapters/04_pipeline_parallelism_and_schedules' },
          { text: '05. Distributed Optimizer (ZeRO-2)', link: '/chapters/05_memory_accounting_and_distributed_optimizer' }
        ]
      },
      {
        text: 'Part III: Context Length & Sparsity',
        collapsed: false,
        items: [
          { text: '06. Context Parallelism & MoE', link: '/chapters/06_context_parallelism_and_moe' },
          { text: '07. Megatron Core Production Engine', link: '/chapters/07_megatron_core_production_architecture' }
        ]
      },
      {
        text: 'Part IV: Production Runtime Specifications',
        collapsed: false,
        items: [
          { text: 'Ref A: Token Pipelines & Packing', link: '/chapters/appendix_a_data_pipeline' },
          { text: 'Ref B: Cluster Resilience & Fault Tolerance', link: '/chapters/appendix_b_cluster_reliability' },
          { text: 'Ref C: Accelerator Silicon Matrix', link: '/chapters/appendix_c_hardware_superchips' }
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
