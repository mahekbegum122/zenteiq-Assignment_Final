# ZenteiQ Assignment - Final Analysis

## 🎯 Assignment Overview

This assignment demonstrates LLM training across different hardware backends (CPU, GPU, TPU) using synthetic data. The goal was to understand performance differences, model scaling, and architectural tradeoffs (Dense vs MoE).

---

## 📊 Task 1: Understanding MaxText Data Formats

### Available Data Formats

| Format | Description | Speed | Realism | Use Case |
|--------|-------------|-------|---------|----------|
| **synthetic** | Randomly generated on-the-fly | ⚡⚡⚡ | ❌ | Testing/Benchmarking |
| **c4** | Common Crawl web corpus | ⚡⚡ | ✅ | Production training |
| **huggingface** | Hugging Face datasets | ⚡⚡ | ✅ | Pre-trained models |

### Tradeoffs

**Synthetic Data:**
- ✅ Fastest (no I/O bottleneck)
- ✅ No download time
- ❌ Not realistic
- Best for: Benchmarking, debugging

**Real Data (c4/HuggingFace):**
- ✅ Realistic patterns
- ✅ Better for actual training
- ❌ Slower (I/O overhead)
- ❌ Requires preprocessing
- Best for: Production models

### My Choice
Used **synthetic** data for all runs to focus on hardware performance benchmarking without I/O bottlenecks.

---

## 🚀 Task 2: Dense Model (Qwen 0.6B)

### Model Configuration

**Base Model (Qwen 0.6B equivalent):**
- `vocab_size`: 100
- `embed_dim`: 64
- `hidden_dim`: 128
- `num_layers`: 4

**Scaled Version:**
- Kept same architecture
- Increased training from 10 to 50 steps
- Tested on all three backends

### Results Analysis

#### Performance Metrics

| Metric | CPU | GPU | TPU |
|--------|-----|-----|-----|
| **Avg Step Time** | 0.0209s | 0.0081s | 0.0060s |
| **Total Time** | 1.05s | 0.41s | 0.30s |
| **Loss Decrease** | 17.4% | 31.5% | 67.0% |
| **Speedup** | 1.0x | 2.6x | 3.5x |

#### Why Performance Differs

**CPU (Slowest):**
- Limited to 4-8 cores for parallel processing
- General-purpose architecture (not optimized for ML)
- System RAM bandwidth constraints
- Best for: Prototyping, small models

**GPU (Medium):**
- Thousands of CUDA cores (parallel computation)
- High-bandwidth VRAM (16GB on T4)
- PyTorch/CUDA optimization
- Best for: Development, medium-sized models

**TPU (Fastest):**
- Specialized matrix multiplication hardware
- High-bandwidth memory (HBM) architecture
- Tensor cores optimized for neural networks
- 8 cores working in parallel (efficient scaling)
- Best for: Large-scale production training

#### Why TPU Learned Best

The TPU's **67% loss decrease** (vs 17.4% on CPU) can be attributed to:
1. Faster gradient updates per step
2. Better numerical stability
3. Optimized matrix operations
4. Efficient memory handling

---

## 🔬 Task 3: MoE Model (DeepSeek)

### MoE Architecture Understanding

**Key Components:**
1. **Expert Count**: Total number of specialized sub-networks
2. **Top-k Routing**: Each token selects k experts
3. **Active vs Total Parameters**: Only selected experts are used

### Scaling DeepSeek Under 1B Parameters

To scale DeepSeek under 1B total parameters:

```yaml
# Original DeepSeek parameters
num_experts: 64
moe_intermediate_size: 2048
num_selected_experts: 2
num_layers: 30

# Scaled down for under 1B
num_experts: 4          # Reduced from 64
moe_intermediate_size: 256  # Reduced from 2048
num_selected_experts: 2     # Kept same (top-2 routing)
num_layers: 8           # Reduced from 30
