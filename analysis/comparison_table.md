# Performance Comparison: CPU vs GPU vs TPU

## 📊 Executive Summary

This document presents a comprehensive comparison of training performance across three hardware backends for a dense language model (Qwen 0.6B equivalent). All runs used **synthetic data** with **50 training steps**.

---

## 🏆 Overall Performance Ranking

---

## 📈 Detailed Comparison Table

| Metric | CPU (JAX) | GPU (PyTorch) | TPU (PyTorch/XLA) |
|--------|-----------|---------------|-------------------|
| **Hardware** | Intel/AMD CPU | NVIDIA Tesla T4 | Google TPU v2-8 |
| **Framework** | JAX 0.4.20 | PyTorch 2.2.0 | PyTorch 2.9.0 |
| **NumPy Version** | 1.26.4 | 1.26.4 | 2.0.2 |
| **Device Detected** | TFRT_CPU_0 | cuda:0 | xla:0 |
| **Cores/Processors** | 1 (virtual) | 1 (virtual) | 1 of 8 detected |
| **Memory Type** | System RAM | GDDR6 VRAM (16GB) | HBM (High Bandwidth) |
| **Avg Step Time (s)** | 0.0209 | 0.0081 | 0.0060 |
| **Total Training Time (s)** | 1.05 | 0.41 | 0.30 |
| **Initial Loss** | 4.6209 | 4.9787 | 4.7204 |
| **Final Loss** | 3.8165 | 3.4110 | 1.5594 |
| **Loss Decrease** | 0.8044 (17.4%) | 1.5677 (31.5%) | 3.1610 (67.0%) |
| **Speedup vs CPU** | 1.0x | 2.6x | 3.5x |
| **Steps Completed** | 50 | 50 | 50 |

---

## 📊 Performance Visualization

### Speed Comparison (Faster = Better)

### Step Time Comparison (Lower = Better)

### Loss Decrease (Higher = Better)

---

## 🏗️ Architecture Diagrams

### 1. CPU Architecture



**Key Characteristics:**
- ✅ General-purpose processing
- ✅ Good for sequential tasks
- ❌ Limited parallelism
- ❌ Not optimized for matrix operations

---

### 2. GPU Architecture


**Key Characteristics:**
- ✅ Thousands of parallel cores
- ✅ High-bandwidth VRAM
- ✅ Tensor cores for ML acceleration
- ✅ Excellent for matrix multiplication

---

### 3. TPU Architecture

**Key Characteristics:**
- ✅ Specialized matrix multiplication hardware
- ✅ High-bandwidth memory (HBM)
- ✅ Efficient parallel processing
- ✅ Optimized for neural networks

---

## 🔬 Detailed Interpretation

### Why Numbers Differ Across Backends

#### 1. CPU (Slowest)

#### 2. GPU (Fast)

#### 3. TPU (Fastest)

---

## 💡 Key Insights

### 1. TPU is 3.5x Faster Than CPU
- **Reason**: Specialized hardware for matrix multiplication
- **Impact**: 67% loss decrease vs 17.4% on CPU
- **Conclusion**: TPUs significantly improve training efficiency

### 2. GPU is 2.6x Faster Than CPU
- **Reason**: Thousands of CUDA cores and optimized libraries
- **Impact**: 31.5% loss decrease vs 17.4% on CPU
- **Conclusion**: GPUs are excellent for development

### 3. Framework Matters
- **CPU**: JAX (worked with compatibility issues)
- **GPU/TPU**: PyTorch (stable and optimized)
- **Conclusion**: Choose framework based on hardware

---

## 🎯 Recommendations

| Scenario | Recommended Hardware | Why |
|----------|---------------------|-----|
| **Prototyping** | CPU | Simplest setup, good for testing |
| **Development** | GPU | Fast, good debugging, accessible |
| **Production (<1B params)** | GPU | Cost-effective, good performance |
| **Production (>1B params)** | TPU | Best performance, cost-effective at scale |

---

## 📝 Conclusion

**Performance Hierarchy**: TPU > GPU > CPU

- **TPU**: Best for large-scale training
- **GPU**: Best for development and medium models
- **CPU**: Best for prototyping and debugging

---

*Data collected on Google Colab (Free Tier) - June 2026*
