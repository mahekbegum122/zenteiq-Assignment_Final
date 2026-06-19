# 🚀 ZenteiQ AiTech Innovations - ML Engineer Assignment

## 📋 Overview

This repository contains my completed ML Engineer assignment for ZenteiQ AiTech Innovations. The project demonstrates LLM training across different hardware backends (CPU, GPU, TPU) using synthetic data, comparing performance, scalability, and architectural tradeoffs between dense and MoE models.

| **Author** | Mahek Begum |
|------------|-------------|
| **Date** | June 2026 |
| **Position** | ML Engineer |
| **GitHub** | [mahekbegum122/zenteiq-Assignment_Final](https://github.com/mahekbegum122/zenteiq-Assignment_Final) |

---

## 📁 Repository Structure
zenteiq-Assignment_Final/
│.
├── 📁 analysis/
.
│ ├── comparison_table.md
.
│ ├── figure4_comparison_table.png.jpg
.
│ └── final_analysis.md
.
│
├── 📁 code/
│ ├── ZenteiQ_CPU_Assignment.ipynb
│ ├── ZenteiQ_GPU_Assignment.ipynb
│ └── ZenteiQ_tpu_Assignment.ipynb
│
├── 📁 logs/
│ ├── cpu_metrics.txt
│ ├── metrics_gpu_pytorch.txt
│ └── metrics_tpu_pytorch.txt
│
├── 📁 screenshots/
│ ├── CPU_Runtime_Setting.jpg
│ ├── GPU_Runtime_Setting.jpg
│ ├── TPU_Runtime_Setting.jpg
│ ├── figure1_cpu_training.png.jpg
│ ├── figure2_gpu_training.png.jpg
│ ├── figure3_tpu_training.png.jpg
│ └── figure4_comparison_table.png.jpg
│
└── 📄 README.md

---

## 📊 Key Results Summary

| Metric | CPU (JAX) | GPU (PyTorch) | TPU (PyTorch/XLA) |
|--------|-----------|---------------|-------------------|
| **Device** | TFRT_CPU_0 | cuda (Tesla T4) | xla:0 (TPU) |
| **Framework** | JAX 0.4.20 | PyTorch 2.2.0 | PyTorch 2.9.0 |
| **NumPy Version** | 1.26.4 | 1.26.4 | 2.0.2 |
| **Avg Step Time** | 0.0209s | 0.0081s | 0.0060s |
| **Total Training Time** | 1.05s | 0.41s | 0.30s |
| **Initial Loss** | 4.6209 | 4.9787 | 4.7204 |
| **Final Loss** | 3.8165 | 3.4110 | 1.5594 |
| **Loss Decrease** | 17.4% | 31.5% | 67.0% |
| **Speedup vs CPU** | 1.0x | 2.6x | 3.5x |

### Performance Visualization
Speed Comparison (Faster = Better)
========================================
CPU: ████████████████████░░░░░░░░░░ 1.0x
.

GPU: ██████████████████████████░░░░ 2.6x
.

TPU: ██████████████████████████████ 3.5x

---

## 🎯 Tasks Completed

### ✅ Task 1: Understanding MaxText Data Formats
- Identified available data formats: **synthetic**, **c4**, **huggingface**
- Analyzed tradeoffs: speed vs realism
- Used **synthetic** data for all runs (fastest for benchmarking)
- Documented findings in `analysis/final_analysis.md`

### ✅ Task 2: Dense Model (Qwen 0.6B → Scaled)
- Trained Qwen 0.6B equivalent on CPU, GPU, and TPU
- 50 training steps per run
- Scaled model appropriately for Colab hardware
- Documented performance differences and interpretations

### ✅ Task 3: MoE Model (DeepSeek)
- Understood MoE architecture: expert count, top-k routing
- Explained scaling DeepSeek under 1B parameters
- Compared MoE vs dense model performance

---

## 🛠️ Technologies Used

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.12 | Programming language |
| **JAX** | 0.4.20 | CPU training framework |
| **PyTorch** | 2.2.0 / 2.9.0 | GPU/TPU training framework |
| **Google Colab** | Free Tier | Cloud-based execution environment |
| **NumPy** | 1.26.4 | Numerical computing |

---

## 📈 Key Insights

### Why Performance Differs

| Backend | Architecture | Memory | Best For |
|---------|--------------|--------|----------|
| **CPU** | General-purpose | System RAM | Prototyping, small models |
| **GPU** | Parallel cores | VRAM (16GB) | Development, medium models |
| **TPU** | Matrix-optimized | HBM | Production, large models |

### Why TPU Performed Best
1. **Specialized hardware** for matrix multiplication
2. **High-bandwidth memory** (HBM) architecture
3. **Tensor cores** optimized for neural networks
4. **8 cores** working in parallel

### Why GPU Performed Well
1. **Thousands of CUDA cores** for parallel computation
2. **PyTorch's CUDA optimization**
3. **Good balance** of speed and accessibility

### Why CPU Was Slowest
1. **Limited cores** (typically 4-8)
2. **General-purpose architecture** (not ML-optimized)
3. **System RAM** vs specialized VRAM/HBM

### Dense vs MoE Comparison

| Aspect | Dense (Qwen) | MoE (DeepSeek) |
|--------|--------------|----------------|
| **Architecture** | All parameters active | Sparse activation (top-k) |
| **Parameter Efficiency** | Lower | Higher (active vs total) |
| **Training Complexity** | Simpler | More complex |
| **Inference Speed** | Good | Can be faster |
| **Memory Usage** | Higher | Lower (with sparsity) |

---

## 🚀 How to Run the Code

### Prerequisites
1. Google Colab account (free)
2. GitHub account (for submission)

### Steps

1. **Open any notebook** in Google Colab:
   - `code/ZenteiQ_CPU_Assignment.ipynb` → Set runtime to **CPU**
   - `code/ZenteiQ_GPU_Assignment.ipynb` → Set runtime to **GPU (T4)**
   - `code/ZenteiQ_tpu_Assignment.ipynb` → Set runtime to **TPU**

2. **Run all cells** (Runtime → Run all)

3. **Training takes ~1-2 seconds** per 50 steps

4. **Results are automatically saved** to `.txt` files and downloaded

---

## 📸 Screenshots

| Figure | Description |
|--------|-------------|
| [CPU Runtime Setting](screenshots/CPU_Runtime_Setting.jpg) | CPU runtime configuration |
| [GPU Runtime Setting](screenshots/GPU_Runtime_Setting.jpg) | GPU runtime configuration |
| [TPU Runtime Setting](screenshots/TPU_Runtime_Setting.jpg) | TPU runtime configuration |
| [Figure 1](screenshots/figure1_cpu_training.png.jpg) | CPU Training Results (17.4% loss decrease) |
| [Figure 2](screenshots/figure2_gpu_training.png.jpg) | GPU Training Results (31.5% loss decrease) |
| [Figure 3](screenshots/figure3_tpu_training.png.jpg) | TPU Training Results (67.0% loss decrease) |
| [Figure 4](screenshots/figure4_comparison_table.png.jpg) | Performance Comparison Table |

---

## 📝 Findings & Observations

### Unexpected Findings
1. **JAX compatibility issues** on Colab GPU required switching to PyTorch
2. **Loss oscillated** on GPU and TPU due to high learning rate
3. **TPU detected only 1 core** in Colab (expected 8) but still performed best
4. **NumPy 2.0 conflicts** required downgrading to 1.26.4

### Key Takeaways
1. **Performance hierarchy**: TPU > GPU > CPU
2. **TPU is 3.5x faster** than CPU for this task
3. **TPU shows best learning** (67% loss decrease vs 17.4% on CPU)
4. **Framework choice matters**: PyTorch was more stable on Colab
5. **MoE routing overhead** makes training more complex

---

## 📧 Contact

| **Name** | Mahek Begum |
|----------|-------------|
| **Email** | mahekbegum54@gmail.com |
| **Phone** | 6304297409 |
| **GitHub** | [mahekbegum122](https://github.com/mahekbegum122) |
| **LinkedIn** | [Mahek Begum](https://www.linkedin.com/in/mahek-begum-620098328/) |
| **Portfolio** | [Mahek Portfolio](https://github.com/mahekbegum122/Mahek_Portfolio/blob/main/README.md) |

---

## 📅 Submission Details

| **Company** | ZenteiQ AiTech Innovations |
|-------------|----------------------------|
| **Position** | ML Engineer |
| **Submission Date** | June 2026 |
| **Repository** | [zenteiq-Assignment_Final](https://github.com/mahekbegum122/zenteiq-Assignment_Final) |

---

**Thank you for reviewing my assignment! I look forward to discussing it in the next step.** 🎉
