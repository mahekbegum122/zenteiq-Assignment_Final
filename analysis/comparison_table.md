# Performance Comparison Table

## Dense Model (Qwen 0.6B) - 50 Steps Training

| Metric | CPU (JAX) | GPU (PyTorch) | TPU (PyTorch/XLA) |
|--------|-----------|---------------|-------------------|
| **Device** | TFRT_CPU_0 | cuda (Tesla T4) | xla:0 (TPU) |
| **Framework** | JAX 0.4.20 | PyTorch 2.2.0 | PyTorch 2.9.0 |
| **NumPy Version** | 1.26.4 | 1.26.4 | 2.0.2 |
| **Avg Step Time (s)** | 0.0209 | 0.0081 | 0.0060 |
| **Total Training Time (s)** | 1.05 | 0.41 | 0.30 |
| **Initial Loss** | 4.6209 | 4.9787 | 4.7204 |
| **Final Loss** | 3.8165 | 3.4110 | 1.5594 |
| **Loss Decrease** | 0.8044 (17.4%) | 1.5677 (31.5%) | 3.1610 (67.0%) |
| **Speedup vs CPU** | 1.0x | 2.6x | 3.5x |
| **Steps Completed** | 50 | 50 | 50 |

---

## Performance Visualization

### Speed Comparison (Faster = Better)

### Step Time Comparison (Lower = Better)


### Loss Decrease (Higher = Better)


---

## Key Insights

### 🏆 Performance Ranking
1. **TPU**: Fastest (3.5x CPU speedup) & Best Learning (67% loss decrease)
2. **GPU**: Fast (2.6x CPU speedup) & Good Learning (31.5% loss decrease)
3. **CPU**: Baseline (1.0x) & Moderate Learning (17.4% loss decrease)

### 📈 Why TPU Performed Best
- Specialized hardware for matrix multiplication
- High-bandwidth memory (HBM) architecture
- Optimized tensor cores for neural networks

### 💡 Why GPU Performed Well
- Thousands of CUDA cores for parallel computation
- PyTorch's CUDA optimization
- Good balance of speed and accessibility

### ⚠️ Why CPU Was Slowest
- General-purpose processor (limited parallelism)
- Not optimized for matrix operations
- System RAM vs specialized VRAM/HBM

---

## MoE vs Dense Model Comparison

| Aspect | Dense (Qwen) | MoE (DeepSeek) |
|--------|--------------|----------------|
| **Architecture** | All parameters active | Sparse activation (top-k routing) |
| **Parameter Efficiency** | Lower | Higher (active vs total) |
| **Training Complexity** | Simpler | More complex |
| **Inference Speed** | Good | Can be faster |
| **Memory Usage** | Higher | Lower (with sparsity) |

### Why MoE Numbers Differ From Dense
1. **Routing Overhead**: Extra computation to decide which experts to use
2. **Active vs Total Parameters**: Only selected experts are active
3. **Communication Overhead**: When experts are on different devices
4. **Training Stability**: More challenging due to routing decisions

---

## Conclusion

The performance hierarchy **TPU > GPU > CPU** holds true for this task:
- **TPU**: Best for large-scale production training
- **GPU**: Best for development and medium-scale training
- **CPU**: Best for prototyping and debugging

---

*Data collected on Google Colab (Free Tier) - June 2026*
