# Comparison Table - CPU vs GPU vs TPU

## Training Results (50 steps, synthetic data)

| Metric | CPU | GPU | TPU |
|--------|-----|-----|-----|
| **Device** | CPU | Tesla T4 | TPU |
| **Avg Step Time** | 0.0209s | 0.0081s | 0.0060s |
| **Total Time** | 1.05s | 0.41s | 0.30s |
| **Loss Decrease** | 17.4% | 31.5% | 67.0% |
| **Speed vs CPU** | 1.0x | 2.6x | 3.5x |

## Summary

- **TPU** = Fastest and best learning (67% loss decrease)
- **GPU** = 2.6x faster than CPU
- **CPU** = Slowest but works
