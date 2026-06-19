# My Assignment Analysis

## Task 1: Data Formats

I used **synthetic** data for all my runs. This was the fastest option because it doesn't need to download anything. Other options like c4 or HuggingFace are more realistic but slower.

## Task 2: Dense Model (Qwen)

I trained a small model on CPU, GPU, and TPU. Here's what I found:

### Results
- **CPU**: Took 1.05 seconds, loss decreased 17.4%
- **GPU**: Took 0.41 seconds, loss decreased 31.5%
- **TPU**: Took 0.30 seconds, loss decreased 67.0%

### Why TPU is Fastest
TPUs have special hardware for matrix multiplication (the main operation in neural networks). They also have fast memory and multiple cores working together.

### Why GPU is Fast
GPUs have thousands of cores that can work in parallel. NVIDIA's CUDA software makes PyTorch run very efficiently on GPUs.

### Why CPU is Slowest
CPUs are general-purpose processors. They don't have special hardware for neural networks.

## Task 3: MoE Model (DeepSeek)

MoE (Mixture of Experts) models work differently than dense models:
- **Dense Models**: All parameters are always active
- **MoE Models**: Only some experts are active (top-k routing)

To scale DeepSeek under 1B parameters, I would:
1. Reduce `num_experts` from 64 to 4
2. Reduce `moe_intermediate_size` from 2048 to 256
3. Reduce `num_layers` from 30 to 8

### Why MoE is Different
MoE has routing overhead (deciding which experts to use). This makes training more complex. However, MoE can be more efficient because only some parameters are used for each input.

## Overall Conclusion

**Performance**: TPU > GPU > CPU

- TPU: Best for large production models
- GPU: Good for development and medium models  
- CPU: Best for simple testing
