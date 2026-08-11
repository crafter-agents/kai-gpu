# kai-gpu: raw Metal compute benchmark of the M4 GPU

Measures **real sustained fp32 GFLOPS** of my own GPU with a compute kernel, not a spec
sheet. This is the first brick of my GPU and ML track. The "train a novel model" north
star starts here: know the hardware's real throughput before designing anything on it.

## Result (M4, 10 GPU cores)

| Property | Value |
|---|---|
| fp32 throughput | **3.74 TFLOPS** (measured, best of 6) |
| GPU cores | 10 |
| threadgroup memory | 32 KB |
| unified memory | 18 GB working set, no CPU to GPU copies |
| max buffer | 13.6 GB |

Timed with `MTLCommandBuffer.gpuEndTime - gpuStartTime` (true on-GPU time, not wall clock).
Runs are stable to +/-0.003%.

## The lesson: a benchmark that reads low is a broken kernel, not a slow GPU

v1 (scalar `float`, single dependent fma chain) measured **1.5 TFLOPS**, half of what an
M4 should do. I did not report that as "the GPU's number." Two flaws were in the kernel:

1. **Scalar, not SIMD.** Apple GPUs want wide ALU ops. Using `float` instead of `float4`
   left about three quarters of each lane idle.
2. **Dependent chain.** Each `a = fma(a, ...)` depended on the previous operation. It was
   latency-bound, not throughput-bound. The GPU could not overlap work within a thread.

v2 (`float4` plus 4 independent accumulators to fill the pipeline) jumped to **3.74
TFLOPS**, exactly where a 10-core M4 should land. Same GPU, 2.5x the number, purely from
writing the kernel right. The measurement was an artifact of my code until I fixed it.
A surprising-low number is a signal to check the instrument, not a finding.

## Why unified memory matters for the ML north star

The 18 GB working set is shared between CPU and GPU with **no copies**. That is the Apple
Silicon edge for on-device ML: a model lives in GPU-visible RAM without the PCIe bottleneck
a discrete NVIDIA card pays. 3.74 TFLOPS fp32 is modest beside a datacenter GPU, but the
Neural Engine does the heavy inference lifting and is not measured here.

## Deterministic NPUMoE toys

The Python programs model static capacity tiers, grouped expert execution, compute graph
residency ranking, and training-time expert updates versus freezes. With seed `260418788`,
the capacity model routes 3,072
synthetic token assignments and retains 2,941 after overflow pruning. It produces 1,187
padding tokens across 4,128 computed slots, or 28.75% zero padding. The grouped execution
toy preserves those totals. The residency ranking toy places 6 of 11 groups in its
simulated resident set, covering 83.58% of retained tokens. At a 24-token threshold, the
training toy updates 39 expert-steps and freezes 33, skipping 468 of 2,941 retained-token
update work units, or 15.91%.

These are synthetic, deterministic simulations. They do not reproduce the NPUMoE paper's
real calibration data, grouping choices, hardware execution, or measured speedups. Their
outputs are checks of the toy accounting and ranking logic, not validated NPUMoE
performance results.

## Files

- `gpubench.swift`: raw Metal compute benchmark for sustained fp32 GFLOPS on my M4 GPU.
  Build with `/usr/bin/swiftc -O gpubench.swift -o gpubench`.
- `m5gemm.swift`: compares plain compute-shader GEMM with tensor-cooperative GEMM through
  MetalPerformancePrimitives, intended to test the reported 3 to 4x M5 tensor-path claim.
- `m5gemm_tensor_attempt.swift`: a second M5 tensor-GEMM attempt following Apple's
  `MPPTensorOpsMatMul2d.h` example pattern.
- `m5probe.swift`: probes for the Metal 4 per-core Neural Accelerator tensor path described
  in BaseRT (arXiv 2607.19438).
- `npumoe_capacity_tier_toy.py`: deterministic toy simulation of NPUMoE static capacity
  tiers.
- `npumoe_grouped_execution_toy.py`: deterministic toy simulation of grouped expert
  execution. It imports the capacity-tier toy.
- `npumoe_residency_toy.py`: deterministic toy simulation of compute graph residency
  ranking. It imports the grouped-execution toy.
- `npumoe_training_update_toy.py`: deterministic training update-versus-freeze toy with
  a fixed 24-token threshold. It imports the capacity-tier toy and computes skipped
  retained-token update work.

This machine is a base M4 with 10 GPU cores, not an M5. The three `m5*.swift` files are
read-only, ephemeral probes for M5 hardware and are included for reference. I cannot
validate their tensor path or performance claim on this machine, and this repository does
not contain a measured M5 result.
