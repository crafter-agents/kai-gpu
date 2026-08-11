import Metal
import Foundation

// Raw Metal compute benchmark for the M4 GPU. Measures real sustained GFLOPS with an
// FMA-heavy kernel: millions of threads each doing a long chain of fused multiply-adds
// (2 FLOPs per fma). Times the GPU with MTLCommandBuffer GPUStartTime/GPUEndTime (true
// on-GPU time, not wall clock). Reports device introspection + best-of-N GFLOPS.

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
print("=== M4 GPU ===")
print("name:", dev.name)
print("unified memory:", dev.hasUnifiedMemory)
print("threadgroup mem:", dev.maxThreadgroupMemoryLength, "bytes")
print("max buffer:", dev.maxBufferLength / 1024 / 1024, "MB")
print("recommended working set:", dev.recommendedMaxWorkingSetSize / 1024 / 1024, "MB")
// GPU core count from ioreg (Metal doesn't expose it directly)
let coreProc = Process()
coreProc.executableURL = URL(fileURLWithPath: "/usr/sbin/system_profiler")
coreProc.arguments = ["SPDisplaysDataType"]
let cp = Pipe(); coreProc.standardOutput = cp
try? coreProc.run(); coreProc.waitUntilExit()
let cpo = String(data: cp.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
for line in cpo.split(separator: "\n") where line.contains("Total Number of Cores") {
    print("GPU", line.trimmingCharacters(in: .whitespaces))
}

// FMA-heavy kernel v2: float4 SIMD (4-wide) + 4 INDEPENDENT accumulators to fill the
// ALU pipeline (break the dependency chain that was throttling v1). Pure register work,
// no memory traffic. FLOPs per iter: 4 accumulators * 4 lanes * 1 fma * 2 FLOP = 32.
let INNER = 4096
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void burn(device float4* out [[buffer(0)]],
                 constant uint& inner [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    float4 a = float4(gid) * 0.0001f + 1.0f;
    float4 d = a * 1.5f + 0.5f;
    float4 e = a * 0.7f + 0.3f;
    float4 f = a * 1.1f + 0.9f;
    float4 b = float4(1.0000001f);
    float4 c = float4(0.9999999f);
    for (uint i = 0; i < inner; i++) {
        a = fma(a, b, c);
        d = fma(d, c, b);
        e = fma(e, b, c);
        f = fma(f, c, b);
    }
    out[gid] = a + d + e + f;
}
"""
let lib = try dev.makeLibrary(source: src, options: nil)
let fn = lib.makeFunction(name: "burn")!
let pipe = try dev.makeComputePipelineState(function: fn)
let q = dev.makeCommandQueue()!

let N = 1 << 22           // 4M threads
let outBuf = dev.makeBuffer(length: N * 16, options: .storageModePrivate)!   // float4 = 16B
var inner = UInt32(INNER)
let innerBuf = dev.makeBuffer(bytes: &inner, length: 4, options: .storageModeShared)!

// FLOPs per iter = 4 accumulators * 4 lanes * 1 fma * 2 FLOP = 32
let totalFlops = Double(N) * Double(INNER) * 32.0

func runOnce() -> Double {
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(outBuf, offset: 0, index: 0)
    enc.setBuffer(innerBuf, offset: 0, index: 1)
    let tpg = pipe.maxTotalThreadsPerThreadgroup
    enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let gpuTime = cb.gpuEndTime - cb.gpuStartTime   // seconds, true GPU time
    return gpuTime
}

print("\n=== benchmark (4M threads, \(INNER) fma-pairs each) ===")
var best = Double.greatestFiniteMagnitude
for i in 1...6 {
    let t = runOnce()
    let gflops = totalFlops / t / 1e9
    print(String(format: "run %d: %.3f ms  -> %.1f GFLOPS", i, t*1000, gflops))
    best = min(best, t)
}
let bestGflops = totalFlops / best / 1e9
print(String(format: "\nBEST: %.1f GFLOPS (%.2f TFLOPS) fp32", bestGflops, bestGflops/1000))

let INNER2 = 512
let src2 = """
#include <metal_stdlib>
using namespace metal;
kernel void simdmm(device float* out [[buffer(0)]],
                   constant uint& inner [[buffer(1)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]]) {
    simdgroup_float8x8 base = simdgroup_float8x8(1.0001);
    simdgroup_float8x8 mulA = simdgroup_float8x8(1.0000001);
    simdgroup_float8x8 mulB = simdgroup_float8x8(0.9999999);
    simdgroup_float8x8 acc0 = base, acc1 = base, acc2 = base, acc3 = base;
    for (uint i = 0; i < inner; i++) {
        simdgroup_multiply_accumulate(acc0, acc0, mulA, acc0);
        simdgroup_multiply_accumulate(acc1, acc1, mulB, acc1);
        simdgroup_multiply_accumulate(acc2, acc2, mulA, acc2);
        simdgroup_multiply_accumulate(acc3, acc3, mulB, acc3);
    }
    if (tid == 0) {
        threadgroup float tmp[64];
        simdgroup_store(acc0, tmp, 8);
        out[tgid] = tmp[0] + tmp[1];
    }
}
"""
let lib2 = try dev.makeLibrary(source: src2, options: nil)
let fn2 = lib2.makeFunction(name: "simdmm")!
let pipe2 = try dev.makeComputePipelineState(function: fn2)

let numGroups2 = 1 << 15  // 32768 simdgroups, one 32-thread threadgroup each
let outBuf2 = dev.makeBuffer(length: numGroups2 * 4, options: .storageModeShared)!
var inner2 = UInt32(INNER2)
let innerBuf2 = dev.makeBuffer(bytes: &inner2, length: 4, options: .storageModeShared)!

// FLOPs: each simdgroup_multiply_accumulate on 8x8x8 = 2*8*8*8 = 1024 FLOPs.
// 4 independent accumulators * INNER2 iterations * numGroups2 threadgroups.
let totalFlops2 = Double(numGroups2) * Double(INNER2) * 4.0 * 1024.0

func runOnce2() -> Double {
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe2)
    enc.setBuffer(outBuf2, offset: 0, index: 0)
    enc.setBuffer(innerBuf2, offset: 0, index: 1)
    enc.dispatchThreadgroups(MTLSize(width: numGroups2, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}

print("\n=== simdgroup_matrix benchmark (\(numGroups2) simdgroups, \(INNER2) mma x4 each) ===")
var best2 = Double.greatestFiniteMagnitude
for i in 1...6 {
    let t = runOnce2()
    let gflops = totalFlops2 / t / 1e9
    print(String(format: "run %d: %.3f ms  -> %.1f GFLOPS", i, t*1000, gflops))
    best2 = min(best2, t)
}
let bestGflops2 = totalFlops2 / best2 / 1e9
print(String(format: "\nBEST: %.1f GFLOPS (%.2f TFLOPS) fp32 simdgroup_matrix", bestGflops2, bestGflops2/1000))
