import Metal
import Foundation

// M5 GEMM: plain compute-shader vs Metal-4 tensor (Neural Accelerators), tensor kernel now
// following Apple's OWN example from MPPTensorOpsMatMul2d.h (tensors as kernel args +
// static_slice + op.run). Read-only, /tmp, ephemeral.

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal") }
print("device:", dev.name)
let q = dev.makeCommandQueue()!
let SZ = 2048; let M = SZ, N = SZ, K = SZ
let flops = 2.0 * Double(M) * Double(N) * Double(K)

func onesBuf(_ n: Int) -> MTLBuffer {
    let b = dev.makeBuffer(length: n*2, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: UInt16.self, capacity: n)
    for i in 0..<n { p[i] = 0x3C00 }  // 1.0 fp16
    return b
}
let A = onesBuf(M*K), B = onesBuf(K*N)
let C = dev.makeBuffer(length: M*N*4, options: .storageModeShared)!

// (A) plain compute-shader GEMM
let plainSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void gemm_plain(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                       device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
                       constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                       uint2 g [[thread_position_in_grid]]) {
    if (g.x>=N||g.y>=M) return;
    float acc=0; for (uint k=0;k<K;k++) acc += float(A[g.y*K+k])*float(B[k*N+g.x]);
    C[g.y*N+g.x]=acc;
}
"""
let plainLib = try dev.makeLibrary(source: plainSrc, options: nil)
let plainPipe = try dev.makeComputePipelineState(function: plainLib.makeFunction(name: "gemm_plain")!)
var m=UInt32(M), n=UInt32(N), k=UInt32(K)
func timePlain() -> Double {
    var best = Double.greatestFiniteMagnitude
    for _ in 0..<20 {
        let cb=q.makeCommandBuffer()!; let e=cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(plainPipe)
        e.setBuffer(A,offset:0,index:0);e.setBuffer(B,offset:0,index:1);e.setBuffer(C,offset:0,index:2)
        e.setBytes(&m,length:4,index:3);e.setBytes(&n,length:4,index:4);e.setBytes(&k,length:4,index:5)
        e.dispatchThreads(MTLSize(width:N,height:M,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:16,depth:1))
        e.endEncoding();cb.commit();cb.waitUntilCompleted()
        best=min(best,cb.gpuEndTime-cb.gpuStartTime)
    }
    return best
}

// (B) tensor GEMM, Apple's exact pattern. Tensors passed as kernel args via the
// [[buffer]] + tensor<> binding. 64x32 * 32xK descriptor with internal k-loop
// (dynamic_extent on K lets the op loop internally).
let tSrc = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void gemm_tensor(tensor<device half,  dextents<int32_t,2>> A [[buffer(0)]],
                        tensor<device half,  dextents<int32_t,2>> B [[buffer(1)]],
                        tensor<device float, dextents<int32_t,2>> C [[buffer(2)]],
                        constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
                        constant uint& K [[buffer(5)]],
                        uint2 tgid [[threadgroup_position_in_grid]]) {
    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;
    if (tgid.y*64 + 63 < M && tgid.x*32 + 31 < N) {
        auto tA = A.static_slice<dynamic_extent, 64>(0, tgid.y*64);
        auto tB = B.static_slice<32, dynamic_extent>(tgid.x*32, 0);
        auto tC = C.static_slice<32, 64>(tgid.x*32, tgid.y*64);
        op.run(tA, tB, tC);
    }
}
"""
print("\n=== plain compute-shader GEMM (2048^3) ===")
let tP = timePlain()
print(String(format: "  %.3f ms  -> %.1f GFLOPS", tP*1000, flops/tP/1e9))

print("\n=== tensor GEMM (M5 Neural Accelerators) ===")
do {
    let tLib = try dev.makeLibrary(source: tSrc, options: nil)
    let tPipe = try dev.makeComputePipelineState(function: tLib.makeFunction(name: "gemm_tensor")!)
    // tensors are bound as buffers; Metal maps the tensor<> arg to the buffer + we pass
    // dimensions. Provide row-major extents via setBuffer (the tensor<> infers from binding).
    var best = Double.greatestFiniteMagnitude
    for _ in 0..<20 {
        let cb=q.makeCommandBuffer()!; let e=cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(tPipe)
        e.setBuffer(A,offset:0,index:0);e.setBuffer(B,offset:0,index:1);e.setBuffer(C,offset:0,index:2)
        e.setBytes(&m,length:4,index:3);e.setBytes(&n,length:4,index:4);e.setBytes(&k,length:4,index:5)
        e.dispatchThreadgroups(MTLSize(width:N/32,height:M/64,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
        e.endEncoding();cb.commit();cb.waitUntilCompleted()
        best=min(best,cb.gpuEndTime-cb.gpuStartTime)
    }
    print(String(format: "  %.3f ms  -> %.1f GFLOPS", best*1000, flops/best/1e9))
    print(String(format: "\n  SPEEDUP tensor vs plain: %.2fx  (SOTA: ~3-4x prefill)", tP/best))
} catch {
    print("  tensor GEMM failed:\n  " + String(describing: error).prefix(600))
}
