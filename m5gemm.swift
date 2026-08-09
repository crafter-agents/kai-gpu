import Metal
import Foundation

// M5 Neural Accelerator validation: measure a GEMM two ways and compare.
//   (A) plain compute-shader GEMM (does NOT use the neural accelerators)
//   (B) tensor-cooperative GEMM via MetalPerformancePrimitives (uses them)
// SOTA claim (Apple + Creative Strategies + BaseRT): B is ~3-4x faster on M5. If measured,
// validates the per-core Neural Accelerator on real hardware. Read-only, /tmp, ephemeral.

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal") }
print("device:", dev.name)
let q = dev.makeCommandQueue()!
let SZ = 2048  // MxNxK square
let M = SZ, N = SZ, K = SZ
let flops = 2.0 * Double(M) * Double(N) * Double(K)

// buffers (fp16 in, fp32 out)
func rndBuf(_ n: Int) -> MTLBuffer {
    let b = dev.makeBuffer(length: n*2, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: UInt16.self, capacity: n)
    for i in 0..<n { p[i] = 0x3C00 } // 1.0 in fp16, deterministic
    return b
}
let A = rndBuf(M*K), B = rndBuf(K*N)
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
    float acc=0;
    for (uint k=0;k<K;k++) acc += float(A[g.y*K+k])*float(B[k*N+g.x]);
    C[g.y*N+g.x]=acc;
}
"""
let plainLib = try dev.makeLibrary(source: plainSrc, options: nil)
let plainPipe = try dev.makeComputePipelineState(function: plainLib.makeFunction(name: "gemm_plain")!)

func timePlain(iters: Int) -> Double {
    var best = Double.greatestFiniteMagnitude
    var m=UInt32(M), n=UInt32(N), k=UInt32(K)
    for _ in 0..<iters {
        let cb=q.makeCommandBuffer()!; let e=cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(plainPipe)
        e.setBuffer(A,offset:0,index:0); e.setBuffer(B,offset:0,index:1); e.setBuffer(C,offset:0,index:2)
        e.setBytes(&m,length:4,index:3); e.setBytes(&n,length:4,index:4); e.setBytes(&k,length:4,index:5)
        let tg=MTLSize(width:16,height:16,depth:1)
        e.dispatchThreads(MTLSize(width:N,height:M,depth:1),threadsPerThreadgroup:tg)
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        best=min(best, cb.gpuEndTime-cb.gpuStartTime)
    }
    return best
}

// (B) tensor-cooperative GEMM. Uses the MetalPerformancePrimitives matmul over cooperative
// tensors, the path that hits the Neural Accelerators. MSL spelling per Apple's Metal 4.
let tensorSrc = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// tile 32x32, K-loop; cooperative matmul accumulates in registers across the simdgroup.
kernel void gemm_tensor(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                        device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
                        constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                        uint2 tgid [[threadgroup_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr int T = 32;
    matmul2d_descriptor desc(T, T, T, false, false, false);
    matmul2d<matmul2d_descriptor(T,T,T,false,false,false), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(mm), float>();
    uint row0 = tgid.y*T, col0 = tgid.x*T;
    for (uint k0=0; k0<K; k0+=T) {
        auto aT = tensor(A + row0*K + k0, dextents<int,2>(K, T));
        auto bT = tensor(B + k0*N + col0, dextents<int,2>(N, T));
        mm.run(aT, bT, cT);
    }
    auto cout = tensor(C + row0*N + col0, dextents<int,2>(N, T));
    cT.store(cout);
}
"""
print("\n=== plain compute-shader GEMM (2048^3) ===")
let tP = timePlain(iters: 20)
print(String(format: "  %.3f ms  -> %.1f GFLOPS", tP*1000, flops/tP/1e9))

print("\n=== tensor-cooperative GEMM (Neural Accelerators) ===")
do {
    let tLib = try dev.makeLibrary(source: tensorSrc, options: nil)
    let tPipe = try dev.makeComputePipelineState(function: tLib.makeFunction(name: "gemm_tensor")!)
    var m=UInt32(M), n=UInt32(N), k=UInt32(K)
    var best = Double.greatestFiniteMagnitude
    for _ in 0..<20 {
        let cb=q.makeCommandBuffer()!; let e=cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(tPipe)
        e.setBuffer(A,offset:0,index:0); e.setBuffer(B,offset:0,index:1); e.setBuffer(C,offset:0,index:2)
        e.setBytes(&m,length:4,index:3); e.setBytes(&n,length:4,index:4); e.setBytes(&k,length:4,index:5)
        e.dispatchThreadgroups(MTLSize(width:N/32,height:M/32,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        best=min(best, cb.gpuEndTime-cb.gpuStartTime)
    }
    print(String(format: "  %.3f ms  -> %.1f GFLOPS", best*1000, flops/best/1e9))
    print(String(format: "\n  SPEEDUP tensor vs plain: %.2fx  (SOTA claims ~3-4x)", tP/best))
} catch {
    print("  tensor GEMM compile FAILED (MSL spelling / API surface):")
    print("  " + String(describing: error).prefix(500))
    print("\n  => hardware present, my cooperative-tensor MSL spelling is off. Honest limit.")
}
