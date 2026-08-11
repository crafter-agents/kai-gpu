import Metal
import MetalPerformanceShaders
import Foundation

// fp32 GEMM through Apple's MetalPerformanceShaders implementation. Times the GPU with
// MTLCommandBuffer GPUStartTime/GPUEndTime and reports best-of-N throughput.

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
guard MPSSupportsMTLDevice(dev) else { fatalError("MPS does not support this Metal device") }
print("=== M4 GPU ===")
print("name:", dev.name)
print("unified memory:", dev.hasUnifiedMemory)
print("threadgroup mem:", dev.maxThreadgroupMemoryLength, "bytes")
print("max buffer:", dev.maxBufferLength / 1024 / 1024, "MB")
print("recommended working set:", dev.recommendedMaxWorkingSetSize / 1024 / 1024, "MB")

let coreProc = Process()
coreProc.executableURL = URL(fileURLWithPath: "/usr/sbin/system_profiler")
coreProc.arguments = ["SPDisplaysDataType"]
let cp = Pipe()
coreProc.standardOutput = cp
try? coreProc.run()
coreProc.waitUntilExit()
let cpo = String(data: cp.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
for line in cpo.split(separator: "\n") where line.contains("Total Number of Cores") {
    print("GPU", line.trimmingCharacters(in: .whitespaces))
}

let M = 2048
let N = 2048
let K = 2048
let iterations = 6
let rowBytesA = MPSMatrixDescriptor.rowBytes(forColumns: K, dataType: .float32)
let rowBytesB = MPSMatrixDescriptor.rowBytes(forColumns: N, dataType: .float32)
let rowBytesC = MPSMatrixDescriptor.rowBytes(forColumns: N, dataType: .float32)

func deterministicBuffer(rows: Int, columns: Int, rowBytes: Int) -> MTLBuffer {
    let buffer = dev.makeBuffer(length: rows * rowBytes, options: .storageModeShared)!
    let valuesPerRow = rowBytes / MemoryLayout<Float>.stride
    let values = buffer.contents().bindMemory(to: Float.self, capacity: rows * valuesPerRow)
    for row in 0..<rows {
        for column in 0..<columns {
            values[row * valuesPerRow + column] = ((row + column) & 1) == 0 ? 1.0 : 0.5
        }
    }
    return buffer
}

let bufferA = deterministicBuffer(rows: M, columns: K, rowBytes: rowBytesA)
let bufferB = deterministicBuffer(rows: K, columns: N, rowBytes: rowBytesB)
let bufferC = dev.makeBuffer(length: M * rowBytesC, options: .storageModeShared)!

let matrixA = MPSMatrix(buffer: bufferA, descriptor: MPSMatrixDescriptor(
    rows: M, columns: K, rowBytes: rowBytesA, dataType: .float32))
let matrixB = MPSMatrix(buffer: bufferB, descriptor: MPSMatrixDescriptor(
    rows: K, columns: N, rowBytes: rowBytesB, dataType: .float32))
let matrixC = MPSMatrix(buffer: bufferC, descriptor: MPSMatrixDescriptor(
    rows: M, columns: N, rowBytes: rowBytesC, dataType: .float32))

let gemm = MPSMatrixMultiplication(
    device: dev,
    transposeLeft: false,
    transposeRight: false,
    resultRows: M,
    resultColumns: N,
    interiorColumns: K,
    alpha: 1.0,
    beta: 0.0)
let queue = dev.makeCommandQueue()!
let totalFlops = 2.0 * Double(M) * Double(N) * Double(K)

func runOnce() -> Double {
    let commandBuffer = queue.makeCommandBuffer()!
    gemm.encode(commandBuffer: commandBuffer,
                leftMatrix: matrixA,
                rightMatrix: matrixB,
                resultMatrix: matrixC)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        fatalError("MPS GEMM command buffer failed: \(error)")
    }
    return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
}

print("\n=== MPSMatrixMultiplication fp32 GEMM (M=\(M), N=\(N), K=\(K)) ===")
var best = Double.greatestFiniteMagnitude
for iteration in 1...iterations {
    let time = runOnce()
    let gflops = totalFlops / time / 1e9
    print(String(format: "run %d: %.3f ms  -> %.1f GFLOPS", iteration, time * 1000, gflops))
    best = min(best, time)
}
let bestGflops = totalFlops / best / 1e9
print(String(format: "\nBEST: %.1f GFLOPS (%.2f TFLOPS) fp32 MPSMatrixMultiplication, M=%d N=%d K=%d, best of %d",
             bestGflops, bestGflops / 1000, M, N, K, iterations))
