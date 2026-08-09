import Metal
import Foundation

// M4 vs M5 probe. SOTA claim to validate (BaseRT, arXiv 2607.19438): the M5 puts matrix
// units INSIDE each GPU core (per-core Neural Accelerators), exposed via the Metal 4
// tensor API (MPP matmul2d over cooperative tensor fragments). Question: does this machine
// expose that tensor path, and which Metal family / features gate it? Read-only, ephemeral.

guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("no Metal") }
print("=== device ===")
print("name:", dev.name)
// chip string from sysctl
let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/sbin/sysctl")
p.arguments = ["-n", "machdep.cpu.brand_string"]
let pipe = Pipe(); p.standardOutput = pipe; try? p.run(); p.waitUntilExit()
let chip = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "?"
print("chip:", chip)

// GPU core count
let sp = Process(); sp.executableURL = URL(fileURLWithPath: "/usr/sbin/system_profiler")
sp.arguments = ["SPDisplaysDataType"]
let spp = Pipe(); sp.standardOutput = spp; try? sp.run(); sp.waitUntilExit()
let spo = String(data: spp.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
for line in spo.split(separator: "\n") where line.contains("Total Number of Cores") {
    print("GPU", line.trimmingCharacters(in: .whitespaces))
}
print("unified memory:", dev.hasUnifiedMemory)
print("recommended working set MB:", dev.recommendedMaxWorkingSetSize/1024/1024)

// Metal family support: apple9 is M3/M4-class; a new family / Metal4 tensor path would
// signal the M5 accelerators.
print("\n=== Metal GPU families ===")
let fams: [(String, MTLGPUFamily)] = [
    ("apple7", .apple7), ("apple8", .apple8), ("apple9", .apple9),
    ("metal3", .metal3),
]
for (n, f) in fams { print("  \(n): \(dev.supportsFamily(f))") }
// apple10 / metal4 may not exist in this SDK's enum; probe by rawValue to be forward-compatible.
// MTLGPUFamily.apple9 == 1009. Try apple10 (1010) via rawValue.
if let apple10 = MTLGPUFamily(rawValue: 1010) {
    print("  apple10 (probed): \(dev.supportsFamily(apple10))")
} else {
    print("  apple10: not in this SDK enum")
}
if let metal4 = MTLGPUFamily(rawValue: 5002) {  // metal3==5001; metal4 would be next
    print("  metal4 (probed 5002): \(dev.supportsFamily(metal4))")
} else {
    print("  metal4: not in this SDK enum")
}

// The tensor API: does MTLTensor / cooperative tensor exist at runtime?
print("\n=== Metal 4 tensor API presence (the M5 signal) ===")
let tensorClasses = ["MTLTensor", "MTL4CommandBuffer", "MTL4CommandQueue", "MTLTensorDescriptor", "MTL4ComputeCommandEncoder"]
for c in tensorClasses {
    print("  \(c): \(NSClassFromString(c) != nil ? "present" : "absent")")
}
// device selectors that would only exist with the tensor path
print("  dev responds to newTensorWithDescriptor:offset:error: \(dev.responds(to: NSSelectorFromString("newTensorWithDescriptor:offset:error:")))")
