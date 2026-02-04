import CoreML
import Foundation

guard CommandLine.arguments.count >= 5 else {
    fputs("Usage: infer <model.mlmodelc> <input.bin> <output_real.bin> <output_imag.bin> [cpu|gpu|all]\n", stderr)
    exit(1)
}

let modelPath = CommandLine.arguments[1]
let inputPath = CommandLine.arguments[2]
let outputRealPath = CommandLine.arguments[3]
let outputImagPath = CommandLine.arguments[4]
let computeArg = CommandLine.arguments.count > 5 ? CommandLine.arguments[5] : "all"

let modelURL = URL(fileURLWithPath: modelPath)
let config = MLModelConfiguration()

switch computeArg {
case "cpu":
    config.computeUnits = .cpuOnly
    fputs("Compute: CPU only\n", stderr)
case "gpu":
    config.computeUnits = .cpuAndGPU
    fputs("Compute: CPU + GPU\n", stderr)
default:
    config.computeUnits = .all
    fputs("Compute: All (CPU + GPU + ANE)\n", stderr)
}

fputs("Loading model...\n", stderr)
let startLoad = Date()
guard let model = try? MLModel(contentsOf: modelURL, configuration: config) else {
    fputs("ERROR: Failed to load model at \(modelPath)\n", stderr)
    exit(1)
}
let loadTime = Date().timeIntervalSince(startLoad)
fputs("Model loaded in \(String(format: "%.1f", loadTime))s\n", stderr)

guard let inputData = try? Data(contentsOf: URL(fileURLWithPath: inputPath)) else {
    fputs("ERROR: Failed to read input file \(inputPath)\n", stderr)
    exit(1)
}

let inputCount = 1 * 2 * 588800
let expectedBytes = inputCount * MemoryLayout<Float>.size
guard inputData.count == expectedBytes else {
    fputs("ERROR: Input file size \(inputData.count) != expected \(expectedBytes) bytes\n", stderr)
    exit(1)
}

guard let inputArray = try? MLMultiArray(shape: [1, 2, 588800], dataType: .float32) else {
    fputs("ERROR: Failed to create input MLMultiArray\n", stderr)
    exit(1)
}

let inputPtr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: inputCount)
inputData.withUnsafeBytes { rawBuffer in
    let srcPtr = rawBuffer.bindMemory(to: Float.self)
    inputPtr.update(from: srcPtr.baseAddress!, count: inputCount)
}

let inputFeature = try! MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: inputArray)])

fputs("Running inference...\n", stderr)
let startInfer = Date()
guard let output = try? model.prediction(from: inputFeature) else {
    fputs("ERROR: Inference failed\n", stderr)
    exit(1)
}
let inferTime = Date().timeIntervalSince(startInfer)
fputs("Inference done in \(String(format: "%.1f", inferTime))s\n", stderr)

guard let realArray = output.featureValue(for: "var_11707")?.multiArrayValue,
      let imagArray = output.featureValue(for: "var_11751")?.multiArrayValue else {
    fputs("ERROR: Could not extract output tensors. Available: ", stderr)
    for name in output.featureNames { fputs("\(name) ", stderr) }
    fputs("\n", stderr)
    exit(1)
}

let outputCount = 1 * 12 * 1151 * 1025

func writeArray(_ array: MLMultiArray, to path: String) {
    let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: outputCount)
    let data = Data(bytes: ptr, count: outputCount * MemoryLayout<Float>.size)
    try! data.write(to: URL(fileURLWithPath: path))
}

writeArray(realArray, to: outputRealPath)
writeArray(imagArray, to: outputImagPath)
fputs("Done.\n", stderr)
