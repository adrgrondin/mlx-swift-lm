// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

package enum PrismHadamardError: Error, LocalizedError {
    case invalidCheckpoint(String)

    package var errorDescription: String? {
        switch self {
        case .invalidCheckpoint(let reason): "Invalid Prism Hadamard checkpoint: \(reason)"
        }
    }
}

/// Metadata for the packed Qwen3.5 language layers used by Bonsai 2.
package struct PrismHadamardConfiguration: Decodable {
    package struct Record: Decodable {
        let path: String
        let block: Int
        let embedding: Bool
        let dtype: String
    }

    private struct Quantization: Decodable {
        let bits: Int
        let groupSize: Int
        let mode: String

        enum CodingKeys: String, CodingKey {
            case bits, mode
            case groupSize = "group_size"
        }
    }

    package let schemaVersion: Int
    package let modules: [Record]
    package let hasVision: Bool

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelType = "model_type"
        case baseModelType = "base_model_type"
        case tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout"
        case modules, components, quantization
    }

    package init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try c.decode(Int.self, forKey: .schemaVersion)
        let modelType = try c.decode(String.self, forKey: .modelType)
        let quantization = try c.decode(Quantization.self, forKey: .quantization)
        hasVision =
            try c.decodeIfPresent([String: Bool].self, forKey: .components)?["vision"] ?? false
        guard modelType == "prism_hadamard_qwen35", [1, 2].contains(schemaVersion),
            quantization.bits == 2, quantization.groupSize == 128, quantization.mode == "affine"
        else {
            throw PrismHadamardError.invalidCheckpoint("unsupported schema or quantization")
        }
        if schemaVersion == 2 {
            guard try c.decode(String.self, forKey: .baseModelType) == "qwen3_5",
                try c.decode(String.self, forKey: .tensorNamespace) == "mlx-vlm-qwen3_5",
                try c.decode(String.self, forKey: .gdnActivationLayout) == "grouped"
            else {
                throw PrismHadamardError.invalidCheckpoint(
                    "unsupported base model or tensor layout")
            }
        }
        modules = try c.decode([Record].self, forKey: .modules)
        var seen = Set<String>()
        guard !modules.isEmpty else {
            throw PrismHadamardError.invalidCheckpoint("empty packed-module manifest")
        }
        for record in modules {
            guard seen.insert(record.path).inserted,
                record.dtype == "float16", [0, 512, 1024, 2048, 4096].contains(record.block)
            else {
                throw PrismHadamardError.invalidCheckpoint(
                    "invalid or duplicate module \(record.path)")
            }
        }
    }

    /// Replace only the declared language layers. The vision tower stays unquantized.
    package func install(in model: Module) throws {
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var updates = [(String, Module)]()
        for record in modules {
            let path = "language_model." + record.path
            let rows: Int
            let width: Int
            if record.embedding, let embedding = leaves[path] as? Embedding {
                (rows, width) = embedding.shape
            } else if !record.embedding, let linear = leaves[path] as? Linear, linear.bias == nil {
                (rows, width) = linear.shape
            } else {
                throw PrismHadamardError.invalidCheckpoint(
                    "unsupported module target \(record.path)")
            }
            guard width > 0, width % 128 == 0,
                record.block == 0 || width % record.block == 0
            else {
                throw PrismHadamardError.invalidCheckpoint(
                    "invalid transform dimensions at \(record.path)")
            }
            let replacement: Module =
                record.embedding
                ? PrismHadamardEmbedding(rows: rows, width: width, block: record.block)
                : PrismHadamardLinear(rows: rows, width: width, block: record.block)
            updates.append((path, replacement))
        }
        try model.update(modules: ModuleChildren.unflattened(updates), verify: [.all])
    }

    /// Reject malformed tensors before parameter updates or any inference optimization.
    package func validate(weights: [String: MLXArray], in model: Module) throws {
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let paths = Set(modules.map { "language_model." + $0.path })
        for key in weights.keys where key.hasSuffix(".scales") {
            guard paths.contains(String(key.dropLast(".scales".count))) else {
                throw PrismHadamardError.invalidCheckpoint("undeclared packed module at \(key)")
            }
        }
        var signChecks = [MLXArray]()
        for record in modules {
            let path = "language_model." + record.path
            for (suffix, dtype) in [
                ("weight", DType.uint32), ("scales", .float16), ("biases", .float16),
            ] {
                let key = path + "." + suffix
                guard let array = weights[key], let parameter = expected[key],
                    array.shape == parameter.shape, array.dtype == dtype
                else {
                    throw PrismHadamardError.invalidCheckpoint("missing or invalid tensor \(key)")
                }
            }
            let key = path + ".signs"
            if record.block != 0 {
                guard let signs = weights[key], let parameter = expected[key],
                    signs.shape == parameter.shape, signs.dtype == .float32
                else {
                    throw PrismHadamardError.invalidCheckpoint(
                        "missing or invalid sign vector \(key)")
                }
                signChecks.append(((signs .== 1) .|| (signs .== -1)).all())
            } else if weights[key] != nil {
                throw PrismHadamardError.invalidCheckpoint("unexpected sign vector \(key)")
            }
        }
        if !signChecks.isEmpty, !stacked(signChecks).all().item(Bool.self) {
            throw PrismHadamardError.invalidCheckpoint("sign vectors must contain only -1 and +1")
        }
    }
}

/// The reference runtime computes the normalized transform in FP32, then restores the input dtype.
package func prismHadamardTransform(
    _ x: MLXArray, block: Int, signs: MLXArray?, inverse: Bool = false
) -> MLXArray {
    guard block != 0, let signs else { return x }
    var transformed = x.asType(.float32)
    if !inverse { transformed = transformed * signs }
    transformed = hadamardTransform(transformed.reshaped(-1, block)).reshaped(x.shape)
    if inverse { transformed = transformed * signs }
    return transformed.asType(x.dtype)
}

// Do not subclass QuantizedLinear: ordinary projection fusion would discard the transform.
package final class PrismHadamardLinear: Linear, Quantized {
    package let groupSize = 128
    package let bits = 2
    package let mode: QuantizationMode = .affine
    let scales: MLXArray
    let biases: MLXArray
    let signs: MLXArray?
    private let block: Int

    package init(rows: Int, width: Int, block: Int) {
        self.block = block
        scales = MLXArray.zeros([rows, width / 128], dtype: .float16)
        biases = MLXArray.zeros([rows, width / 128], dtype: .float16)
        signs = block == 0 ? nil : MLXArray.ones([width])
        super.init(weight: MLXArray.zeros([rows, width / 16], dtype: .uint32))
    }

    package override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 16) }

    package override func callAsFunction(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            prismHadamardTransform(x, block: block, signs: signs), weight,
            scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
    }
}

package final class PrismHadamardEmbedding: Embedding, Quantized {
    package let groupSize = 128
    package let bits = 2
    package let mode: QuantizationMode = .affine
    let scales: MLXArray
    let biases: MLXArray
    let signs: MLXArray?
    private let block: Int

    package init(rows: Int, width: Int, block: Int) {
        self.block = block
        scales = MLXArray.zeros([rows, width / 128], dtype: .float16)
        biases = MLXArray.zeros([rows, width / 128], dtype: .float16)
        signs = block == 0 ? nil : MLXArray.ones([width])
        super.init(weight: MLXArray.zeros([rows, width / 16], dtype: .uint32))
    }

    package override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 16) }

    package override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let indices = x.reshaped(-1)
        let rows = dequantized(
            weight[indices], scales: scales[indices], biases: biases[indices],
            groupSize: groupSize, bits: bits
        ).reshaped(x.shape + [shape.1]).asType(.float16)
        return prismHadamardTransform(rows, block: block, signs: signs, inverse: true)
    }

    package override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            prismHadamardTransform(x, block: block, signs: signs), weight,
            scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
    }
}
