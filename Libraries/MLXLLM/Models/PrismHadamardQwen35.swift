// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Text inference for Bonsai 2, including language weights from vision-capable packs.
final class PrismHadamardQwen35: Qwen35Model, ModelWeightValidating {
    private let packedConfiguration: PrismHadamardConfiguration

    init(configuration data: Data) throws {
        let decoder = JSONDecoder.json5()
        packedConfiguration = try decoder.decode(PrismHadamardConfiguration.self, from: data)
        let configuration = try decoder.decode(Qwen35Configuration.self, from: data)
        super.init(configuration)
        try packedConfiguration.install(in: self)
    }

    override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        if packedConfiguration.schemaVersion == 1 {
            return Dictionary(
                uniqueKeysWithValues: weights.map { ("language_model." + $0.key, $0.value) })
        }
        // Packed tensors and norms are already in MLX layout; do not shift or transpose them.
        return weights.filter { !$0.key.hasPrefix("vision_tower.") }
    }

    func validate(weights: [String: MLXArray]) throws {
        try packedConfiguration.validate(weights: weights, in: self)
    }
}
