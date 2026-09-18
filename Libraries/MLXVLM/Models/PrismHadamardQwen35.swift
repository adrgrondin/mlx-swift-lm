// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Bonsai 2's packed language model with the unrotated Qwen3.5 vision tower.
final class PrismHadamardQwen35: Qwen35, ModelWeightValidating {
    private let packedConfiguration: PrismHadamardConfiguration

    init(configuration data: Data) throws {
        let decoder = JSONDecoder.json5()
        packedConfiguration = try decoder.decode(PrismHadamardConfiguration.self, from: data)
        guard packedConfiguration.schemaVersion == 2, packedConfiguration.hasVision else {
            throw PrismHadamardError.invalidCheckpoint(
                "vision inference requires a schema-2 vision pack")
        }
        let configuration = try decoder.decode(Qwen35Configuration.self, from: data)
        super.init(configuration)
        try packedConfiguration.install(in: self)
    }

    override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Both towers already use MLX layout, including convolution and norm weights.
        weights
    }

    override func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        weights
    }

    func validate(weights: [String: MLXArray]) throws {
        try packedConfiguration.validate(weights: weights, in: self)
    }
}
