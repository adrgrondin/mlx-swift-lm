// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Text inference for Bonsai 2, including language weights from vision-capable packs.
final class PrismHadamardQwen35: Qwen35Model {
    private let packedConfiguration: PrismHadamardConfiguration

    init(configuration data: Data) throws {
        let decoder = JSONDecoder.json5()
        packedConfiguration = try decoder.decode(PrismHadamardConfiguration.self, from: data)
        let configuration = try decoder.decode(Qwen35Configuration.self, from: data)
        super.init(configuration)
        try packedConfiguration.install(in: self)
    }

    private var checkpointMapping: CheckpointNameMapping {
        if packedConfiguration.schemaVersion == 1 {
            return .init([.replacePrefix("", with: "language_model")])
        }
        let visionNamespaces = (modelComponents[.vision] ?? []).flatMap(\.namespaces)
        return .init((excludedCheckpointNamespaces + visionNamespaces).map { .excludePrefix($0) })
    }

    override func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let checkpoint = try checkpoint.mapNames(using: checkpointMapping)
        try validate(weights: checkpoint.weights)
        return checkpoint
    }

    override func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        // Packed tensors and norms already use MLX layout.
        try ModelCheckpoint(weights: weights).mapNames(using: checkpointMapping).weights
    }

    func validate(weights: [String: MLXArray]) throws {
        try packedConfiguration.validate(weights: weights, in: self)
    }
}
