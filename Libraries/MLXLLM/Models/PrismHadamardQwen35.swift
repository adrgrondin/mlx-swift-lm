// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen3.5 text model loaded from a Hadamard-rotated ternary pack
/// (`model_type: prism_hadamard_qwen35`, `base_model_type: qwen3_5`).
///
/// The stock ``Qwen35Model`` with the manifest's modules replaced by rotated layers before the
/// weights load. `HadamardQuantizedCheckpointPolicy` prepares the checkpoint after
/// ``Qwen35Model`` has mapped its names.
public final class PrismHadamardQwen35Model: Qwen35Model {

    /// The pack's manifest, validated for the Qwen3.5 text class.
    public let checkpointPolicy: HadamardQuantizedCheckpointPolicy
    private let packedConfiguration: PrismHadamardConfiguration?

    /// Builds the base model and substitutes the manifest modules.
    ///
    /// - Throws: `HadamardQuantizedCheckpointError` when the manifest is not one this loader
    ///   can honor or names a module the base model does not have.
    public convenience init(
        _ configuration: Qwen35Configuration, manifest: HadamardQuantizedManifest
    )
        throws
    {
        try self.init(configuration, manifest: manifest, packedConfiguration: nil)
    }

    convenience init(configuration data: Data) throws {
        let decoder = JSONDecoder.json5()
        let pack = try PrismHadamardConfiguration.loadManifest(from: data, decoder: decoder)
        let configuration = try decoder.decode(Qwen35Configuration.self, from: data)
        try self.init(configuration, manifest: pack.manifest, packedConfiguration: pack.legacy)
    }

    private init(
        _ configuration: Qwen35Configuration, manifest: HadamardQuantizedManifest,
        packedConfiguration: PrismHadamardConfiguration?
    ) throws {
        self.packedConfiguration = packedConfiguration
        checkpointPolicy = try HadamardQuantizedCheckpointPolicy(
            manifest: manifest, baseModelType: "qwen3_5", tensorNamespace: "mlx-vlm-qwen3_5",
            pathPrefix: "language_model.")
        super.init(configuration)
        if let packedConfiguration {
            try packedConfiguration.install(in: self)
        } else {
            try checkpointPolicy.substituteModules(in: self)
        }
    }

    public override func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws
        -> ModelCheckpoint
    {
        let prepared =
            if packedConfiguration != nil {
                try checkpoint.mapNames(using: checkpointMapping)
            } else {
                try super.prepareCheckpoint(checkpoint)
            }
        return try checkpointPolicy.preparePrismCheckpoint(prepared, in: self)
    }

    private var checkpointMapping: CheckpointNameMapping {
        if packedConfiguration?.schemaVersion == 1 {
            return .init([.replacePrefix("", with: "language_model")])
        }
        let visionNamespaces = (modelComponents[.vision] ?? []).flatMap(\.namespaces)
        return .init((excludedCheckpointNamespaces + visionNamespaces).map { .excludePrefix($0) })
    }

    public override func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let mapped =
            if packedConfiguration != nil {
                try ModelCheckpoint(weights: weights).mapNames(using: checkpointMapping)
            } else {
                ModelCheckpoint(weights: try super.sanitize(weights: weights))
            }
        // The app's mapped loader validates its own packed arrays and keeps them file-backed.
        return checkpointPolicy.normalizingUnpackedWeights(in: mapped).weights
    }
}

func createPrismHadamardQwen35Model(configuration data: Data) throws -> any LanguageModel {
    try PrismHadamardQwen35Model(configuration: data)
}
