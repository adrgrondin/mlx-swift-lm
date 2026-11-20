// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen3.5 vision-language model loaded from a Hadamard-rotated ternary pack
/// (`model_type: prism_hadamard_qwen35`, `base_model_type: qwen3_5`).
///
/// The stock ``Qwen35`` with the manifest's language-model modules replaced by rotated layers
/// before the weights load. `HadamardQuantizedCheckpointPolicy` prepares the checkpoint after
/// ``Qwen35`` has mapped its names. The vision tower is not rotated. It still loads on first use,
/// through the same `prepareCheckpoint`.
public final class PrismHadamardQwen35: Qwen35 {

    /// The pack's manifest, validated for the Qwen3.5 vision class.
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
        guard pack.hasVision != false else {
            throw PrismHadamardError.invalidCheckpoint("checkpoint does not declare vision support")
        }
        let configuration = try decoder.decode(Qwen35Configuration.self, from: data)
        try self.init(configuration, manifest: pack.manifest, packedConfiguration: pack.legacy)
    }

    private init(
        _ configuration: Qwen35Configuration, manifest: HadamardQuantizedManifest,
        packedConfiguration: PrismHadamardConfiguration?
    ) throws {
        self.packedConfiguration = packedConfiguration
        if let packedConfiguration,
            packedConfiguration.schemaVersion != 2 || !packedConfiguration.hasVision
        {
            throw PrismHadamardError.invalidCheckpoint(
                "vision inference requires a schema-2 vision pack")
        }
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
            packedConfiguration == nil ? try super.prepareCheckpoint(checkpoint) : checkpoint
        return try checkpointPolicy.preparePrismCheckpoint(prepared, in: self)
    }

    public override func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        packedConfiguration == nil ? try super.sanitize(weights: weights) : weights
    }

    public override func sanitize(weights: [String: MLXArray], metadata: [String: String]) throws
        -> [String: MLXArray]
    {
        packedConfiguration == nil
            ? try super.sanitize(weights: weights, metadata: metadata) : weights
    }
}

func createPrismHadamardQwen35Model(configuration data: Data) throws -> any LanguageModel {
    try PrismHadamardQwen35(configuration: data)
}
