// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Configuration of LiquidAI's LFM3 vision-language model.
public struct LFM3VLConfiguration: Codable, Sendable, ModelConfigurationValidating {
    public let textConfiguration: LFM3Configuration
    public let visionConfiguration: LFM3VisionConfiguration
    let downsampleFactor: Int
    let projectorHiddenSize: Int
    let projectorBias: Bool
    let projectorHiddenAct: String
    let projectorNormEps: Float
    let imageTokenId: Int
    let videoTokenId: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case downsampleFactor = "downsample_factor"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorBias = "projector_bias"
        case projectorHiddenAct = "projector_hidden_act"
        case projectorNormEps = "projector_layer_norm_eps"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
    }

    private enum CheckpointKeys: String, CodingKey {
        case architectures
        case autoMap = "auto_map"
    }

    public init(from decoder: Decoder) throws {
        let identity = try decoder.container(keyedBy: CheckpointKeys.self)
        let architectures =
            try identity.decodeIfPresent([String].self, forKey: .architectures) ?? []
        let autoMap = try identity.decodeIfPresent([String: String].self, forKey: .autoMap) ?? [:]
        guard !architectures.contains("D1Model"), autoMap["AutoModel"]?.hasSuffix("D1Model") != true
        else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3-VL does not support D1Model checkpoints; d1 requires set attention and typed readout"
            )
        }
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(LFM3Configuration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(
            LFM3VisionConfiguration.self, forKey: .visionConfiguration)
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        projectorHiddenSize = try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorHiddenAct =
            try c.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        projectorNormEps = try c.decodeIfPresent(Float.self, forKey: .projectorNormEps) ?? 1e-6
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 125030
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 125029
    }

    public func validateModelConfiguration() throws {
        try textConfiguration.validateModelConfiguration()
        try visionConfiguration.validateModelConfiguration()
        guard downsampleFactor == visionConfiguration.mergeSize, projectorHiddenSize > 0,
            projectorHiddenAct == "gelu", projectorNormEps.isFinite, projectorNormEps > 0,
            imageTokenId >= 0, imageTokenId < textConfiguration.vocabularySize,
            videoTokenId >= 0, videoTokenId < textConfiguration.vocabularySize,
            imageTokenId != videoTokenId
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Unsupported LFM3-VL projector or token configuration")
        }
    }
}

/// LFM3-VL supports text, multiple images, and timestamped video.
public final class LFM3VL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") var languageModel: LFM3LanguageModel
    @ModuleInfo(key: "vision_tower") var visionTower: LFM3VisionModel
    @ModuleInfo(key: "multi_modal_projector") var projector: LFM3MultiModalProjector
    public let configuration: LFM3VLConfiguration
    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }

    public init(_ configuration: LFM3VLConfiguration) {
        self.configuration = configuration
        _languageModel.wrappedValue = LFM3LanguageModel(configuration.textConfiguration)
        _visionTower.wrappedValue = LFM3VisionModel(configuration.visionConfiguration)
        _projector.wrappedValue = LFM3MultiModalProjector(configuration)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        let embeddings = languageModel.embed(tokens)
        let ids = tokens.asArray(Int.self)
        func insert(pixels: MLXArray?, grids: [THW]?, token: Int) throws {
            let positions = ids.enumerated().compactMap { $0.element == token ? $0.offset : nil }
            guard let pixels, let grids else {
                guard positions.isEmpty else {
                    throw VLMError.processing("LFM3 visual placeholders have no media")
                }
                return
            }
            guard tokens.dim(0) == 1 else {
                throw VLMError.processing("LFM3-VL media requires a single prompt")
            }
            let features = projector(try visionTower(pixels, grids: grids)).asType(embeddings.dtype)
            guard positions.count == features.dim(0) else {
                throw VLMError.processing("LFM3 visual features and placeholder counts differ")
            }
            embeddings[0, MLXArray(positions), 0...] = features
        }
        try insert(
            pixels: input.image?.pixels, grids: input.image?.frames,
            token: configuration.imageTokenId)
        try insert(
            pixels: input.video?.pixels, grids: input.video?.frames,
            token: configuration.videoTokenId)
        return .logits(
            try languageModel.prefill(
                embeddings: embeddings, lengths: input.text.sequenceLengths,
                cache: cache, prefill: prefill))
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = try checkpoint.mapNames(
            using: .init([
                .replacePrefix("model.language_model", with: "language_model.model"),
                .replacePrefix("model.vision_tower", with: "vision_tower"),
                .replacePrefix("model.multi_modal_projector", with: "multi_modal_projector"),
                .replacePrefix("lm_head", with: "language_model.lm_head"),
            ]))
        checkpoint = try prepareLFM3Checkpoint(
            checkpoint, tiedEmbeddings: configuration.textConfiguration.tieWordEmbeddings)
        for (name, value) in checkpoint.weights where name.hasSuffix("conv.spatial_conv.weight") {
            if value.ndim == 4 && value.dim(1) == 1 && value.dim(3) != 1 {
                checkpoint.weights[name] = value.transposed(0, 2, 3, 1)
            }
        }
        return checkpoint
    }
}
