//
//  Gemma4.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for the `"gemma4"` model_type.
/// This is a thin wrapper around `Gemma4TextConfiguration` that handles the
/// nested `text_config` structure from HuggingFace model configs.
public struct Gemma4Configuration: Codable, Sendable {
    var modelType: String = "gemma4"
    var textConfig: Gemma4TextConfiguration
    var vocabSize: Int = 262144
    // Gemma 4 QAT mobile (wNa8o8) quantization config. Lives at the top level
    // of the `gemma4` config (not inside `text_config`); propagated into
    // `textConfig` below so the text model can drive its own sanitize path.
    var quantizationConfig: GemmaMobileQuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case vocabSize = "vocab_size"
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        self.quantizationConfig = try container.decodeIfPresent(
            GemmaMobileQuantizationConfig.self, forKey: .quantizationConfig)

        // If text_config is present, decode from it; otherwise treat entire config as text config
        if let textConfig = try container.decodeIfPresent(
            Gemma4TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
            // Propagate vocab_size into text config
            self.textConfig.vocabSize = self.vocabSize
            // Propagate the top-level quantization_config into the text config
            // when the text_config didn't carry its own.
            if self.textConfig.quantizationConfig == nil {
                self.textConfig.quantizationConfig = self.quantizationConfig
            }
        } else {
            self.textConfig = try Gemma4TextConfiguration(from: decoder)
            if self.quantizationConfig == nil {
                self.quantizationConfig = self.textConfig.quantizationConfig
            }
        }
    }
}

// MARK: - Model

public class Gemma4Model: Module, LLMModel, KVCacheDimensionProvider {
    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    /// The text model, exposed at `@_spi(GemmaEncoder)` scope.
    ///
    /// This is the type the model factory actually produces for `gemma4_unified`, so an
    /// encoder-style client tap cannot reach `Gemma4TextModel` without it — exposing only
    /// the inner types is not sufficient for a real load path.
    @ModuleInfo(key: "language_model") @_spi(GemmaEncoder) public var languageModel: Gemma4TextModel
    fileprivate let quantizationConfig: GemmaMobileQuantizationConfig?

    public init(_ config: Gemma4Configuration) {
        self.quantizationConfig = config.quantizationConfig
        self._languageModel.wrappedValue = Gemma4TextModel(config.textConfig)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitize(weights: weights, metadata: [:])
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String])
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            var k = key

            // Strip "model." prefix
            let startsWithModel = k.hasPrefix("model.")
            k = k.replacingOccurrences(of: "model.", with: "", options: .anchored)

            // Skip vision/audio weights. `vision_embedder` is the gemma4_unified
            // (12B) encoder-free vision module; without it, loading a multimodal
            // gemma4_unified checkpoint (e.g. mlx-community/gemma-4-12B-it-4bit)
            // through the text path fails with `Unhandled keys ["vision_embedder"]`.
            if k.hasPrefix("vision_tower") || k.hasPrefix("multi_modal_projector")
                || k.hasPrefix("audio_tower") || k.hasPrefix("embed_audio")
                || k.hasPrefix("embed_vision") || k.hasPrefix("vision_embedder")
            {
                continue
            }

            if !startsWithModel {
                sanitized[k] = value
                continue
            }

            // Remap language_model keys
            if k.hasPrefix("language_model") {
                k = k.replacingOccurrences(
                    of: "language_model.", with: "language_model.model.", options: .anchored)
            }

            sanitized[k] = value
        }

        // MoE expert remap (single-arg sanitize deliberately skips the text
        // model's mobile path — the wrapper drives it on the full
        // `language_model.*` namespace below so the leaf-module paths match the
        // post-sanitize weight keys).
        sanitized = languageModel.sanitize(weights: sanitized)

        if let qc = quantizationConfig, qc.isGemmaMobile {
            sanitized = applyGemmaMobileQuantization(model: self, weights: sanitized, config: qc)
        }
        return sanitized
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }
}

// MARK: - Load-time precompilation (Phase 5)

extension Gemma4Model: NativePrecompilable {
    /// Precompile native compiled functions and free mobile-format weights.
    ///
    /// Forwards to the wrapped `Gemma4TextModel`, which honors
    /// `Gemma4TextModel.precompileAtLoad`. Called by `loadWeights` after weights
    /// are loaded and modules are replaced.
    public func precompileNativeFunctions() {
        languageModel.precompileNativeFunctions()
    }
}

// MARK: - LoRA

extension Gemma4Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.loraLayers
    }
}

// MARK: - Chat conventions

extension Gemma4Model {
    public var toolCallFormat: ToolCallFormat? { .gemma4 }
}
