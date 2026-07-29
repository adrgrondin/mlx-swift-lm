// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// End-to-end coverage for the Gemma 4 QAT mobile (wNa8o8) load path through the
/// **VLM** model (`MLXVLM.Gemma4`, registered for `model_type: "gemma4"`). The
/// model loading flow tries the VLM registry before the LLM registry, so the
/// multimodal `gemma-4-E2B-it-qat-mobile-mlx-mm` checkpoint loads here. This
/// test verifies that the Gemma mobile quantization is applied to both the text
/// model and the vision tower (including `Linear`s nested inside
/// `Gemma4ClippableLinear`), and that a text-only forward pass produces finite
/// logits (see GEMMA4_QAT_MOBILE_SWIFT_PORT_PLAN.md §5.11).
@Suite(.serialized)
struct Gemma4MobileVLMIntegrationTests {

    /// Path to the real `gemma-4-E2B-it-qat-mobile-mlx-mm` checkpoint. This is a
    /// local path and may not exist in all environments; the test skips if the
    /// checkpoint is absent.
    private static let realCheckpointURL = URL(
        filePath: "/Users/adrgrondin/Workspace/mlx-vlm/gemma-4-E2B-it-qat-mobile-mlx-mm")

    private static var cachedStandardE2BURL: URL? {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appending(path: ".cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-4bit")
        let referenceURL = root.appending(path: "refs/main")
        guard
            let reference = try? String(contentsOf: referenceURL, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !reference.isEmpty
        else { return nil }
        return root.appending(path: "snapshots/\(reference)")
    }

    @Test("Real gemma-4-E2B-it-qat-mobile-mlx-mm loads through the VLM path with quantized layers")
    func loadsRealMobileCheckpointThroughVLM() throws {
        let dir = Self.realCheckpointURL
        guard FileManager.default.fileExists(atPath: dir.appending(component: "config.json").path)
        else {
            // Checkpoint not available — skip this test.
            return
        }

        let configData = try Data(contentsOf: dir.appending(component: "config.json"))
        let decoder = JSONDecoder.json5()
        decoder.userInfo[.rawConfigData] = configData
        let config = try decoder.decode(Gemma4Configuration.self, from: configData)
        #expect(config.quantizationConfig?.isGemmaMobile == true)

        let model = Gemma4(config)
        try loadWeights(modelDirectory: dir, model: model)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())

        // Text model: lm_head → 2-bit GemmaQuantizedLinear.
        let lmHead = try #require(modules["language_model.lm_head"] as? GemmaQuantizedLinear)
        #expect(lmHead.numBits == 2)

        // Text model: embed_tokens → 2-bit per-row GemmaQuantizedEmbedding.
        let embed = try #require(
            modules["language_model.model.embed_tokens"] as? GemmaQuantizedEmbedding)
        #expect(embed.numBits == 2)
        #expect(embed.numBlocks == 1)

        // Text model: layer 0 attention → 4-bit.
        let qProj = try #require(
            modules["language_model.model.layers.0.self_attn.q_proj"] as? GemmaQuantizedLinear)
        #expect(qProj.numBits == 4)

        // Text model: layer 0 mlp → 4-bit (layers 0–14).
        let gate0 = try #require(
            modules["language_model.model.layers.0.mlp.gate_proj"] as? GemmaQuantizedLinear)
        #expect(gate0.numBits == 4)

        // Text model: layer 15 mlp → 2-bit (layers 15–34).
        let gate15 = try #require(
            modules["language_model.model.layers.15.mlp.gate_proj"] as? GemmaQuantizedLinear)
        #expect(gate15.numBits == 2)

        // Vision tower: the inner Linear inside Gemma4ClippableLinear is
        // replaced with an 8-bit GemmaQuantizedLinear (vision_tower → 8 bits).
        let vProjLinear = try #require(
            modules["vision_tower.encoder.layers.0.self_attn.v_proj.linear"]
                as? GemmaQuantizedLinear)
        #expect(vProjLinear.numBits == 8)

        // Forward pass (text-only) produces finite logits of the right shape.
        let cache = model.newCache(parameters: nil)
        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let logits = model(tokens, cache: cache)
        eval(logits)
        #expect(logits.shape == [1, 3, config.textConfiguration.vocabularySize])
        let values = logits.asType(.float32).asArray(Float.self)
        #expect(
            values.allSatisfy { $0.isFinite },
            "logits must be finite, got first few: \(values.prefix(10))")
    }

    @Test("Standard mlx-community E2B 4-bit stays on ordinary affine quantization")
    func standardE2BUsesOrdinaryAffineQuantization() throws {
        guard let dir = Self.cachedStandardE2BURL,
            FileManager.default.fileExists(atPath: dir.appending(path: "config.json").path)
        else { return }

        let configData = try Data(contentsOf: dir.appending(path: "config.json"))
        let decoder = JSONDecoder.json5()
        decoder.userInfo[.rawConfigData] = configData
        let config = try decoder.decode(Gemma4Configuration.self, from: configData)
        let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)

        #expect(config.quantizationConfig?.isGemmaMobile != true)
        #expect(config.textConfiguration.tieWordEmbeddings)

        let model = Gemma4(config)
        #expect(model.languageModel.lmHead == nil)

        try loadWeights(
            modelDirectory: dir,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)
        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        #expect(
            modules["language_model.model.embed_tokens"] is QuantizedEmbedding)
        #expect(
            modules["language_model.model.layers.0.self_attn.q_proj"] is QuantizedLinear)
    }

    @Test("Official E4B mobile config uses the same format-driven path")
    func e4bMobileConfigIsFormatDriven() throws {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appending(
                path:
                    ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-mobile-transformers")
        let referenceURL = root.appending(path: "refs/main")
        guard
            let reference = try? String(contentsOf: referenceURL, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !reference.isEmpty
        else { return }

        let configURL = root.appending(path: "snapshots/\(reference)/config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else { return }

        let data = try Data(contentsOf: configURL)
        let decoder = JSONDecoder.json5()
        decoder.userInfo[.rawConfigData] = data
        let config = try decoder.decode(Gemma4Configuration.self, from: data)
        let text = config.textConfiguration
        let quantization = try #require(config.quantizationConfig)

        #expect(quantization.isGemmaMobile)
        #expect(text.hiddenLayers == 42)
        #expect(text.numKVSharedLayers == 18)
        #expect(text.hiddenSizePerLayerInput == 256)
        #expect(!text.attentionKEqV)
        #expect(!text.enableMoEBlock)
        #expect(!text.tieWordEmbeddings)
        #expect(text.layerTypes.count == text.hiddenLayers)
        #expect(
            resolveModuleBits(
                path: "language_model.model.layers.41.mlp.gate_proj",
                config: quantization) == 4)
        #expect(
            resolveModuleBits(
                path: "language_model.model.layers.41.self_attn.q_proj",
                config: quantization) == 4)
    }

    @Test("preprocessor_config.json without processor_class falls back to processor_config.json")
    func processorConfigFallback() throws {
        let dir = Self.realCheckpointURL
        guard FileManager.default.fileExists(atPath: dir.appending(component: "config.json").path)
        else {
            return  // Checkpoint not available — skip.
        }

        // The checkpoint ships an audio feature-extractor config as
        // preprocessor_config.json (no `processor_class`) and the actual
        // processor config (with `processor_class: Gemma4Processor`) as
        // processor_config.json. `BaseProcessorConfiguration` must tolerate the
        // missing field so the factory can fall back.
        let pre = try Data(contentsOf: dir.appending(component: "preprocessor_config.json"))
        let preCfg = try JSONDecoder.json5().decode(BaseProcessorConfiguration.self, from: pre)
        #expect(preCfg.processorClass == nil)

        let proc = try Data(contentsOf: dir.appending(component: "processor_config.json"))
        let procCfg = try JSONDecoder.json5().decode(BaseProcessorConfiguration.self, from: proc)
        #expect(procCfg.processorClass == "Gemma4Processor")
    }
}
