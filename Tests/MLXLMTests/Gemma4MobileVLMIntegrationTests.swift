// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

import MLXVLM

/// End-to-end coverage for the Gemma 4 QAT mobile (wNa8o8) load path through the
/// **VLM** model (`MLXVLM.Gemma4`, registered for `model_type: "gemma4"`). The
/// model loading flow tries the VLM registry before the LLM registry, so the
/// multimodal `gemma-4-E2B-it-qat-mobile-mlx-mm` checkpoint loads here. This
/// test verifies that the Gemma mobile quantization is applied to both the text
/// model and the vision tower (including `Linear`s nested inside
/// `Gemma4ClippableLinear`), and that a text-only forward pass produces finite
/// logits (see GEMMA4_QAT_MOBILE_SWIFT_PORT_PLAN.md §5.11).
struct Gemma4MobileVLMIntegrationTests {

    /// Path to the real `gemma-4-E2B-it-qat-mobile-mlx-mm` checkpoint. This is a
    /// local path and may not exist in all environments; the test skips if the
    /// checkpoint is absent.
    private static let realCheckpointURL = URL(filePath: "/Users/adrgrondin/Workspace/mlx-vlm/gemma-4-E2B-it-qat-mobile-mlx-mm")

    @Test("Real gemma-4-E2B-it-qat-mobile-mlx-mm loads through the VLM path with quantized layers")
    func loadsRealMobileCheckpointThroughVLM() throws {
        let dir = Self.realCheckpointURL
        guard FileManager.default.fileExists(atPath: dir.appending(component: "config.json").path) else {
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
        #expect(values.allSatisfy { $0.isFinite }, "logits must be finite, got first few: \(values.prefix(10))")
    }
}
