// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

private struct Gemma4UnifiedTestTokenizer: Tokenizer {
    let vocabularySize: Int = 64
    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = nil
    let unknownTokenId: Int? = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [31, 2]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [31, 2]
    }
}

struct Gemma4UnifiedTests {
    private func decodeConfig(_ json: String) throws -> Gemma4UnifiedConfiguration {
        try JSONDecoder.json5().decode(Gemma4UnifiedConfiguration.self, from: Data(json.utf8))
    }

    private func tinyTextConfig() throws -> Gemma4UnifiedConfiguration {
        try decodeConfig(
            """
            {
              "model_type": "gemma4_unified",
              "vocab_size": 32,
              "image_token_id": 31,
              "audio_token_id": 30,
              "video_token_id": 29,
              "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "head_dim": 8,
                "global_head_dim": 8,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 8,
                "sliding_window_pattern": 1,
                "attention_k_eq_v": true,
                "use_double_wide_mlp": false,
                "layer_types": ["full_attention"],
                "tie_word_embeddings": true
              },
              "vision_config": null,
              "audio_config": null
            }
            """)
    }

    @Test("Gemma4 Unified config decodes unified defaults and eoa_token_index")
    func configDecoding() throws {
        let config = try decodeConfig(
            """
            {
              "model_type": "gemma4_unified",
              "eoa_token_index": 258883,
              "text_config": { "model_type": "gemma4_unified_text" },
              "vision_config": { "model_type": "gemma4_unified_vision" },
              "audio_config": { "model_type": "gemma4_unified_audio" }
            }
            """)

        #expect(config.modelType == "gemma4_unified")
        #expect(config.eoaTokenId == 258883)
        #expect(config.textConfiguration.modelType == "gemma4_unified_text")
        #expect(config.textConfiguration.hiddenSize == 3840)
        #expect(config.textConfiguration.numKVSharedLayers == 0)
        #expect(config.textConfiguration.hiddenSizePerLayerInput == 0)
        #expect(config.textConfiguration.attentionKEqV)
        #expect(config.textConfiguration.useBidirectionalAttention == "vision")
        #expect(config.visionConfiguration?.modelPatchSize == 48)
    }

    @Test("Public 12B QAT config decodes as unified Gemma 4")
    func public12BQATConfigCompatibility() throws {
        // Architecture-critical fields from mlx-community/gemma-4-12B-it-qat-4bit.
        let json =
            """
            {
              "model_type": "gemma4_unified",
              "image_token_id": 258880,
              "audio_token_id": 258881,
              "video_token_id": 258884,
              "boi_token_id": 255999,
              "eoi_token_id": 258882,
              "eoa_token_index": 258883,
              "tie_word_embeddings": true,
              "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.0.mlp.gate_proj": {
                  "group_size": 64,
                  "bits": 8
                }
              },
              "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 3840,
                "num_hidden_layers": 48,
                "intermediate_size": 15360,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 0,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 1024,
                "max_position_embeddings": 262144,
                "final_logit_softcapping": 30.0,
                "attention_k_eq_v": true,
                "use_bidirectional_attention": "vision",
                "rope_parameters": {
                  "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                  },
                  "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                  }
                },
                "use_double_wide_mlp": false,
                "enable_moe_block": false,
                "tie_word_embeddings": true
              },
              "vision_config": {
                "model_type": "gemma4_unified_vision",
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "mm_embed_dim": 3840,
                "mm_posemb_size": 1120,
                "output_proj_dims": 3840
              },
              "audio_config": {
                "model_type": "gemma4_unified_audio",
                "audio_embed_dim": 640,
                "rms_norm_eps": 0.000001
              }
            }
            """
        let config = try decodeConfig(json)
        let baseConfig = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: Data(json.utf8))

        let text = config.textConfiguration
        #expect(config.modelType == "gemma4_unified")
        #expect(config.quantization?.groupSize == 64)
        #expect(config.quantization?.bits == 4)
        #expect(config.quantization?.mode == .affine)
        #expect(
            baseConfig.perLayerQuantization?.quantization(
                layer: "language_model.model.layers.0.mlp.gate_proj")?.bits == 8)
        #expect(
            baseConfig.perLayerQuantization?.quantization(
                layer: "language_model.model.layers.0.self_attn.q_proj")?.bits == 4)
        #expect(config.tieWordEmbeddings && text.tieWordEmbeddings)
        #expect(text.modelType == "gemma4_unified_text")
        #expect(text.hiddenSize == 3840)
        #expect(text.hiddenLayers == 48)
        #expect(text.intermediateSize == 15_360)
        #expect(text.attentionHeads == 16)
        #expect(text.kvHeads == 8)
        #expect(text.globalKVHeads == 1)
        #expect(text.attentionKEqV)
        #expect(text.numKVSharedLayers == 0)
        #expect(text.hiddenSizePerLayerInput == 0)
        #expect(text.maxPositionEmbeddings == 262_144)
        #expect(text.finalLogitSoftcapping == 30.0)
        #expect(text.useBidirectionalAttention == "vision")
        #expect(
            text.ropeParameters["full_attention"]?["partial_rotary_factor"]?.asFloat() == 0.25)
        #expect(!text.useDoubleWideMLP)
        #expect(!text.enableMoEBlock)
        #expect(text.layerTypes.count == 48)
        #expect(
            text.layerTypes.enumerated().compactMap { index, type in
                type == "full_attention" ? index : nil
            } == [5, 11, 17, 23, 29, 35, 41, 47])
    }

    @Test("Unified topology is driven by K=V, PLE, and MoE configuration")
    func unifiedUsesConfiguredTopology() throws {
        let model = Gemma4Unified(try tinyTextConfig())
        let layer = try #require(
            model.loraLayers.compactMap { $0 as? Gemma4TextDecoderLayer }.first)

        #expect(layer.selfAttention.useKEqV)
        #expect(layer.selfAttention.vProj == nil)
        #expect(!layer.enableMoE)
        #expect(layer.router == nil)
        #expect(layer.experts == nil)
        #expect(layer.perLayerInputGate == nil)
        #expect(layer.perLayerProjection == nil)
    }

    @Test("Gemma4 Unified text-only prepare chunks prefill")
    func textOnlyPrepareChunksPrefill() throws {
        let model = Gemma4Unified(try tinyTextConfig())
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray([0, 2, 3, 4, 5, 1]).expandedDimensions(axis: 0))

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: .init(stepSize: 2))

        guard case .logits(let output) = result else {
            Issue.record("Expected text-only Gemma4Unified.prepare to return logits")
            return
        }

        #expect(output.logits.shape == [1, 1, 32])
        #expect(cache.allSatisfy { $0.offset == 6 })
    }

    @Test("Gemma4 Unified text-only prefill is chunk-size invariant")
    func textOnlyPrefillChunkSizeInvariant() throws {
        // Pin the initializer weights. Task-local rather than MLXRandom.seed:
        // tests in this suite run concurrently and share the global RNG.
        let config = try tinyTextConfig()
        let model = withRandomState(MLXRandom.RandomState(seed: 42)) {
            Gemma4Unified(config)
        }
        // Run in f16, as real models do.
        model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float16) : $0 }
        let input = LMInput(tokens: MLXArray([0, 2, 3, 4, 5, 1]).expandedDimensions(axis: 0))

        func prefill(windowSize: Int) throws -> (logits: MLXArray, cacheOffsets: [Int]) {
            let cache = try model.newCache(parameters: nil)
            let result = try model.prepare(
                input, cache: cache, state: nil, prefill: .init(stepSize: windowSize))
            guard case .logits(let output) = result else {
                Issue.record("Expected text-only Gemma4Unified.prepare to return logits")
                return (MLXArray.zeros([1, 32]), [])
            }
            return (output.logits[0..., -1, 0...], cache.map { $0.offset })
        }

        let full = try prefill(windowSize: 16)
        let chunked = try prefill(windowSize: 2)
        #expect(full.cacheOffsets == chunked.cacheOffsets)
        #expect(chunked.cacheOffsets.allSatisfy { $0 == 6 })

        let diff = abs(full.logits.asType(.float32) - chunked.logits.asType(.float32)).max()
        eval(diff)

        #expect(
            diff.item(Float.self) < 1e-4,
            "Chunked and single-pass prefill logits differ by \(diff.item(Float.self)).")
    }

    @Test("Gemma4 Unified processor emits model patches and position ids")
    func processorPatchifiesImages() async throws {
        let data = Data(
            """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "image_token_id": 31,
              "boi_token_id": 28,
              "eoi_token_id": 29,
              "image_processor": {
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "model_patch_size": 4,
                "max_soft_tokens": 4,
                "size": { "height": 8, "width": 8 }
              }
            }
            """.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4UnifiedProcessorConfiguration.self, from: data)
        let processor = Gemma4UnifiedProcessor(config, tokenizer: Gemma4UnifiedTestTokenizer())
        let image = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", images: [.ciImage(image)]))

        #expect(input.image?.pixels.shape == [1, 4, 48])
        #expect(input.image?.positionIds?.shape == [1, 4, 2])
        #expect(input.text.tokens.asArray(Int32.self) == [28, 31, 31, 31, 31, 29, 2])
    }

    @Test("Gemma4 Unified processor resizes images into max soft-token budget")
    func processorResizesImagesIntoSoftTokenBudget() throws {
        let data = Data(
            """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "image_token_id": 31,
              "boi_token_id": 28,
              "eoi_token_id": 29,
              "image_processor": {
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "model_patch_size": 48,
                "max_soft_tokens": 280,
                "do_resize": true,
                "do_rescale": true,
                "rescale_factor": 0.00392156862745098,
                "size": { "height": 224, "width": 224 }
              }
            }
            """.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4UnifiedProcessorConfiguration.self, from: data)
        let processor = Gemma4UnifiedProcessor(config, tokenizer: Gemma4UnifiedTestTokenizer())

        let targetSize = try config.aspectRatioPreservingSize(
            for: CGSize(width: 2174, height: 1356))
        #expect(targetSize == CGSize(width: 1008, height: 624))
        #expect(config.fixedSize == CGSize(width: 768, height: 768))
        #expect(config.doResize)
        #expect(config.doRescale)
        #expect(abs(config.rescaleFactor - (1.0 / 255.0)) < 1e-12)

        let image = CIImage(color: .black).cropped(
            to: CGRect(x: 0, y: 0, width: 2174, height: 1356))
        let imageData = try processor.preprocess(images: [image], processing: nil)

        #expect(imageData.pixels.shape == [1, 280, 6912])
        #expect(imageData.positionIds.shape == [1, 280, 2])
        #expect(imageData.tokenCounts == [273])

        let frame = try #require(imageData.frames.first)
        #expect(frame.t == 1)
        #expect(frame.h == 624)
        #expect(frame.w == 1008)
    }

    @Test("Gemma4 Unified model accepts vision embeddings")
    func modelVisionForward() throws {
        let config = try decodeConfig(
            """
            {
              "model_type": "gemma4_unified",
              "vocab_size": 32,
              "image_token_id": 31,
              "audio_token_id": 30,
              "video_token_id": 29,
              "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 1,
                "head_dim": 8,
                "global_head_dim": 8,
                "vocab_size": 32,
                "vocab_size_per_layer_input": 32,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 8,
                "sliding_window_pattern": 1,
                "attention_k_eq_v": true,
                "use_double_wide_mlp": false,
                "layer_types": ["full_attention"],
                "tie_word_embeddings": true
              },
              "vision_config": {
                "model_type": "gemma4_unified_vision",
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "model_patch_size": 4,
                "mm_embed_dim": 8,
                "mm_posemb_size": 4,
                "num_soft_tokens": 4,
                "output_proj_dims": 8
              },
              "audio_config": null
            }
            """)
        let model = Gemma4Unified(config)
        let inputIds = MLXArray([0, 31, 31, 31, 31, 1]).reshaped(1, 6)
        let pixelValues = MLXArray.zeros([1, 4, 48], dtype: .float32)
        let positionIds = MLXArray.zeros([1, 4, 2], dtype: .int32)
        let input = LMInput(
            text: .init(tokens: inputIds),
            image: .init(pixels: pixelValues, positionIds: positionIds)
        )

        let result = try model.prepare(
            input, cache: try model.newCache(parameters: nil), state: nil, prefill: .init())

        guard case .logits(let output) = result else {
            Issue.record("Expected Gemma4Unified.prepare to return logits")
            return
        }
        #expect(output.logits.shape == [1, 6, 32])
    }
}
