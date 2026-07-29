// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXVLM

/// Configuration and construction regressions for the non-E-series Gemma 4
/// families. Exact public dimensions belong in fixtures only; production
/// dispatch is driven by the decoded topology fields exercised below.
struct Gemma4ModelFamilyTests {
    @Test("Public E4B topology decodes without E2B assumptions")
    func e4BConfiguration() throws {
        let layerTypes = (0 ..< 42).map {
            ($0 + 1).isMultiple(of: 6) ? "\"full_attention\"" : "\"sliding_attention\""
        }.joined(separator: ",")
        let json =
            """
            {
              "model_type": "gemma4_text",
              "hidden_size": 2560,
              "num_hidden_layers": 42,
              "intermediate_size": 10240,
              "num_attention_heads": 8,
              "num_key_value_heads": 2,
              "head_dim": 256,
              "global_head_dim": 512,
              "vocab_size": 262144,
              "vocab_size_per_layer_input": 262144,
              "num_kv_shared_layers": 18,
              "hidden_size_per_layer_input": 256,
              "sliding_window": 512,
              "max_position_embeddings": 131072,
              "rms_norm_eps": 1e-6,
              "attention_k_eq_v": false,
              "use_double_wide_mlp": false,
              "enable_moe_block": false,
              "layer_types": [\(layerTypes)],
              "tie_word_embeddings": false
            }
            """
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))

        #expect(config.hiddenSize == 2560)
        #expect(config.hiddenLayers == 42)
        #expect(config.intermediateSize == 10_240)
        #expect(config.kvHeads == 2)
        #expect(config.numKVSharedLayers == 18)
        #expect(config.hiddenSizePerLayerInput == 256)
        #expect(!config.attentionKEqV)
        #expect(!config.enableMoEBlock)
        #expect(!config.useDoubleWideMLP)
        #expect(!config.tieWordEmbeddings)
        #expect(
            config.layerTypes.enumerated().compactMap { index, type in
                type == "full_attention" ? index : nil
            } == [5, 11, 17, 23, 29, 35, 41])
    }

    @Test("Public 31B dense topology decodes without E-series assumptions")
    func dense31BConfiguration() throws {
        let config = try largeTextConfiguration(
            hiddenSize: 5376,
            hiddenLayers: 60,
            intermediateSize: 21_504,
            attentionHeads: 32,
            kvHeads: 16,
            globalKVHeads: 4,
            enableMoE: false,
            numExperts: nil,
            topKExperts: nil,
            moeIntermediateSize: nil)

        #expect(config.hiddenSize == 5376)
        #expect(config.hiddenLayers == 60)
        #expect(config.intermediateSize == 21_504)
        #expect(config.attentionHeads == 32)
        #expect(config.kvHeads == 16)
        #expect(config.globalKVHeads == 4)
        #expect(config.numKVSharedLayers == 0)
        #expect(config.hiddenSizePerLayerInput == 0)
        #expect(config.attentionKEqV)
        #expect(!config.enableMoEBlock)
        #expect(!config.useDoubleWideMLP)
        #expect(config.layerTypes.count == 60)
        #expect(config.layerTypes[5] == "full_attention")
    }

    @Test("Public 26B-A4B MoE topology decodes without dense assumptions")
    func moe26BA4BConfiguration() throws {
        let config = try largeTextConfiguration(
            hiddenSize: 2816,
            hiddenLayers: 30,
            intermediateSize: 2112,
            attentionHeads: 16,
            kvHeads: 8,
            globalKVHeads: 2,
            enableMoE: true,
            numExperts: 128,
            topKExperts: 8,
            moeIntermediateSize: 704)

        #expect(config.hiddenSize == 2816)
        #expect(config.hiddenLayers == 30)
        #expect(config.intermediateSize == 2112)
        #expect(config.attentionHeads == 16)
        #expect(config.kvHeads == 8)
        #expect(config.globalKVHeads == 2)
        #expect(config.numKVSharedLayers == 0)
        #expect(config.hiddenSizePerLayerInput == 0)
        #expect(config.attentionKEqV)
        #expect(config.enableMoEBlock)
        #expect(config.numExperts == 128)
        #expect(config.topKExperts == 8)
        #expect(config.moeIntermediateSize == 704)
        #expect(config.layerTypes.count == 30)
        #expect(config.layerTypes[5] == "full_attention")
    }

    @Test("Dense and MoE modules are selected from capabilities, not model dimensions")
    func topologyDrivesModuleConstructionAndForward() throws {
        let denseConfig = try tinyLargeFamilyConfiguration(enableMoE: false)
        let dense = Gemma4TextLanguageModel(denseConfig)
        let denseLayers = dense.model.layers

        #expect(denseLayers.count == 6)
        #expect(dense.model.embedTokensPerLayer == nil)
        #expect(denseLayers.allSatisfy { !$0.enableMoE && $0.router == nil && $0.experts == nil })
        #expect(!denseLayers[0].selfAttention.useKEqV)
        #expect(denseLayers[0].selfAttention.vProj != nil)
        #expect(denseLayers[5].selfAttention.useKEqV)
        #expect(denseLayers[5].selfAttention.vProj == nil)

        let moeConfig = try tinyLargeFamilyConfiguration(enableMoE: true)
        let moe = Gemma4TextLanguageModel(moeConfig)
        let moeLayers = moe.model.layers

        #expect(moeLayers.count == 6)
        #expect(moe.model.embedTokensPerLayer == nil)
        #expect(moeLayers.allSatisfy { $0.enableMoE && $0.router != nil && $0.experts != nil })
        #expect(moeLayers[5].selfAttention.useKEqV)
        #expect(moeLayers[5].selfAttention.vProj == nil)

        let tokens = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let denseLogits = dense(tokens).logits
        let moeLogits = moe(tokens).logits
        eval([denseLogits, moeLogits])

        #expect(denseLogits.shape == [1, 3, 64])
        #expect(moeLogits.shape == [1, 3, 64])
        #expect(denseLogits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(moeLogits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    private func largeTextConfiguration(
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        globalKVHeads: Int,
        enableMoE: Bool,
        numExperts: Int?,
        topKExperts: Int?,
        moeIntermediateSize: Int?
    ) throws -> Gemma4TextConfiguration {
        let layerTypes = (0 ..< hiddenLayers).map {
            ($0 + 1).isMultiple(of: 6) ? "\"full_attention\"" : "\"sliding_attention\""
        }.joined(separator: ",")
        let optional: (Int?) -> String = { $0.map(String.init) ?? "null" }
        let json =
            """
            {
              "model_type": "gemma4_text",
              "hidden_size": \(hiddenSize),
              "num_hidden_layers": \(hiddenLayers),
              "intermediate_size": \(intermediateSize),
              "num_attention_heads": \(attentionHeads),
              "num_key_value_heads": \(kvHeads),
              "num_global_key_value_heads": \(globalKVHeads),
              "head_dim": 256,
              "global_head_dim": 512,
              "vocab_size": 262144,
              "vocab_size_per_layer_input": 262144,
              "num_kv_shared_layers": 0,
              "hidden_size_per_layer_input": 0,
              "sliding_window": 1024,
              "max_position_embeddings": 262144,
              "rms_norm_eps": 1e-6,
              "attention_k_eq_v": true,
              "use_double_wide_mlp": false,
              "enable_moe_block": \(enableMoE),
              "num_experts": \(optional(numExperts)),
              "top_k_experts": \(optional(topKExperts)),
              "moe_intermediate_size": \(optional(moeIntermediateSize)),
              "layer_types": [\(layerTypes)],
              "tie_word_embeddings": true
            }
            """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyLargeFamilyConfiguration(enableMoE: Bool) throws
        -> Gemma4TextConfiguration
    {
        let moeFields =
            enableMoE
            ? """
              "num_experts": 4,
              "top_k_experts": 2,
              "moe_intermediate_size": 16,
            """
            : ""
        let json =
            """
            {
              "model_type": "gemma4_text",
              "hidden_size": 32,
              "num_hidden_layers": 6,
              "intermediate_size": 64,
              "num_attention_heads": 2,
              "num_key_value_heads": 2,
              "num_global_key_value_heads": 1,
              "head_dim": 16,
              "global_head_dim": 16,
              "vocab_size": 64,
              "vocab_size_per_layer_input": 64,
              "num_kv_shared_layers": 0,
              "hidden_size_per_layer_input": 0,
              "sliding_window": 32,
              "max_position_embeddings": 64,
              "rms_norm_eps": 1e-6,
              "attention_k_eq_v": true,
              "use_double_wide_mlp": false,
              "enable_moe_block": \(enableMoE),
              \(moeFields)
              "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
              ],
              "tie_word_embeddings": true
            }
            """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }
}
