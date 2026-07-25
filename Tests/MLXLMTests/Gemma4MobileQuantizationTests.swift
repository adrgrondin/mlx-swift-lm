// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Unit tests for the Gemma 4 QAT mobile (wNa8o8) quantization primitives ported
/// from `mlx_vlm/quantization/gemma_mobile.py`. They verify the unpacking,
/// SRQ, dequantization, and per-layer bit resolution are bit-exact against the
/// Python reference (see GEMMA4_QAT_MOBILE_SWIFT_PORT_PLAN.md §5.10).
struct Gemma4MobileQuantizationTests {

    // MARK: - Unpacking

    @Test("unpackInt2 matches the LSB-first uint8 layout")
    func unpackInt2() throws {
        // 0b11_10_01_00 → bits [1:0]=00, [3:2]=01, [5:4]=10, [7:6]=11 → [-2, -1, 0, 1]
        let packed = MLXArray([UInt8(0b1110_0100)], [1])
        let out = MLXLMCommon.unpackInt2(packed, inFeatures: 4)
        eval(out)
        #expect(out.shape == [4])
        #expect(out.asArray(Int8.self) == [-2, -1, 0, 1])
    }

    @Test("unpackInt2 unpacks a 2D weight and trims to in_features")
    func unpackInt2Weight() throws {
        // Two bytes → 8 values; request only 6 (trims the trailing 2).
        let packed = MLXArray([UInt8(0b1110_0100), UInt8(0b0000_0000)], [1, 2])
        let out = MLXLMCommon.unpackInt2(packed, inFeatures: 6)
        eval(out)
        #expect(out.shape == [1, 6])
        #expect(out.asArray(Int8.self) == [-2, -1, 0, 1, -2, -2])
    }

    @Test("unpackInt4 matches the low-nibble-first uint8 layout")
    func unpackInt4() throws {
        // 0b1000_0111 → low nibble 0x7=7→-1, high nibble 0x8=8→0 → [-1, 0]
        let packed = MLXArray([UInt8(0b1000_0111)], [1])
        let out = MLXLMCommon.unpackInt4(packed, inFeatures: 2)
        eval(out)
        #expect(out.shape == [2])
        #expect(out.asArray(Int8.self) == [-1, 0])
    }

    @Test("unpackInt8 returns int8 directly")
    func unpackInt8() throws {
        let packed = MLXArray([Int8(-3), Int8(5), Int8(127)], [3])
        let out = unpackInt(packed, numBits: 8, inFeatures: 3)
        eval(out)
        #expect(out.asArray(Int8.self) == [-3, 5, 127])
    }

    // MARK: - SRQ

    @Test("applySRQ is a no-op when the scale is zero (uncalibrated)")
    func applySRQUncalibrated() throws {
        let x = MLXArray([Float(0.1), Float(0.4), Float(-1.0)], [3])
        let out = applySRQ(x, scale: MLXArray(Float(0.0)))
        eval(out)
        #expect(out.asArray(Float.self) == [Float(0.1), Float(0.4), Float(-1.0)])
    }

    @Test("applySRQ fake-quantizes to int8 and back with clipping")
    func applySRQCalibrated() throws {
        let x = MLXArray([Float(0.1), Float(0.4), Float(20.0), Float(-20.0)], [4])
        let out = applySRQ(x, scale: MLXArray(Float(0.1)))
        eval(out)
        let vals = out.asArray(Float.self)
        // round(x / 0.1) clipped to [-128, 127] then * 0.1
        Self.assertApproximatelyEqual(vals, [0.1, 0.4, 12.7, -12.8], tolerance: 1e-5)
    }

    // MARK: - Dequantization

    @Test("dequantizeWeight unpacks and scales per-channel")
    func dequantizeWeightInt4() throws {
        // int4: one row, two values packed in one byte 0x87 → [-1, 0]
        let weight = MLXArray([UInt8(0x87)], [1, 1])
        let scale = MLXArray([Float(0.5)], [1, 1])
        let out = dequantizeWeight(weight, weightScale: scale, numBits: 4, inputDims: 2)
        eval(out)
        #expect(out.shape == [1, 2])
        Self.assertApproximatelyEqual(out.asArray(Float.self), [-0.5, 0.0], tolerance: 1e-6)
    }

    @Test("dequantizeEmbeddingRows handles block-wise scales")
    func dequantizeEmbeddingRowsBlockWise() throws {
        // 1 row, 4 values, int4 packed in 2 bytes: 0x87 → [-1, 0], 0x77 → [-1, -1]
        let rows = MLXArray([UInt8(0x87), UInt8(0x77)], [1, 2])
        // 2 blocks of size 2: scale block 0 = 1.0, block 1 = 0.5
        let scales = MLXArray([Float(1.0), Float(0.5)], [1, 2])
        let out = dequantizeEmbeddingRows(
            rows, scales: scales, numBits: 4, embeddingDim: 4, numBlocks: 2)
        eval(out)
        #expect(out.shape == [1, 4])
        // block 0: [-1, 0] * 1.0 = [-1, 0]; block 1: [-1, -1] * 0.5 = [-0.5, -0.5]
        Self.assertApproximatelyEqual(
            out.asArray(Float.self), [-1.0, 0.0, -0.5, -0.5], tolerance: 1e-6)
    }

    // MARK: - Per-layer bit resolution

    /// The E2B mobile `module_quant_configs` schema (regex patterns in the
    /// HuggingFace `language_model.*` namespace), built programmatically so the
    /// regex backslashes are plain Swift escapes.
    private func e2bMobileConfig() -> GemmaMobileQuantizationConfig {
        var config = GemmaMobileQuantizationConfig()
        config.quantMethod = "gemma"
        config.numBits = 4
        config.quantizeEmbeddings = true
        config.modulesToNotConvert = [
            "model.vision_tower.patch_embedder",
            "model.audio_tower.subsample_conv_projection",
            "model.audio_tower.output_proj",
            "relative_k_proj",
            "model.embed_audio",
            "model.embed_vision",
            "per_layer_model_projection",
        ]
        // First-match-wins order (matches the published config.json).
        config.moduleQuantConfigs = [
            ("^lm_head$", 2),
            ("language_model\\.embed_tokens$", 2),
            ("language_model\\.embed_tokens_per_layer$", 4),
            ("language_model\\.layers\\.(\\d|1[0-4])\\.mlp\\.", 4),
            ("language_model\\.layers\\.\\d+\\.mlp\\.", 2),
            ("language_model\\.layers\\.\\d+\\.self_attn\\.", 4),
            ("language_model\\.layers\\.\\d+\\.per_layer_input_gate$", 8),
            ("language_model\\.layers\\.\\d+\\.per_layer_projection$", 8),
        ]
        return config
    }

    @Test("resolveModuleBits resolves the E2B mobile schema (gemma4 namespace)")
    func resolveModuleBitsGemma4() throws {
        let config = e2bMobileConfig()
        #expect(resolveModuleBits(path: "language_model.lm_head", config: config) == 2)
        #expect(resolveModuleBits(path: "language_model.model.embed_tokens", config: config) == 2)
        #expect(
            resolveModuleBits(path: "language_model.model.embed_tokens_per_layer", config: config)
                == 4)
        // MLP layers 0–14 → 4 bits, 15–34 → 2 bits (first-match-wins).
        #expect(
            resolveModuleBits(path: "language_model.model.layers.0.mlp.gate_proj", config: config)
                == 4)
        #expect(
            resolveModuleBits(path: "language_model.model.layers.14.mlp.down_proj", config: config)
                == 4)
        #expect(
            resolveModuleBits(path: "language_model.model.layers.15.mlp.gate_proj", config: config)
                == 2)
        #expect(
            resolveModuleBits(path: "language_model.model.layers.34.mlp.up_proj", config: config)
                == 2)
        // Attention → 4 bits; PLE gates/projections → 8 bits.
        #expect(
            resolveModuleBits(path: "language_model.model.layers.0.self_attn.q_proj", config: config)
                == 4)
        #expect(
            resolveModuleBits(
                path: "language_model.model.layers.0.per_layer_input_gate", config: config) == 8)
        #expect(
            resolveModuleBits(
                path: "language_model.model.layers.0.per_layer_projection", config: config) == 8)
        // modules_to_not_convert → nil (stays fp).
        #expect(
            resolveModuleBits(
                path: "language_model.model.per_layer_model_projection", config: config) == nil)
    }

    @Test("resolveModuleBits resolves the E2B mobile schema (gemma4_text namespace)")
    func resolveModuleBitsGemma4Text() throws {
        let config = e2bMobileConfig()
        #expect(resolveModuleBits(path: "lm_head", config: config) == 2)
        #expect(resolveModuleBits(path: "model.embed_tokens", config: config) == 2)
        #expect(resolveModuleBits(path: "model.embed_tokens_per_layer", config: config) == 4)
        #expect(resolveModuleBits(path: "model.layers.0.mlp.gate_proj", config: config) == 4)
        #expect(resolveModuleBits(path: "model.layers.15.mlp.gate_proj", config: config) == 2)
        #expect(resolveModuleBits(path: "model.layers.0.self_attn.q_proj", config: config) == 4)
        #expect(resolveModuleBits(path: "model.per_layer_model_projection", config: config) == nil)
    }

    @Test("GemmaMobileQuantizationConfig decodes from JSON preserving order")
    func decodeQuantizationConfig() throws {
        // Simplified patterns (no regex backslashes) so the JSON is literal.
        let json = """
            {
              "quant_method": "gemma",
              "num_bits": 4,
              "quantize_embeddings": true,
              "modules_to_not_convert": ["per_layer_model_projection"],
              "module_quant_configs": {
                "^lm_head$": {"num_bits": 2},
                "language_model.embed_tokens$": {"num_bits": 2},
                "language_model.layers.[0-9]+.mlp.": {"num_bits": 4},
                "language_model.layers.[0-9]+.self_attn.": {"num_bits": 4}
              }
            }
            """
        // No rawConfigData in userInfo → falls back to alphabetical sort.
        let config = try JSONDecoder().decode(
            GemmaMobileQuantizationConfig.self, from: Data(json.utf8))
        #expect(config.isGemmaMobile)
        #expect(config.quantizeEmbeddings)
        #expect(config.numBits == 4)
        #expect(config.modulesToNotConvert == ["per_layer_model_projection"])
        // Alphabetical fallback (JSONDecoder doesn't preserve insertion order):
        // ^lm_head$ < language_model.embed_tokens$ < ...mlp. < ...self_attn.
        #expect(
            config.moduleQuantConfigs.map { $0.0 }
                == ["^lm_head$", "language_model.embed_tokens$",
                    "language_model.layers.[0-9]+.mlp.", "language_model.layers.[0-9]+.self_attn."])
        #expect(resolveModuleBits(path: "language_model.lm_head", config: config) == 2)
        #expect(resolveModuleBits(path: "language_model.model.layers.0.mlp.gate_proj", config: config) == 4)
    }

    @Test("GemmaMobileQuantizationConfig preserves JSON insertion order with rawConfigData")
    func decodeQuantizationConfigInsertionOrder() throws {
        // Patterns in an order where insertion != alphabetical. The catch-all
        // 2-bit mlp pattern is listed first and the specific 4-bit pattern
        // second — the opposite of alphabetical. With insertion-order
        // recovery the 2-bit pattern must come first (matching the Python
        // reference's first-match-wins semantics).
        let json = """
            {
              "quant_method": "gemma",
              "num_bits": 4,
              "module_quant_configs": {
                "language_model.layers.[0-9]+.mlp.": {"num_bits": 2},
                "language_model.layers.([0-9]|1[0-4]).mlp.": {"num_bits": 4}
              }
            }
            """
        let data = Data(json.utf8)
        let decoder = JSONDecoder()
        decoder.userInfo[.rawConfigData] = data
        let config = try decoder.decode(GemmaMobileQuantizationConfig.self, from: data)
        // Insertion order preserved: 2-bit catch-all first, 4-bit specific second.
        #expect(
            config.moduleQuantConfigs.map { $0.0 }
                == ["language_model.layers.[0-9]+.mlp.", "language_model.layers.([0-9]|1[0-4]).mlp."])
        #expect(config.moduleQuantConfigs.map { $0.1 } == [2, 4])
        // First-match-wins: a layer-0 mlp path hits the 2-bit catch-all first.
        #expect(resolveModuleBits(path: "language_model.model.layers.0.mlp.gate_proj", config: config) == 2)
    }

    // MARK: - Helpers

    private static func assertApproximatelyEqual(
        _ actual: [Float], _ expected: [Float], tolerance: Float
    ) {
        #expect(actual.count == expected.count)
        for (a, e) in zip(actual, expected) {
            #expect(abs(a - e) < tolerance, "got \(actual), expected \(expected)")
        }
    }
}
