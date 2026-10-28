// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

func lfm3TestConfig(
    layers: [String] = ["conv", "full_attention", "sliding_attention"], window: Int = 256
) throws -> LFM3Configuration {
    let json: [String: Any] = [
        "model_type": "lfm3", "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24,
        "num_hidden_layers": layers.count, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "conv_L_cache": 7, "sliding_window": window, "layer_types": layers,
    ]
    let config = try JSONDecoder().decode(
        LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
    try config.validateModelConfiguration()
    return config
}

final class LFM3Tests: XCTestCase {
    func testPartialRotaryCoordinatesFrequenciesAndBatchOffsets() {
        Device.withDefaultDevice(.cpu) {
            let rope = LFM3RotaryEmbedding(dimensions: 128, base: 1e6, partial: 0.5)
            let input = MLXArray((0 ..< 512).map { Float($0 % 128) / 128 }).reshaped(2, 1, 2, 128)
            for dtype in [DType.float32, .bfloat16] {
                let x = input.asType(dtype)
                let actual = rope(x, offset: MLXArray([Int32(17), 262142])).asType(.float32)
                    .asArray(Float.self)
                let values = x.asType(.float32).asArray(Float.self)
                for batch in 0 ..< 2 {
                    for token in 0 ..< 2 {
                        let base = (batch * 2 + token) * 128
                        let position = Float((batch == 0 ? 17 : 262142) + token)
                        for i in 0 ..< 64 {
                            let angle =
                                i < 32
                                ? position / Foundation.pow(Float(1e6), Float(2 * i) / 128) : 0
                            let left = values[base + i]
                            let right = values[base + 64 + i]
                            XCTAssertEqual(
                                actual[base + i], left * cos(angle) - right * sin(angle),
                                accuracy: dtype == .float32 ? 0.02 : 0.03)
                            XCTAssertEqual(
                                actual[base + 64 + i], right * cos(angle) + left * sin(angle),
                                accuracy: dtype == .float32 ? 0.02 : 0.03)
                            if i >= 32 {
                                XCTAssertEqual(actual[base + i], left)
                                XCTAssertEqual(actual[base + 64 + i], right)
                            }
                        }
                    }
                }
            }
        }
    }

    func testAttentionAppliesOneGatePerHeadBeforeOutputProjection() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try lfm3TestConfig(layers: ["full_attention"])
            let attention = LFM3Attention(c, type: "full_attention")
            let gate = MLXArray.zeros([2, 16])
            gate[1, 0] = MLXArray(log(Float(3)))
            let value = MLXArray.zeros([8, 16])
            for i in 0 ..< 8 { value[i, i] = MLXArray(Float(1)) }
            try attention.update(
                parameters: ModuleParameters.unflattened([
                    "q_proj.weight": MLXArray.zeros([16, 16]),
                    "k_proj.weight": MLXArray.zeros([8, 16]),
                    "v_proj.weight": value, "o_proj.weight": eye(16), "gate_proj.weight": gate,
                ]), verify: [.noUnusedKeys])
            let output = attention(
                MLXArray((1 ... 16).map(Float.init)).reshaped(1, 1, 16), cache: nil
            ).asArray(Float.self)
            for i in 0 ..< 8 {
                XCTAssertEqual(output[i], Float(i + 1) * 0.5, accuracy: 1e-6)
                XCTAssertEqual(output[i + 8], Float(i + 1) * 0.75, accuracy: 1e-6)
            }
        }
    }

    func testSandwichNormNormalizesOperatorBeforeAddingResidual() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try lfm3TestConfig(layers: ["conv"])
            let layer = LFM3DecoderLayer(c, type: "conv")
            let kernel = MLXArray.zeros([16, 7, 1])
            kernel[0..., 6, 0] = MLXArray.ones([16])
            try layer.update(
                parameters: ModuleParameters.unflattened([
                    "conv.in_proj.weight": concatenated([eye(16), eye(16), eye(16)]),
                    "conv.out_proj.weight": eye(16), "conv.conv.weight": kernel,
                    "feed_forward.w1.weight": MLXArray.zeros([24, 16]),
                ]), verify: [.noUnusedKeys])
            let values = (1 ... 16).map(Float.init)
            let norm = sqrt(values.reduce(0) { $0 + $1 * $1 } / 16 + c.normEps)
            let cubed = values.map { pow($0 / norm, 3) }
            let postNorm = sqrt(cubed.reduce(0) { $0 + $1 * $1 } / 16 + c.normEps)
            let result = layer(MLXArray(values).reshaped(1, 1, 16), cache: nil).asArray(Float.self)
            for i in values.indices {
                XCTAssertEqual(result[i], values[i] + cubed[i] / postNorm, accuracy: 1e-5)
            }
        }
    }

    func testCachedChunksMatchFullForwardAcrossConvolutionAndSlidingWindowBoundaries() throws {
        try Device.withDefaultDevice(.cpu) {
            let configuration = try lfm3TestConfig()
            let model = withRandomState(MLXRandom.RandomState(seed: 17)) {
                LFM3Model(configuration)
            }
            let tokens = MLXArray((0 ..< 273).map { Int32($0 % 32) }).reshaped(1, -1)
            let full = model(tokens, cache: nil)
            for chunks in [[3, 4, 250, 16], Array(repeating: 1, count: 273)] {
                let cache = try model.newCache(parameters: nil)
                var start = 0
                let outputs = chunks.map { length in
                    defer { start += length }
                    return model(tokens[0..., start ..< start + length], cache: cache)
                }
                let cached = concatenated(outputs, axis: 1)
                XCTAssertLessThan(abs(full - cached).max().item(Float.self), 2e-4)
            }
        }
    }

    func testPaddedConvolutionStateKeepsLastValidInputsAcrossChunks() throws {
        try Device.withDefaultDevice(.cpu) {
            let conv = LFM3ShortConv(try lfm3TestConfig(layers: ["conv"]))
            let x = MLXArray((0 ..< 320).map { Float($0 % 23) / 23 }).reshaped(2, 10, 16)
            let cache = MambaCache()
            cache.prepare(lengths: [4, 10])
            _ = conv(x[0..., 0 ..< 7, 0...], cache: cache)
            _ = conv(x[0..., 7..., 0...], cache: cache)
            cache.finalize()
            let next = MLXArray.ones([2, 1, 16])
            let actual = conv(next, cache: cache)
            for row in 0 ..< 2 {
                let length = row == 0 ? 4 : 10
                let referenceCache = MambaCache()
                _ = conv(x[row ..< row + 1, 0 ..< length, 0...], cache: referenceCache)
                let expected = conv(next[row ..< row + 1], cache: referenceCache)
                XCTAssertLessThan(
                    abs(actual[row ..< row + 1] - expected).max().item(Float.self), 1e-5)
            }
        }
    }

    func testCheckpointMappingPreservesEmbeddingPrecisionAndIsIdempotent() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LFM3Model(try lfm3TestConfig())
            let checkpoint = ModelCheckpoint(
                weights: [
                    "language_model.model.embed_tokens.weight": MLXArray.zeros([32, 16]),
                    "language_model.model.layers.0.conv.conv.weight": MLXArray(0 ..< 112).reshaped(
                        16, 1, 7),
                ],
                perLayerQuantization: .init(
                    quantization: .init(groupSize: 32, bits: 4),
                    perLayerQuantization: [
                        "language_model.model.embed_tokens": .quantize(
                            .init(groupSize: 32, bits: 8))
                    ]))
            let prepared = try model.prepareCheckpoint(checkpoint)
            XCTAssertEqual(
                prepared.perLayerQuantization?.quantization(layer: "model.embed_tokens")?.bits, 8)
            let twice = try model.prepareCheckpoint(prepared)
            XCTAssertEqual(Set(prepared.weights.keys), Set(twice.weights.keys))
            XCTAssertEqual(twice.weights["model.layers.0.conv.conv.weight"]?.shape, [16, 7, 1])
            XCTAssertEqual(
                twice.weights["model.layers.0.conv.conv.weight"]?.asArray(Int32.self),
                Array(0 ..< 112).map(Int32.init))
        }
    }

    func testUnfinishedReasoningRemainsSeparateFromAnswer() {
        for prefilled in [true, false] {
            var message = Chat.Message.assistant(
                prefilled ? "unfinished reasoning" : "<think>unfinished reasoning")
            if prefilled { message.prefilledReasoningStartDelimiter = "<think>" }
            for multimodal in [false, true] {
                let generator = LFM3MessageGenerator(multimodal: multimodal)
                let raw = generator.generate(message: message)
                XCTAssertEqual(raw["reasoning_content"] as? String, "unfinished reasoning")
                if multimodal {
                    XCTAssertEqual(
                        raw["content"] as? [[String: String]], [["type": "text", "text": ""]])
                } else {
                    XCTAssertEqual(raw["content"] as? String, "")
                }
                let answer = generator.generate(message: .assistant("ordinary answer"))
                XCTAssertNil(answer["reasoning_content"])
                if multimodal {
                    XCTAssertEqual(
                        answer["content"] as? [[String: String]],
                        [["type": "text", "text": "ordinary answer"]])
                } else {
                    XCTAssertEqual(answer["content"] as? String, "ordinary answer")
                }
            }
        }
    }

    func testReasoningHistoryAndXMLToolCallsUseReleasedWireFormat() throws {
        let call = ToolCall(function: .init(name: "lookup", arguments: ["key": "a", "count": 2]))
        var message = Chat.Message.assistant("reason</think>answer", toolCalls: [call])
        message.prefilledReasoningStartDelimiter = "<think>"
        let raw = LFM3MessageGenerator().generate(message: message)
        XCTAssertEqual(raw["reasoning_content"] as? String, "reason")
        XCTAssertEqual(raw["content"] as? String, "answer")
        XCTAssertNotNil(raw["tool_calls"])
        let parsed = ToolCallFormat.xmlFunction.createParser().parse(
            content: """
                <tool_call>
                <function=lookup>
                <parameter=key>
                a
                </parameter>
                </function>
                </tool_call>
                """, tools: nil)
        XCTAssertEqual(parsed?.function.name, "lookup")
        XCTAssertEqual(parsed?.function.arguments["key"], .string("a"))
    }
}
