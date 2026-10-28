// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

func lfm3TestConfig(
    layers: [String] = ["conv", "full_attention", "sliding_attention"], window: Int = 256
) throws -> LFM3Configuration {
    let json: [String: Any] = [
        "model_type": "lfm3", "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24,
        "num_hidden_layers": layers.count, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "conv_L_cache": 7, "sliding_window": window, "layer_types": layers,
    ]
    return try JSONDecoder().decode(
        LFM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
}

final class LFM3ExtensionTests: XCTestCase {
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

    func testReasoningHistoryPreservesToolMetadata() {
        let call = ToolCall(function: .init(name: "lookup", arguments: ["key": "a", "count": 2]))
        var message = Chat.Message.assistant("reason</think>answer", toolCalls: [call])
        message.prefilledReasoningStartDelimiter = "<think>"
        let raw = LFM3MessageGenerator().generate(message: message)
        XCTAssertEqual(raw["reasoning_content"] as? String, "reason")
        XCTAssertEqual(raw["content"] as? String, "answer")
        XCTAssertNotNil(raw["tool_calls"])
    }
}
