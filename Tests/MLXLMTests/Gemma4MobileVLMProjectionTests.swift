// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

final class Gemma4MobileVLMProjectionTests: XCTestCase {
    func testFusedQKVRevalidatesAfterProjectionUpdates() throws {
        let data = Data(
            """
            {"model_type":"gemma4_text","hidden_size":512,"num_hidden_layers":1,
             "intermediate_size":128,"num_attention_heads":1,"num_key_value_heads":1,
             "head_dim":128,"global_head_dim":128,"vocab_size":32,
             "layer_types":["sliding_attention"],"sliding_window":32,
             "attention_k_eq_v":false,"num_kv_shared_layers":0}
            """.utf8)
        let config = try JSONDecoder().decode(MLXVLM.Gemma4TextConfiguration.self, from: data)

        func projection(inputScale: Float = 0) -> GemmaQuantizedLinear {
            let result = GemmaQuantizedLinear(inputDims: 512, outputDims: 128, numBits: 4)
            result.update(
                parameters: ModuleParameters.unflattened([
                    ("weight", MLXArray([UInt8](repeating: 0x99, count: 128 * 256), [128, 256])),
                    ("input_activation_scale", MLXArray(inputScale)),
                ]))
            return result
        }

        for update in ["replacement", "parameters", "leafParameters"] {
            let attention = MLXVLM.Gemma4TextAttention(config: config, layerIdx: 0)
            try attention.update(
                modules: ModuleChildren.unflattened([
                    ("q_proj", projection() as Module),
                    ("k_proj", projection() as Module),
                    ("v_proj", projection() as Module),
                ]), verify: .all)

            func expectKeys(_ expected: Float) throws {
                let (_, state, _) = attention(MLXArray.ones([1, 1, 512]))
                guard case .regular(let keys, _)? = state else {
                    return XCTFail("Expected regular KV state")
                }
                XCTAssertLessThan(
                    abs(keys - expected).max().item(Float.self), 1e-5, update)
            }

            try expectKeys(1)
            for scale: Float in [100, 0, 100] {
                switch update {
                case "replacement":
                    try attention.update(
                        modules: ModuleChildren.unflattened([
                            ("k_proj", projection(inputScale: scale) as Module)
                        ]), verify: .all)
                case "parameters":
                    try attention.update(
                        parameters: ModuleParameters.unflattened([
                            ("k_proj.input_activation_scale", MLXArray(scale))
                        ]), verify: [.noUnusedKeys, .shapeMismatch])
                default:
                    let key = try XCTUnwrap(attention.kProj as? GemmaQuantizedLinear)
                    try key.update(
                        parameters: ModuleParameters.unflattened([
                            ("input_activation_scale", MLXArray(scale))
                        ]), verify: [.noUnusedKeys, .shapeMismatch])
                }
                // A scale of 100 rounds K's input to zero; Q/V remain uncalibrated.
                try expectKeys(scale == 0 ? 1 : 0)
            }
        }
    }
}
