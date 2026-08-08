import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class MapleSanitizeTests: XCTestCase {
    private func model(tied: Bool = false) throws -> MapleModel {
        let json = """
            {
              "model_type":"maple", "hidden_size":8, "intermediate_size":12,
              "moe_intermediate_size":4, "num_hidden_layers":1,
              "num_attention_heads":2, "num_key_value_heads":1, "head_dim":4,
              "num_experts":2, "num_experts_per_tok":1,
              "vocab_size":16, "layer_types":["full_attention"],
              "tie_word_embeddings":\(tied),
              "quantization":{"group_size":4,"bits":2}
            }
            """
        return MapleModel(try JSONDecoder().decode(MapleConfiguration.self, from: Data(json.utf8)))
    }

    func testExpandsCompactRowAlphaAndPreservesExpandedTensors() throws {
        let m = try model()
        var weights: [String: MLXArray] = [
            "x.weight": zeros([2, 2], dtype: .uint32),
            "x.row_alpha": MLXArray([0.25, 0.5] as [Float]),
        ]
        var result = m.sanitize(weights: weights)
        XCTAssertNil(result["x.row_alpha"])
        XCTAssertEqual(result["x.scales"]?.shape, [2, 8])
        eval(result["x.scales"]!, result["x.biases"]!)
        XCTAssertEqual(result["x.scales"]!.asArray(Float.self).first, 0.25)
        XCTAssertEqual(result["x.biases"]!.asArray(Float.self).first, -0.25)

        weights["x.scales"] = ones([2, 8])
        weights["x.biases"] = zeros([2, 8])
        result = m.sanitize(weights: weights)
        XCTAssertEqual(result["x.scales"]!.sum().item(Float.self), 16)
        XCTAssertEqual(result["x.biases"]!.sum().item(Float.self), 0)
    }

    func testStacksExpertsAndFusesUpThenGateForAllSuffixes() throws {
        let m = try model()
        var weights: [String: MLXArray] = [:]
        let prefix = "model.layers.0.mlp.experts"
        for expert in 0 ..< 2 {
            for projection in ["up_proj", "gate_proj", "down_proj"] {
                let value = Float(
                    10 * expert + (projection == "up_proj" ? 1 : projection == "gate_proj" ? 2 : 3))
                weights["\(prefix).\(expert).\(projection).weight"] = full(
                    [projection == "down_proj" ? 8 : 4, projection == "down_proj" ? 4 : 8],
                    values: MLXArray(value))
            }
        }
        let result = m.sanitize(weights: weights)
        let fused = try XCTUnwrap(
            result["model.layers.0.mlp.switch_mlp.up_gate_proj.weight"])
        eval(fused)
        XCTAssertEqual(fused.shape, [2, 8, 8])
        XCTAssertEqual(fused[0, 0, 0].item(Float.self), 1)
        XCTAssertEqual(fused[0, 4, 0].item(Float.self), 2)
        XCTAssertEqual(fused[1, 0, 0].item(Float.self), 11)
        XCTAssertNotNil(result["model.layers.0.mlp.switch_mlp.down_proj.weight"])
        XCTAssertFalse(result.keys.contains { $0.contains(".experts.") })
    }

    func testFusesQKVInOrderForQuantizationSuffixes() throws {
        let m = try model()
        var weights: [String: MLXArray] = [:]
        let prefix = "model.layers.0.self_attn"
        for suffix in ["weight", "scales", "biases", "bias"] {
            for (index, projection) in ["q_proj", "k_proj", "v_proj"].enumerated() {
                let shape = suffix == "bias" ? [1] : [1, 2]
                weights["\(prefix).\(projection).\(suffix)"] = full(
                    shape, values: MLXArray(Float(index + 1)))
            }
        }
        let result = m.sanitize(weights: weights)
        for suffix in ["weight", "scales", "biases", "bias"] {
            let fused = try XCTUnwrap(result["\(prefix).qkv_proj.\(suffix)"])
            eval(fused)
            let values = fused.asArray(Float.self)
            XCTAssertEqual(values.first, 1)
            XCTAssertEqual(values.last, 3)
        }
    }

    func testCleansTiedHeadFlashHeadAndRotaryWeights() throws {
        let result = try model(tied: true).sanitize(weights: [
            "lm_head.weight": ones([1]),
            "lm_head.scales": ones([1]),
            "lm_head_flash.centroids": ones([1]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": ones([1]),
            "model.word_embeddings.weight": ones([1]),
        ])
        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNil(result["lm_head.scales"])
        XCTAssertNil(result["lm_head_flash.centroids"])
        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.word_embeddings.weight"])
    }
}
