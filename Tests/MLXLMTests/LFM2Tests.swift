import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class LFM2Tests: XCTestCase {

    private func makeConfig(blockFFDim: Int? = nil) throws -> LFM2Configuration {
        let explicitBlockFFDim = blockFFDim.map { "\"block_ff_dim\": \($0)," } ?? ""
        let json = """
            {
                "model_type": "lfm2",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 24,
                \(explicitBlockFFDim)
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "norm_eps": 1e-5,
                "conv_bias": false,
                "conv_L_cache": 3,
                "block_multiple_of": 8,
                "block_ffn_dim_multiplier": 1.0,
                "block_auto_adjust_ff_dim": false,
                "layer_types": ["full_attention"]
            }
            """
        return try JSONDecoder().decode(LFM2Configuration.self, from: Data(json.utf8))
    }

    func testIntermediateSizeIsUsedWhenBlockFFDimIsMissing() throws {
        let configuration = try makeConfig()

        XCTAssertEqual(configuration.blockFFDim, 24)
    }

    func testBlockFFDimTakesPrecedenceOverIntermediateSize() throws {
        let configuration = try makeConfig(blockFFDim: 16)

        XCTAssertEqual(configuration.blockFFDim, 16)
    }

    func testSanitizeRemovesMLXVLMTopLevelLanguageModelPrefix() throws {
        let model = LFM2Model(try makeConfig())
        let weights = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "language_model.model.embed_tokens.scales": MLXArray.zeros([32, 1]),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.scales"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["language_model.model.embed_tokens.weight"])
    }

    func testSanitizeKeepsExistingStandaloneNamespace() throws {
        let model = LFM2Model(try makeConfig())
        let weights = ["model.embed_tokens.weight": MLXArray.zeros([32, 8])]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }
}
