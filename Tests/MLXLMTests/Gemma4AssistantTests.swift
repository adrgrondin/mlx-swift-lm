import MLXVLM
import XCTest

final class Gemma4AssistantTests: XCTestCase {
    func testGemma4AssistantConfigurationDefaultsSharedKVLayers() throws {
        let json = """
            {
              "model_type": "gemma4_assistant",
              "backbone_hidden_size": 16,
              "block_size": 4,
              "tie_word_embeddings": true,
              "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "global_head_dim": 4,
                "vocab_size": 16,
                "hidden_size_per_layer_input": 0,
                "num_kv_shared_layers": 0,
                "layer_types": ["sliding_attention", "full_attention"]
              }
            }
            """

        let config = try JSONDecoder().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "gemma4_assistant")
        XCTAssertEqual(config.textConfiguration.numKVSharedLayers, 2)
    }
}
