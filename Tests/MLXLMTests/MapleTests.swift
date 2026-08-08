import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class MapleTests: XCTestCase {
    private func configuration(
        layerTypes: [String] = ["sliding_attention", "full_attention"],
        tied: Bool = false
    ) throws -> MapleConfiguration {
        let types = layerTypes.map { "\"\($0)\"" }.joined(separator: ",")
        let json = """
            {
              "model_type": "maple",
              "hidden_size": 8,
              "intermediate_size": 12,
              "moe_intermediate_size": 4,
              "num_hidden_layers": \(layerTypes.count),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "num_experts": 3,
              "num_experts_per_tok": 2,
              "first_k_dense_replace": 1,
              "partial_rotary_factor": 0.5,
              "vocab_size": 16,
              "sliding_window": 4,
              "layer_types": [\(types)],
              "tie_word_embeddings": \(tied),
              "quantization": {"group_size": 4, "bits": 2}
            }
            """
        return try JSONDecoder().decode(MapleConfiguration.self, from: Data(json.utf8))
    }

    func testConfigurationDefaultsAndLayerTypeResolution() throws {
        let config = try JSONDecoder().decode(
            MapleConfiguration.self,
            from: Data("{\"model_type\":\"maple\",\"num_hidden_layers\":2,\"layer_types\":[]}".utf8)
        )
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.numExperts, 256)
        XCTAssertEqual(config.layerTypes, ["full_attention", "full_attention"])
        XCTAssertEqual(config.partialRotaryFactor, 0.5)
        try config.validateModelConfiguration()
    }

    func testConfigurationValidationRejectsMalformedValues() throws {
        var config = try configuration()
        config.layerTypes = ["full_attention"]
        XCTAssertThrowsError(try config.validateModelConfiguration())

        config = try configuration()
        config.numExpertsPerToken = config.numExperts + 1
        XCTAssertThrowsError(try config.validateModelConfiguration())

        config = try configuration()
        config.partialRotaryFactor = 0.25  // one rotary dimension
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRMSNormUsesFloat32AndRestoresActivationDType() throws {
        let norm = MapleRMSNorm(dimensions: 4, eps: 1e-6)
        try norm.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXArray([0.5, 1.0, 1.5, 2.0] as [Float])
            ]), verify: [])
        let input = MLXArray([1.0, -2.0, 3.0, -4.0] as [Float]).asType(.bfloat16)
        let output = norm(input)
        let reference = MLXFast.rmsNorm(
            input.asType(.float32), weight: norm.weight.asType(.float32), eps: 1e-6
        ).asType(.bfloat16)
        eval(output, reference)
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertEqual(output.asArray(Float.self), reference.asArray(Float.self))
    }

    func testRouterUsesFloat32SoftmaxAndRenormalizesTopK() throws {
        let gate = MapleGate(try configuration())
        try gate.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXArray(
                    [
                        1.0, 0, 0, 0, 0, 0, 0, 0,
                        0.9999, 0, 0, 0, 0, 0, 0, 0,
                        -1.0, 0, 0, 0, 0, 0, 0, 0,
                    ] as [Float]
                ).reshaped(3, 8)
            ]), verify: [])
        let x = MLXArray([1.0, 0, 0, 0, 0, 0, 0, 0] as [Float]).asType(.bfloat16)
            .reshaped(1, 1, 8)
        let result = gate(x)
        eval(result.indices, result.scores)
        let ids = Set(result.indices.asArray(Int32.self))
        XCTAssertEqual(ids, Set([0, 1]))
        XCTAssertEqual(result.scores.dtype, .float32)
        XCTAssertEqual(result.scores.sum().item(Float.self), 1, accuracy: 1e-6)
    }

    func testExpertClampAndFloat32AggregationPreserveDType() {
        let gate = MLXArray([8, -8] as [Float]).asType(.bfloat16)
        let up = MLXArray([9, -9] as [Float]).asType(.bfloat16)
        let activated = mapleClampedSwiGLU(gate, up)
        let expected =
            silu(minimum(gate, MLXArray(Float(7)).asType(.bfloat16)))
            * clip(
                up, min: MLXArray(Float(-7)).asType(.bfloat16),
                max: MLXArray(Float(7)).asType(.bfloat16))
        eval(activated, expected)
        XCTAssertEqual(activated.dtype, .bfloat16)
        XCTAssertEqual(activated.asArray(Float.self), expected.asArray(Float.self))

        let outputs = MLXArray([1000, 0.125, -1000, 0.25] as [Float]).asType(.bfloat16)
            .reshaped(1, 1, 2, 2)
        let scores = MLXArray([0.5001, 0.4999] as [Float]).reshaped(1, 1, 2)
        let aggregated = mapleAggregateExpertOutputs(outputs, scores)
        let reference = (outputs.asType(.float32) * expandedDimensions(scores, axis: -1)).sum(
            axis: -2
        ).asType(.bfloat16)
        eval(aggregated, reference)
        XCTAssertEqual(aggregated.dtype, .bfloat16)
        XCTAssertEqual(aggregated.asArray(Float.self), reference.asArray(Float.self))
    }

    func testSlidingPartialRoPEAndFullNoPEAtNonzeroOffset() throws {
        let config = try configuration()
        let sliding = MapleAttention(config, layerType: "sliding_attention")
        let full = MapleAttention(config, layerType: "full_attention")
        try full.update(parameters: sliding.parameters(), verify: [.all])

        let slidingCache = KVCacheSimple()
        let fullCache = KVCacheSimple()
        let oldKeys = MLXArray([0.2, -0.1, 0.4, 0.3, -0.5, 0.6, 0.7, -0.2] as [Float])
            .reshaped(1, 1, 2, 4)
        let oldValues = MLXArray([0.3, 0.8, -0.4, 0.1, 0.5, -0.7, 0.2, 0.9] as [Float])
            .reshaped(1, 1, 2, 4)
        _ = slidingCache.update(keys: oldKeys, values: oldValues)
        _ = fullCache.update(keys: oldKeys, values: oldValues)
        let input = MLXArray([0.1, 0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8] as [Float])
            .reshaped(1, 1, 8)
        let rotated = sliding(input, mask: .none, cache: slidingCache)
        let unrotated = full(input, mask: .none, cache: fullCache)
        eval(rotated, unrotated)
        XCTAssertEqual(slidingCache.offset, 3)
        XCTAssertEqual(fullCache.offset, 3)
        XCTAssertNotEqual(rotated.asArray(Float.self), unrotated.asArray(Float.self))
    }

    func testGenericQuantizationCanRepresentMixedMapleLayout() throws {
        let json = """
            {"model_type":"maple", "hidden_size":32, "intermediate_size":32,
             "moe_intermediate_size":32, "num_hidden_layers":2,
             "num_attention_heads":2, "num_key_value_heads":1, "head_dim":16,
             "num_experts":2, "num_experts_per_tok":1, "first_k_dense_replace":1,
             "vocab_size":32, "layer_types":["full_attention","full_attention"]}
            """
        let config = try JSONDecoder().decode(MapleConfiguration.self, from: Data(json.utf8))
        let model = MapleModel(config)
        quantize(model: model) { path, _ in
            if path == "model.word_embeddings" || path == "lm_head" {
                return (32, 4, .affine)
            }
            return (32, 2, .affine)
        }
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        XCTAssertTrue(leaves["model.layers.0.self_attn.qkv_proj"] is QuantizedLinear)
        XCTAssertTrue(
            leaves["model.layers.1.mlp.switch_mlp.up_gate_proj"] is QuantizedSwitchLinear)
        XCTAssertTrue(leaves["model.word_embeddings"] is QuantizedEmbedding)
        XCTAssertTrue(leaves["lm_head"] is QuantizedLinear)
        XCTAssertFalse(leaves["model.layers.1.mlp.gate"] is Quantized)
    }

    func testExpertGraphBenchmark() throws {
        guard ProcessInfo.processInfo.environment["MAPLE_EXPERT_BENCHMARK"] == "1" else {
            throw XCTSkip("Set MAPLE_EXPERT_BENCHMARK=1 to run")
        }
        let gate = MLXRandom.normal([8, 512]).asType(.bfloat16)
        let up = MLXRandom.normal([8, 512]).asType(.bfloat16)
        let outputs = MLXRandom.normal([1, 1, 8, 2048]).asType(.bfloat16)
        let scores = softmax(MLXRandom.normal([1, 1, 8]), axis: -1)

        func portableActivation() -> MLXArray {
            silu(minimum(gate, MLXArray(Float(7)).asType(gate.dtype)))
                * clip(
                    up, min: MLXArray(Float(-7)).asType(up.dtype),
                    max: MLXArray(Float(7)).asType(up.dtype))
        }
        func portableAggregate() -> MLXArray {
            (outputs.asType(.float32) * expandedDimensions(scores, axis: -1))
                .sum(axis: -2).asType(outputs.dtype)
        }
        eval(mapleClampedSwiGLU(gate, up), mapleAggregateExpertOutputs(outputs, scores))
        eval(portableActivation(), portableAggregate())

        func measure(_ body: () -> MLXArray) -> Double {
            let start = ProcessInfo.processInfo.systemUptime
            for _ in 0 ..< 200 { eval(body()) }
            return ProcessInfo.processInfo.systemUptime - start
        }
        let compiled =
            measure { mapleClampedSwiGLU(gate, up) }
            + measure { mapleAggregateExpertOutputs(outputs, scores) }
        let portable = measure(portableActivation) + measure(portableAggregate)
        print(
            "Maple expert graphs: compiled=\(compiled)s portable=\(portable)s speedup=\(portable / compiled)x"
        )
    }

    func testMixedCachesAndPrefillDecodeContinuation() throws {
        let model = MapleModel(try configuration())
        let cache = model.newCache(parameters: nil)
        XCTAssertTrue(cache[0] is RotatingKVCache)
        XCTAssertTrue(cache[1] is KVCacheSimple)

        let prefill = model(MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3), cache: cache)
        eval(prefill)
        XCTAssertEqual(prefill.shape, [1, 3, 16])
        XCTAssertEqual(cache[0].offset, 3)
        XCTAssertEqual(cache[1].offset, 3)

        let decode = model(MLXArray([4] as [Int32]).reshaped(1, 1), cache: cache)
        eval(decode)
        XCTAssertEqual(decode.shape, [1, 1, 16])
        XCTAssertEqual(cache[0].offset, 4)
        XCTAssertEqual(cache[1].offset, 4)
        XCTAssertTrue(decode.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testTypeRegistryCreatesMapleAndRegistryContainsCheckpoint() async throws {
        let data = Data(
            "{\"model_type\":\"maple\",\"hidden_size\":8,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":4,\"num_hidden_layers\":1,\"num_experts\":2,\"num_experts_per_tok\":1,\"vocab_size\":16}"
                .utf8)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "maple")
        XCTAssertTrue(model is MapleModel)
        XCTAssertTrue(LLMRegistry.shared.contains(id: "deepgrove/maple-2bit-mlx"))
        XCTAssertEqual(LLMRegistry.maple2bitMLX.name, "deepgrove/maple-2bit-mlx")
    }

    func testTiedAndUntiedHeadsProduceLogits() throws {
        for tied in [false, true] {
            let model = MapleModel(try configuration(layerTypes: ["full_attention"], tied: tied))
            let logits = model(MLXArray([1] as [Int32]).reshaped(1, 1), cache: nil)
            eval(logits)
            XCTAssertEqual(logits.shape, [1, 1, 16])
        }
    }
}
