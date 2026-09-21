// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class PrismHadamardTests: XCTestCase {
    private func configuration(schema: Int = 2) throws -> Data {
        var records: [[String: Any]] = [
            ["path": "model.embed_tokens", "block": 512, "embedding": true, "dtype": "float16"],
            ["path": "lm_head", "block": 512, "embedding": false, "dtype": "float16"],
        ]
        for layer in 0 ..< 2 {
            let projections =
                layer == 0
                ? ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
                : ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
            for suffix in projections + ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                records.append([
                    "path": "model.layers.\(layer).\(suffix)", "block": 512,
                    "embedding": false, "dtype": "float16",
                ])
            }
        }
        let config: [String: Any] = [
            "schema_version": schema, "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5", "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "components": ["text": true, "vision": schema == 2],
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": records, "image_token_id": 15, "video_token_id": 14,
            "vision_start_token_id": 13, "tie_word_embeddings": false,
            "text_config": [
                "model_type": "qwen3_5_text", "hidden_size": 512, "intermediate_size": 512,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 128, "vocab_size": 16, "tie_word_embeddings": false,
                "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
                "partial_rotary_factor": 0.25,
                "rope_parameters": ["rope_theta": 10_000_000, "mrope_section": [4, 6, 6]],
            ],
            "vision_config": [
                "model_type": "qwen3_5", "depth": 1, "hidden_size": 32, "intermediate_size": 64,
                "out_hidden_size": 512,
                "num_heads": 4, "patch_size": 2, "spatial_merge_size": 1,
                "temporal_patch_size": 1, "num_position_embeddings": 16,
            ],
        ]
        return try JSONSerialization.data(withJSONObject: config)
    }

    func testPublished27BTopologyKeepsWeightsPacked() throws {
        // Shapes and counts from the published pack's safetensors header; no weight download/eval.
        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        let templates = try XCTUnwrap(config["modules"] as? [[String: Any]])
        var records = Array(templates.prefix(2))
        for layer in 0 ..< 64 {
            let sourceLayer = (layer + 1) % 4 == 0 ? 1 : 0
            let prefix = "model.layers.\(sourceLayer)."
            for var record in templates where (record["path"] as? String)?.hasPrefix(prefix) == true
            {
                let path = try XCTUnwrap(record["path"] as? String)
                record["path"] = "model.layers.\(layer)." + path.dropFirst(prefix.count)
                records.append(record)
            }
        }
        config["modules"] = records.map { record in
            var record = record
            record["block"] = 1024
            return record
        }
        var text = try XCTUnwrap(config["text_config"] as? [String: Any])
        text.merge([
            "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "vocab_size": 248320, "linear_num_value_heads": 48, "linear_num_key_heads": 16,
            "full_attention_interval": 4,
            "rope_parameters": ["rope_theta": 10_000_000, "mrope_section": [11, 11, 10]],
        ]) { _, new in new }
        config["text_config"] = text
        var vision = try XCTUnwrap(config["vision_config"] as? [String: Any])
        vision.merge([
            "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
            "out_hidden_size": 5120, "num_heads": 16, "patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 2304,
        ]) { _, new in new }
        config["vision_config"] = vision
        let model = try MLXVLM.PrismHadamardQwen35(
            configuration: JSONSerialization.data(withJSONObject: config))
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(records.count, 402)
        XCTAssertEqual(parameters.count, 2390)
        XCTAssertEqual(parameters.values.reduce(0) { $0 + $1.size }, 2_589_078_768)
        XCTAssertEqual(
            parameters["language_model.model.embed_tokens.weight"]?.shape, [248320, 320])
        XCTAssertEqual(
            parameters["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"]?.shape,
            [10240, 320])
        XCTAssertEqual(
            parameters["language_model.model.layers.63.self_attn.q_proj.weight"]?.shape,
            [12288, 320])
        XCTAssertEqual(
            parameters["vision_tower.patch_embed.proj.weight"]?.shape, [1152, 2, 16, 16, 3])
    }

    private func signs(width: Int) -> MLXArray {
        MLXArray((0 ..< width).map { Float(($0 / 3) % 2 == 0 ? 1 : -1) })
    }

    // Independent dense Walsh matrix, not a call to the transform under test.
    private func referenceTransform(_ x: MLXArray, signs: MLXArray, inverse: Bool = false)
        -> MLXArray
    {
        let block = 512
        let h = MLXArray(
            (0 ..< block * block).map { index -> Float in
                let parity = ((index / block) & (index % block)).nonzeroBitCount % 2
                return (parity == 0 ? 1 : -1) / sqrt(Float(block))
            }
        ).reshaped(block, block)
        var result = x.asType(.float32)
        if !inverse { result = result * signs }
        result = matmul(result.reshaped(-1, block), h).reshaped(x.shape)
        if inverse { result = result * signs }
        return result.asType(x.dtype)
    }

    private func assertClose(
        _ a: MLXArray, _ b: MLXArray, tolerance: Float = 0.005, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, file: file, line: line)
        XCTAssertLessThanOrEqual(
            abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self), tolerance,
            file: file, line: line)
    }

    func testForwardAndInverseTransformsMatchIndependentReference() {
        let x = sin(MLXArray(0 ..< 3072).asType(.float32) * 0.01).reshaped(1, 3, 1024).asType(
            .float16)
        let signs = signs(width: 1024)
        let forward = prismHadamardTransform(x, block: 512, signs: signs)
        assertClose(forward, referenceTransform(x, signs: signs))
        assertClose(prismHadamardTransform(forward, block: 512, signs: signs, inverse: true), x)
        assertClose(
            prismHadamardTransform(x, block: 512, signs: signs, inverse: true),
            referenceTransform(x, signs: signs, inverse: true))
        assertClose(prismHadamardTransform(x, block: 0, signs: nil), x, tolerance: 0)
    }

    func testPackedLinearAndEmbeddingPreserveTransformsAndCheckpointKeys() throws {
        let dense = (sin(MLXArray(0 ..< 2048).asType(.float32) * 0.03) * 0.05).reshaped(4, 512)
            .asType(.float16)
        let (weight, scales, biases) = quantized(dense, groupSize: 128, bits: 2)
        let signs = signs(width: 512)
        let arrays = [
            "weight": weight, "scales": scales, "biases": try XCTUnwrap(biases), "signs": signs,
        ]
        let linear = PrismHadamardLinear(rows: 4, width: 512, block: 512)
        let embedding = PrismHadamardEmbedding(rows: 4, width: 512, block: 512)
        try linear.update(parameters: ModuleParameters.unflattened(arrays), verify: [.all])
        try embedding.update(parameters: ModuleParameters.unflattened(arrays), verify: [.all])
        let x = cos(MLXArray(0 ..< 1024).asType(.float32) * 0.03).reshaped(1, 2, 512).asType(
            .float16)
        let unpacked = dequantized(weight, scales: scales, biases: biases, groupSize: 128, bits: 2)
        assertClose(linear(x), matmul(referenceTransform(x, signs: signs), unpacked.T))
        let tokens = MLXArray([3, 0, 3, 1]).reshaped(2, 2)
        assertClose(
            embedding(tokens), referenceTransform(unpacked[tokens], signs: signs, inverse: true))
        assertClose(embedding.asLinear(x), linear(x))
        XCTAssertTrue(quantizeSingle(layer: linear, groupSize: 128, bits: 2) == nil)
        XCTAssertEqual(Set(linear.parameters().flattened().map(\.0)), Set(arrays.keys))
    }

    private func checkpoint(configuration data: Data) throws -> [String: MLXArray] {
        let config = try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data)
        let base = MLXVLM.Qwen35(config)
        var weights = Dictionary(uniqueKeysWithValues: base.parameters().flattened())
        let metadata = try JSONDecoder().decode(PrismHadamardConfiguration.self, from: data)
        for record in metadata.modules {
            let path = "language_model." + record.path
            let original = try XCTUnwrap(weights[path + ".weight"])
            let dense = (sin(MLXArray(0 ..< original.size).asType(.float32) * 0.03) * 0.02)
                .reshaped(original.shape).asType(.float16)
            let (weight, scales, biases) = quantized(dense, groupSize: 128, bits: 2)
            weights[path + ".weight"] = weight
            weights[path + ".scales"] = scales
            weights[path + ".biases"] = try XCTUnwrap(biases)
            weights[path + ".signs"] = signs(width: original.dim(1))
        }
        return weights
    }

    private func withCheckpoint(_ weights: [String: MLXArray], body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: weights, metadata: ["format": "mlx"],
            url: directory.appendingPathComponent("model.safetensors"))
        try body(directory)
    }

    func testBothRegistriesLoadVisionPackAndAgreeOnTextPrefillAndDecode() async throws {
        let data = try configuration()
        let weights = try checkpoint(configuration: data)
        let llm = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "prism_hadamard_qwen35")
        let vlm = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "prism_hadamard_qwen35")
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try withCheckpoint(weights) { directory in
            try loadWeights(
                modelDirectory: directory, model: llm,
                perLayerQuantization: base.perLayerQuantization)
            try loadWeights(
                modelDirectory: directory, model: vlm,
                perLayerQuantization: base.perLayerQuantization)
        }
        let reference = MLXLLM.Qwen35Model(
            try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data))
        var denseWeights = llm.sanitize(weights: weights)
        let metadata = try JSONDecoder().decode(PrismHadamardConfiguration.self, from: data)
        for record in metadata.modules {
            let path = "language_model." + record.path
            let packed = try XCTUnwrap(denseWeights[path + ".weight"])
            let scales = try XCTUnwrap(denseWeights.removeValue(forKey: path + ".scales"))
            let biases = try XCTUnwrap(denseWeights.removeValue(forKey: path + ".biases"))
            let signs = try XCTUnwrap(denseWeights.removeValue(forKey: path + ".signs"))
            denseWeights[path + ".weight"] = referenceTransform(
                dequantized(packed, scales: scales, biases: biases, groupSize: 128, bits: 2),
                signs: signs, inverse: true)
        }
        try reference.update(parameters: ModuleParameters.unflattened(denseWeights), verify: [.all])
        let referenceCache = try reference.newCache(parameters: nil)
        let llmCache = try llm.newCache(parameters: nil)
        let vlmCache = try vlm.newCache(parameters: nil)
        var state: LMOutput.State?
        for tokens in [MLXArray([1, 2, 3]).reshaped(1, 3), MLXArray([4]).reshaped(1, 1)] {
            let a = llm(LMInput.Text(tokens: tokens), cache: llmCache, state: nil).logits
            let result = vlm(LMInput.Text(tokens: tokens), cache: vlmCache, state: state)
            state = result.state
            XCTAssertTrue(isFinite(a).all().item(Bool.self))
            assertClose(a, result.logits, tolerance: 0.02)
            assertClose(a, reference(tokens, cache: referenceCache), tolerance: 0.02)
        }
        // Loading/preparation must not turn transformed projections into ordinary fused linears.
        for model in [llm, vlm] {
            let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            XCTAssertTrue(
                leaves["language_model.model.layers.0.linear_attn.in_proj_qkv"]
                    is PrismHadamardLinear)
            XCTAssertTrue(leaves["language_model.model.embed_tokens"] is PrismHadamardEmbedding)
        }
        XCTAssertFalse(llm.parameters().flattened().contains { $0.0.hasPrefix("vision_tower.") })
        XCTAssertTrue(vlm.parameters().flattened().contains { $0.0.hasPrefix("vision_tower.") })
    }

    func testTextOnlyLoaderDoesNotReadVisionTensorData() throws {
        let data = try configuration()
        let weights = try checkpoint(configuration: data)
        let model = try MLXLLM.PrismHadamardQwen35(configuration: data)
        let textWeights = weights.filter { !$0.key.hasPrefix("vision_tower.") }
        try withCheckpoint(textWeights) { directory in
            let visionURL = directory.appendingPathComponent("model-vision.safetensors")
            try save(
                arrays: weights.filter { $0.key.hasPrefix("vision_tower.") }, url: visionURL)
            let handle = try FileHandle(forUpdating: visionURL)
            defer { try? handle.close() }
            let headerLength = try XCTUnwrap(handle.read(upToCount: 8)).withUnsafeBytes {
                $0.loadUnaligned(as: UInt64.self).littleEndian
            }
            // Keep the vision header but remove its data: any tensor read would fail.
            try handle.truncate(atOffset: 8 + headerLength)

            try loadWeights(modelDirectory: directory, model: model)
        }
        let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(Set(loaded.keys), Set(textWeights.keys))
        for (name, expected) in textWeights {
            assertClose(try XCTUnwrap(loaded[name]), expected, tolerance: 0)
        }
    }

    func testVisionPackProcessesAnImageAndContinuesWithText() throws {
        let data = try configuration()
        let model = try MLXVLM.PrismHadamardQwen35(configuration: data)
        try withCheckpoint(checkpoint(configuration: data)) {
            try loadWeights(modelDirectory: $0, model: model)
        }
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(
            text: .init(tokens: MLXArray([1, 13, 15, 15, 15, 15, 2]).reshaped(1, 7)),
            image: .init(pixels: MLXArray.ones([4, 3 * 2 * 2]), frames: [THW(1, 2, 2)]))
        let prepared = try model.prepare(input, cache: cache, state: nil, prefill: .init())
        guard case .logits(let output) = prepared else { return XCTFail("expected vision logits") }
        XCTAssertTrue(isFinite(output.logits).all().item(Bool.self))
        XCTAssertEqual(output.logits.dim(-1), 16)
        let continued = model(
            .init(tokens: MLXArray([3]).reshaped(1, 1)), cache: cache, state: output.state)
        XCTAssertTrue(isFinite(continued.logits).all().item(Bool.self))
        XCTAssertEqual(continued.logits.shape, [1, 1, 16])
    }

    func testSchemaOneTextPackLoadsWithoutVision() throws {
        let data = try configuration(schema: 1)
        let weights = try checkpoint(configuration: data)
        let textWeights = Dictionary(
            uniqueKeysWithValues: weights.compactMap { key, value in
                key.hasPrefix("language_model.")
                    ? (String(key.dropFirst("language_model.".count)), value) : nil
            })
        let model = try MLXLLM.PrismHadamardQwen35(configuration: data)
        try withCheckpoint(textWeights) { try loadWeights(modelDirectory: $0, model: model) }
        XCTAssertThrowsError(try MLXVLM.PrismHadamardQwen35(configuration: data))
    }

    func testMalformedMetadataFailsBeforeInference() throws {
        let data = try configuration()
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for (key, value) in [
            ("schema_version", 3 as Any), ("gdn_activation_layout", "ungrouped" as Any),
            ("quantization", ["bits": 4, "group_size": 128, "mode": "affine"] as Any),
        ] {
            var invalid = original
            invalid[key] = value
            XCTAssertThrowsError(
                try MLXLLM.PrismHadamardQwen35(
                    configuration: JSONSerialization.data(withJSONObject: invalid)))
        }
        for change in ["duplicate", "path", "block", "dtype", "embedding"] {
            var invalid = original
            var records = try XCTUnwrap(invalid["modules"] as? [[String: Any]])
            switch change {
            case "duplicate": records.append(records[0])
            case "path": records[0]["path"] = "model.missing"
            case "block": records[0]["block"] = 1024
            case "dtype": records[0]["dtype"] = "float32"
            default: records[0]["embedding"] = false
            }
            invalid["modules"] = records
            XCTAssertThrowsError(
                try MLXLLM.PrismHadamardQwen35(
                    configuration: JSONSerialization.data(withJSONObject: invalid)))
        }
    }

    func testMalformedWeightsAbortTheSharedLoader() throws {
        let data = try configuration()
        let original = try checkpoint(configuration: data)
        let prefix = "language_model.model.embed_tokens."
        for mutation in [
            "missingSigns", "invalidSigns", "signShape", "packedShape", "scaleDType", "undeclared",
        ] {
            var weights = original
            switch mutation {
            case "missingSigns": weights.removeValue(forKey: prefix + "signs")
            case "invalidSigns": weights[prefix + "signs"] = MLXArray.zeros([512])
            case "signShape": weights[prefix + "signs"] = MLXArray.ones([1, 512])
            case "packedShape":
                weights[prefix + "weight"] = MLXArray.zeros([16, 64], dtype: .uint32)
            case "scaleDType":
                weights[prefix + "scales"] = weights[prefix + "scales"]?.asType(.float32)
            default: weights["language_model.unknown.scales"] = MLXArray.ones([1])
            }
            let model = try MLXLLM.PrismHadamardQwen35(configuration: data)
            try withCheckpoint(weights) { directory in
                XCTAssertThrowsError(try loadWeights(modelDirectory: directory, model: model)) {
                    error in
                    XCTAssertTrue(error is PrismHadamardError)
                }
            }
        }
    }

    func testProcessorUsesQwenVisionConfiguration() async throws {
        let data = try configuration()
        let context = VLMProcessorLoadingContext(
            modelId: "prism-ml/test", modelType: "prism_hadamard_qwen35", configurationData: data)
        let fallback = try XCTUnwrap(
            VLMProcessorLoadingRegistry.shared.fallbackProcessorConfiguration(for: context))
        XCTAssertEqual(fallback.processorType, "Qwen3VLProcessor")
        XCTAssertEqual(
            try VLMProcessorLoadingRegistry.shared.processorType(
                for: context, declaredProcessorType: nil), "Qwen3VLProcessor")
    }
}
