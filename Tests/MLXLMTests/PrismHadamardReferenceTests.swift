// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class PrismHadamardReferenceTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Tensor: Decodable {
            let shape: [Int]
            let dtype: String
            let pattern: [Double]

            func array() throws -> MLXArray {
                let count = shape.reduce(1, *)
                _ = try XCTUnwrap(pattern.first)
                let values = (0 ..< count).map { pattern[$0 % pattern.count] }
                if dtype == "uint32" {
                    return MLXArray(values.map { UInt32($0) }).reshaped(shape)
                }
                let type = try XCTUnwrap(["float16": DType.float16, "float32": .float32][dtype])
                return MLXArray(values.map { Float($0) }).reshaped(shape).asType(type)
            }
        }

        struct Case: Decodable {
            struct Step: Decodable {
                let tokens: [Int]
                let logits: [Float]
            }
            let tied: Bool
            let steps: [Step]
            let fullLogits: [Float]

            enum CodingKeys: String, CodingKey {
                case tied, steps
                case fullLogits = "full_logits"
            }
        }

        let weights: [String: Tensor]
        let cases: [Case]
    }

    func testPythonReferencePrefillAndCachedDecode() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "prism_hadamard_reference", withExtension: "json"))
        let data = try Data(contentsOf: url)
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let originalConfig = try XCTUnwrap(root["config"] as? [String: Any])
        let originalWeights = try fixture.weights.mapValues { try $0.array() }

        for schema in [1, 2] {
            for testCase in fixture.cases {
                var config = originalConfig
                config["schema_version"] = schema
                config["tensor_namespace"] = schema == 1 ? "mlx-lm-text" : "mlx-vlm-qwen3_5"
                var text = try XCTUnwrap(config["text_config"] as? [String: Any])
                text["tie_word_embeddings"] = testCase.tied
                config["text_config"] = text
                config["tie_word_embeddings"] = testCase.tied
                let records = try XCTUnwrap(config["modules"] as? [[String: Any]])
                if testCase.tied {
                    config["modules"] = records.filter { $0["path"] as? String != "lm_head" }
                }
                let configData = try JSONSerialization.data(withJSONObject: config)
                var models: [any LanguageModel] = [
                    try MLXLLM.PrismHadamardQwen35(configuration: configData)
                ]
                if schema == 2 {
                    models.append(try MLXVLM.PrismHadamardQwen35(configuration: configData))
                }
                for model in models {
                    var weights = originalWeights.filter {
                        !testCase.tied || !$0.key.hasPrefix("language_model.lm_head.")
                    }
                    if schema == 1 {
                        weights = Dictionary(
                            uniqueKeysWithValues: weights.map {
                                (String($0.key.dropFirst("language_model.".count)), $0.value)
                            })
                    } else {
                        // The Python text reference has no vision weights; they are unused here.
                        weights.merge(
                            model.parameters().flattened().filter {
                                $0.0.hasPrefix("vision_tower.")
                            }
                        ) { current, _ in current }
                    }
                    let directory = FileManager.default.temporaryDirectory
                        .appendingPathComponent(UUID().uuidString)
                    try FileManager.default.createDirectory(
                        at: directory, withIntermediateDirectories: true)
                    defer { try? FileManager.default.removeItem(at: directory) }
                    try save(
                        arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
                    let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
                    try loadWeights(
                        modelDirectory: directory, model: model,
                        perLayerQuantization: base.perLayerQuantization)
                    model.train(false)
                    let cache = try model.newCache(parameters: nil)
                    var state: LMOutput.State?
                    let label =
                        "\(String(reflecting: type(of: model))), schema \(schema), tied \(testCase.tied)"
                    for (index, step) in testCase.steps.enumerated() {
                        let result = model(
                            .init(tokens: MLXArray(step.tokens).reshaped(1, -1)),
                            cache: cache, state: state)
                        state = result.state
                        assertLogits(
                            result.logits, expected: step.logits, count: step.tokens.count,
                            label: "\(label), step \(index)")
                    }
                    let tokens = testCase.steps.flatMap(\.tokens)
                    let full = model(
                        .init(tokens: MLXArray(tokens).reshaped(1, -1)), cache: nil, state: nil)
                    assertLogits(
                        full.logits, expected: testCase.fullLogits, count: tokens.count,
                        label: "\(label), uncached")
                }
            }
        }
    }

    private func assertLogits(
        _ actual: MLXArray, expected: [Float], count: Int, label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, [1, count, 16], label, file: file, line: line)
        XCTAssertTrue(isFinite(actual).all().item(Bool.self), label, file: file, line: line)
        let reference = MLXArray(expected).reshaped(actual.shape)
        let error = abs(actual.asType(.float32) - reference).max().item(Float.self)
        XCTAssertLessThanOrEqual(error, 0.001, label, file: file, line: line)
    }
}
