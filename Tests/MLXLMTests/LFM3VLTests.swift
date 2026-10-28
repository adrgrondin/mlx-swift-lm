// Copyright © 2026 Apple Inc.

import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

final class LFM3VLTests: XCTestCase {
    private func config(tiedWordEmbeddings: Bool = true, patchSize: Int = 2) throws
        -> LFM3VLConfiguration
    {
        var text = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(lfm3TestConfig(window: 8)))
                as? [String: Any])
        text["tie_word_embeddings"] = tiedWordEmbeddings
        let json: [String: Any] = [
            "text_config": text, "image_token_id": 30, "video_token_id": 29,
            "projector_hidden_size": 16,
            "vision_config": [
                "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "patch_size": patchSize,
                "layer_types": ["conv", "full_attention"],
            ],
        ]
        let config = try JSONDecoder().decode(
            LFM3VLConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
        try config.validateModelConfiguration()
        return config
    }

    func testSpatialConvolutionUsesRasterNeighborsNotMergeBlockNeighbors() throws {
        try Device.withDefaultDevice(.cpu) {
            let conv = LFM3VisionConv(try config().visionConfiguration)
            let weights = (1 ... 9).map { Float($0) / 10 }
            let kernel = tiled(MLXArray(weights).reshaped(1, 3, 3, 1), repetitions: [16, 1, 1, 1])
            try conv.update(
                parameters: ModuleParameters.unflattened([
                    "in_proj.weight": concatenated([eye(16), eye(16), eye(16)]),
                    "out_proj.weight": eye(16), "spatial_conv.weight": kernel,
                ]), verify: [.noUnusedKeys])
            var coordinates: [(Int, Int)] = []
            for h in stride(from: 0, to: 4, by: 2) {
                for w in stride(from: 0, to: 6, by: 2) {
                    for dh in 0 ..< 2 { for dw in 0 ..< 2 { coordinates.append((h + dh, w + dw)) } }
                }
            }
            let input = coordinates.flatMap { h, w in
                Array(repeating: Float(h * 6 + w + 1) / 24, count: 16)
            }
            let output = conv(MLXArray(input).reshaped(24, 16), shapes: [THW(1, 4, 6)]).asArray(
                Float.self)
            for (index, coordinate) in coordinates.enumerated() {
                let (h, w) = coordinate
                var sum: Float = 0
                for dh in -1 ... 1 {
                    for dw in -1 ... 1
                    where (0 ..< 4).contains(h + dh) && (0 ..< 6).contains(w + dw) {
                        let value = Float((h + dh) * 6 + w + dw + 1) / 24
                        sum += value * value * weights[(dh + 1) * 3 + dw + 1]
                    }
                }
                XCTAssertEqual(output[index * 16], sum * Float(h * 6 + w + 1) / 24, accuracy: 1e-5)
            }
        }
    }

    func testVisionTowerIsolatesImagesAndTemporalGroups() throws {
        try Device.withDefaultDevice(.cpu) {
            let tower = LFM3VisionModel(try config().visionConfiguration)
            let a = MLXArray((0 ..< 576).map { Float($0 % 19) / 19 }).reshaped(24, 24)
            let b = MLXArray((0 ..< 384).map { Float($0 % 13) / 13 }).reshaped(16, 24)
            let separateA = try tower(a, grids: [THW(1, 4, 6)])
            let separateB = try tower(b, grids: [THW(1, 4, 4)])
            let mixed = try tower(concatenated([a, b]), grids: [THW(1, 4, 6), THW(1, 4, 4)])
            XCTAssertLessThan(abs(mixed[0 ..< 24] - separateA).max().item(Float.self), 1e-5)
            XCTAssertLessThan(abs(mixed[24...] - separateB).max().item(Float.self), 1e-5)
            let video = try tower(concatenated([a, a * 2]), grids: [THW(2, 4, 6)])
            XCTAssertLessThan(abs(video[0 ..< 24] - separateA).max().item(Float.self), 1e-5)
            let separateSecondFrame = try tower(a * 2, grids: [THW(1, 4, 6)])
            XCTAssertLessThan(abs(video[24...] - separateSecondFrame).max().item(Float.self), 1e-5)
        }
    }

    func testQuantizedVisionPatchifierMatchesDequantizedReferenceForSignedPixels() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try config(patchSize: 4).visionConfiguration
            let tower = withRandomState(MLXRandom.RandomState(seed: 37)) {
                LFM3VisionModel(c)
            }
            var referenceWeights = Dictionary(uniqueKeysWithValues: tower.parameters().flattened())
            quantize(
                model: tower, groupSize: 32, bits: 4,
                filter: { path, _ in path == "patchifier" })
            let patchifier = try XCTUnwrap(tower.patchifier as? QuantizedLinear)
            referenceWeights["patchifier.weight"] = dequantized(
                patchifier.weight, scales: patchifier.scales, biases: patchifier.biases,
                groupSize: 32, bits: 4)
            let reference = LFM3VisionModel(c)
            try reference.update(
                parameters: ModuleParameters.unflattened(referenceWeights), verify: .all)
            let pixels = MLXArray((0 ..< 384).map { Float(($0 % 17) - 8) / 10 }).reshaped(4, 96)
            for dtype in [DType.float32, .bfloat16] {
                let input = pixels.asType(dtype)
                let expected = try reference(input, grids: [THW(1, 2, 2)])
                let actual = try tower(input, grids: [THW(1, 2, 2)])
                XCTAssertGreaterThan(abs(expected).max().item(Float.self), 0.1)
                XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 2e-4)
            }
        }
    }

    func testFactoryRejectsD1CheckpointsWithoutRejectingOrdinaryLFM3VL() async throws {
        let base = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(config())) as? [String: Any])
        for identity: [String: Any] in [
            ["architectures": ["D1Model"]],
            ["auto_map": ["AutoModel": "modeling_d1_lfm3_vl.D1Model"]],
            [
                "architectures": ["Lfm3VlForConditionalGeneration"],
                "auto_map": ["AutoModel": "modeling_d1_lfm3_vl.D1Model"],
            ],
        ] {
            let json = base.merging(identity) { _, value in value }
            do {
                _ = try await VLMTypeRegistry.shared.createModel(
                    configuration: JSONSerialization.data(withJSONObject: json),
                    modelType: "lfm3_vl")
                XCTFail("D1 metadata must not create a causal chat model")
            } catch let ModelFactoryError.invalidConfiguration(message) {
                XCTAssertTrue(message.contains("D1Model"), message)
            }
        }
        let ordinary = base.merging([
            "architectures": ["Lfm3VlForConditionalGeneration"],
            "auto_map": ["AutoModel": "modeling_lfm3_vl.Lfm3VlForConditionalGeneration"],
        ]) { _, value in value }
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: JSONSerialization.data(withJSONObject: ordinary), modelType: "lfm3_vl")
        XCTAssertTrue(model is LFM3VL)
    }

    func testProjectorNormalizesIndividualPatchesBeforeMergingAndUsesExactGELU() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try config()
            let projector = LFM3MultiModalProjector(c)
            let select = MLXArray.zeros([16, 64])
            for i in 0 ..< 16 { select[i, i] = MLXArray(Float(1)) }
            try projector.update(
                parameters: ModuleParameters.unflattened([
                    "linear_1.weight": select, "linear_1.bias": MLXArray.zeros([16]),
                    "linear_2.weight": eye(16), "linear_2.bias": MLXArray.zeros([16]),
                ]), verify: [.noUnusedKeys])
            let values = (0 ..< 64).map { Float($0) }
            let actual = projector(MLXArray(values).reshaped(4, 16)).asArray(Float.self)
            let mean: Float = 7.5
            let variance = (0 ..< 16).reduce(Float(0)) { $0 + pow(Float($1) - mean, 2) } / 16
            for i in 0 ..< 16 {
                let normalized = (Float(i) - mean) / sqrt(variance + c.projectorNormEps)
                let expected = 0.5 * normalized * (1 + erf(normalized / sqrt(Float(2))))
                XCTAssertEqual(actual[i], expected, accuracy: 1e-6)
            }
        }
    }

    func testMultimodalPrefillMatchesManualFeatureInsertionAcrossChunks() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LFM3VL(try config())
            let pixels = MLXArray((0 ..< 576).map { Float($0 % 17) / 17 }).reshaped(24, 24)
            let tokens = MLXArray([Int32(1), 30, 30, 30, 30, 30, 30, 2]).reshaped(1, -1)
            let embeddings = model.languageModel.embed(tokens)
            embeddings[0, 1 ..< 7, 0...] = model.projector(
                try model.visionTower(pixels, grids: [THW(1, 4, 6)]))
            let full = model.languageModel.logits(embeddings: embeddings, cache: nil)
            let expected = full[0..., (tokens.dim(1) - 1) ..< tokens.dim(1), 0...]
            let input = LMInput(
                text: .init(tokens: tokens), image: .init(pixels: pixels, frames: [THW(1, 4, 6)]))
            let output = try model.prepare(
                input, cache: model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 3))
            guard case .logits(let output) = output else {
                return XCTFail("Expected multimodal logits")
            }
            XCTAssertLessThan(
                abs(
                    output.logits[0..., (output.logits.dim(1) - 1) ..< output.logits.dim(1), 0...]
                        - expected
                ).max().item(Float.self), 1e-4)
            let bad = LMInput(
                text: .init(tokens: MLXArray([Int32(30)]).reshaped(1, 1)),
                image: .init(pixels: pixels, frames: [THW(1, 4, 6)]))
            XCTAssertThrowsError(
                try model.prepare(
                    bad, cache: model.newCache(parameters: nil), state: nil, prefill: .init()))
        }
    }

    func testOriginalAndMLXNamespacesLoadStrictlyWithIdempotentConvolutionLayouts() throws {
        try Device.withDefaultDevice(.cpu) {
            for tied in [true, false] {
                let model = LFM3VL(try config(tiedWordEmbeddings: tied))
                let runtime = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                var original: [String: MLXArray] = [:]
                for (name, value) in runtime {
                    var key = "model." + name
                    var tensor = value
                    if name.hasPrefix("language_model.model.") {
                        key = name.replacingOccurrences(
                            of: "language_model.model.", with: "model.language_model.")
                    } else if name.hasPrefix("language_model.lm_head.") {
                        key = name.replacingOccurrences(
                            of: "language_model.lm_head.", with: "lm_head.")
                    }
                    if name.hasSuffix("conv.conv.weight") { tensor = value.transposed(0, 2, 1) }
                    if name.hasSuffix("conv.spatial_conv.weight") {
                        tensor = value.transposed(0, 3, 1, 2)
                    }
                    original[key] = tensor
                }
                if tied { original["lm_head.weight"] = MLXArray.ones([32, 16]) }
                let settings = BaseConfiguration.PerLayerQuantization(perLayerQuantization: [
                    "model.language_model.embed_tokens": .quantize(.init(groupSize: 32, bits: 8)),
                    "lm_head": .quantize(.init(groupSize: 32, bits: 8)),
                ])
                let prepared = try model.prepareCheckpoint(
                    .init(weights: original, perLayerQuantization: settings))
                try model.update(
                    parameters: ModuleParameters.unflattened(prepared.weights), verify: .all)
                let twice = try model.prepareCheckpoint(prepared)
                try model.update(
                    parameters: ModuleParameters.unflattened(twice.weights), verify: .all)
                XCTAssertEqual(
                    twice.perLayerQuantization?.quantization(
                        layer: "language_model.model.embed_tokens")?.bits, 8)
                XCTAssertEqual(
                    twice.perLayerQuantization?.quantization(layer: "language_model.lm_head")?.bits,
                    tied ? nil : 8)
                let tokens = MLXArray([Int32(1), 2]).reshaped(1, 2)
                XCTAssertTrue(isFinite(model(tokens, cache: nil)).all().item(Bool.self))
            }
        }
    }

    func testVideoSamplingMatchesReleasedSourceIndicesIncludingOddCounts() throws {
        for (total, fps, expected) in [
            (105, 30.0, [0, 17, 35, 52, 69, 87, 104]),
            (30, 30.0, [0, 10, 19, 29]),
            (3, 30.0, [0, 1, 2]),
        ] {
            XCTAssertEqual(
                try LFM3VLProcessor.sampleIndices(
                    total: total, sourceFPS: fps, processing: .init()), expected)
        }
        XCTAssertThrowsError(
            try LFM3VLProcessor.sampleIndices(
                total: 30, sourceFPS: 30, processing: .init(targetFramesPerSecond: .nan)))
        XCTAssertEqual(
            try LFM3VLProcessor.targetSize(
                height: 80, width: 80, factor: 32, minPixels: 1, maxPixels: 10000
            ).0, 64)
        XCTAssertEqual(
            try LFM3VLProcessor.targetSize(
                height: 1080, width: 1920, frames: 60, factor: 32,
                minPixels: 4096, maxPixels: 25_165_824
            ).0, 480)
    }

    private struct ByteTokenizer: Tokenizer {
        let prompt: String
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
        }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            encode(text: prompt, addSpecialTokens: false)
        }
    }

    func testVideoProcessorPadsOnlyPatchificationAndWritesPairTimestamps() async throws {
        let json = """
            {"image_processor":{"size":{"shortest_edge":1024,"longest_edge":8192}},
             "video_processor":{"size":{"shortest_edge":1024,"longest_edge":8192}}}
            """
        let c = try JSONDecoder().decode(LFM3VLProcessorConfiguration.self, from: Data(json.utf8))
        let tokenizer = ByteTokenizer(prompt: "<|vision_start|><|video|><|vision_end|>describe")
        let processor = LFM3VLProcessor(c, tokenizer: tokenizer)
        let image = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32))
        let frames = (0 ..< 105).map {
            VideoFrame(image: .ciImage(image), timeStamp: CMTime(value: Int64($0), timescale: 30))
        }
        let prepared = try await processor.prepare(
            input: UserInput(prompt: "describe", videos: [.frames(frames)]))
        let text = tokenizer.decode(tokenIds: prepared.text.tokens.asArray(Int.self))
        let expected =
            ["0.3", "1.5", "2.6", "3.5"].map {
                "<\($0) seconds><|vision_start|><|video|><|vision_end|>"
            }.joined() + "describe"
        XCTAssertEqual(text, expected)
        let video = try XCTUnwrap(prepared.video)
        XCTAssertEqual(video.frames?.first?.t, 4)
        XCTAssertEqual(video.pixels.shape, [16, 1536])
        XCTAssertEqual(video.pixels.min().item(Float.self), -1, accuracy: 1e-6)
        XCTAssertEqual(video.pixels.max().item(Float.self), -1, accuracy: 1e-6)
    }

    func testIndexedAssetSamplingCanDecodeLastSourceFrame() async throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "1080p_30", withExtension: "mov"))
        let sampled = try await MediaProcessing.indexedVideoFrames(.url(url)) { total, _ in
            XCTAssertEqual(total, 150)
            return [0, total - 1]
        }
        XCTAssertEqual(sampled.images.count, 2)
        XCTAssertEqual(sampled.seconds[0], 0)
        XCTAssertEqual(sampled.seconds[1], 149.0 / 30, accuracy: 1e-6)
    }
}
