// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite(.serialized)
struct Inkling2Tests {
    private func tinyModel(fullHeadDim: Int = 16) throws -> Inkling2 {
        let json = """
            {
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 4,
                "vocab_size": 64, "unpadded_vocab_size": 60,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": \(fullHeadDim),
                "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 16,
                "sliding_window_size": 4, "local_layer_ids": [0, 1, 3], "d_rel": 4,
                "rel_extent": 3, "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
                "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 5.0, "rel_l2_clip_threshold": 2.0,
                "sconv_kernel_size": 3, "n_layers_per_attn_res_block": 1,
                "n_routed_experts": 4, "num_experts_per_tok": 2, "intermediate_size": 32,
                "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
                "final_logit_softcapping": 30.0,
                "loop_config": {"core_start": 1, "core_end": 3, "num_passes": 3, "gate_rank": 4}
              },
              "vision_config": {
                "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                "native_patch_size": 2, "adapter_merge_size": 2, "posemb_grid_size": 4,
                "projection_hidden_dim": 16, "projector_bias": false, "mlp_ratio": 2.0,
                "layer_norm_eps": 1e-5
              }
            }
            """
        let config = try JSONDecoder().decode(Inkling2Configuration.self, from: Data(json.utf8))
        return try Inkling2(config)
    }

    @Test func loopCachesMatchUncachedForwardAfterSlidingWrap() throws {
        let model = try tinyModel()
        #expect(model.configuration.text.visits == [0, 1, 2, 1, 2, 1, 2, 3])
        let cache = try model.newCache(parameters: nil).map { $0 as! CacheList }
        #expect((cache[1][0] as? RotatingKVCache) !== (cache[3][0] as? RotatingKVCache))
        #expect((cache[1][1] as? ArraysCache) !== (cache[3][1] as? ArraysCache))
        #expect(cache[2][0] is KVCacheSimple)
        #expect(cache[3][0] is RotatingKVCache)
        let ids = MLXArray([1, 12, 13, 24, 15, 6, 20, 10, 16, 21, 22, 23]).reshaped(1, 12)
        let expected = model(ids, cache: nil)
        #expect(expected.shape == [1, 12, 60])
        for range in [0 ..< 3, 3 ..< 6, 6 ..< 7, 7 ..< 8, 8 ..< 11, 11 ..< 12] {
            let output = model(ids[0..., range], cache: cache)
            let error = abs(output - expected[0..., range, 0...]).max().item(Float.self)
            #expect(error < 2e-3)
        }
        #expect(cache.allSatisfy { $0[0].offset == 12 })
        #expect(cache.allSatisfy { $0[1].state.count == 4 })
    }

    @Test func boundedAttentionKeepsPinnedPositionsAcrossPrefillAndDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            try checkPinnedPositions()
        }
    }

    private func checkPinnedPositions() throws {
        let model = try tinyModel()
        for (index, layer) in model.backbone.languageModel.layers.enumerated() {
            let projection = layer.attention.relative.proj
            projection[.ellipsis] = MLXRandom.normal(
                projection.shape, key: MLXRandom.key(UInt64(index)))
        }
        let parameters = GenerateParameters(
            kvCache: .init(
                capacity: try .init(maxTokens: 5, preservedPrefixTokens: 2)))
        let status = try model.cacheStatus(parameters: parameters)
        #expect(status.capacityAppliedLayerCount == 3)
        let chunked = try model.newCache(parameters: parameters)
        let whole = try model.newCache(parameters: parameters)
        let fullAttention = try #require((chunked[2] as? CacheList)?[0] as? RotatingKVCache)
        #expect(fullAttention.keepCount == 2)
        #expect((chunked[0] as? CacheList)?[0].maxSize == 4)
        let ids = MLXArray([1, 12, 13, 24, 15, 6, 20, 10, 16, 21, 22, 23]).reshaped(1, 12)
        let expected = model(ids, cache: whole)
        eval(expected)
        for range in [0 ..< 1, 1 ..< 4, 4 ..< 7, 7 ..< 8, 8 ..< 9, 9 ..< 11, 11 ..< 12] {
            let output = model(ids[0..., range], cache: chunked)
            let error = abs(output - expected[0..., range, 0...]).max().item(Float.self)
            #expect(error < 2e-3, "Logit error for \(range): \(error)")
        }
        #expect(fullAttention.offset == 12)
        #expect(fullAttention.currentKeyValues()?.keys.dim(2) == 5)
        #expect(fullAttention.evictedTokenCount == 7)
    }

    @Test func invalidCacheCapacityIsRejected() throws {
        let model = try tinyModel()
        #expect(throws: KVCacheConfigurationError.invalidCapacity(0)) {
            try model.newCache(parameters: .init(maxKVSize: 0))
        }
    }

    @Test func affineCacheDecodesAfterConversion() throws {
        let model = try tinyModel(fullHeadDim: 64)
        var cache = try model.newCache(parameters: nil)
        eval(model(MLXArray([1, 12, 13]).reshaped(1, 3), cache: cache))
        let result = try applyKVCacheConfiguration(
            cache: &cache, configuration: .init(strategy: .affine(.fourBit)))
        #expect(result.convertedLayerCount == 3)
        let output = model(MLXArray([24]).reshaped(1, 1), cache: cache)
        #expect(output.shape == [1, 1, 60])
        #expect(all(isFinite(output)).item(Bool.self))
        let fullAttention = try #require((cache[2] as? CacheList)?[0] as? QuantizedKVCache)
        #expect(fullAttention.offset == 4)
    }

    @Test func imagePreprocessingCompositesAndPads() throws {
        let bounds = CGRect(x: 0, y: 0, width: 9, height: 9)
        let clear = CIImage(color: .clear).cropped(to: bounds)
        let red = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 8, width: 9, height: 1))
        let (pixels, frame) = try Inkling2Processor.preprocess(
            red.composited(over: clear), block: 4)
        #expect(frame.h == 12 && frame.w == 12)
        #expect(pixels[0, 0].asArray(Float.self) == [1, -1, -1])
        #expect(abs(pixels[1, 0, 0].item(Float.self) - 0.6) < 1e-6)
        #expect(abs(pixels[1, 8, 0].item(Float.self) - 0.2) < 1e-6)
        #expect(abs(pixels[9, 0, 0].item(Float.self) - (-1 - 2.0 / 255)) < 1e-6)
    }

    @Test func imageBudgetIncludesAlignmentAndPreservesSmallImages() throws {
        func image(_ width: Int, _ height: Int) -> CIImage {
            CIImage(color: .red).cropped(to: CGRect(x: -3, y: 7, width: width, height: height))
        }
        let small = image(39, 32)
        #expect(
            try Inkling2Processor.fitImage(small, block: 40, maxPixels: 1600).extent == small.extent
        )
        #expect(throws: VLMError.self) {
            try Inkling2Processor.fitImage(small, block: 40, maxPixels: 1599)
        }
        for (width, height, budget, blocks) in [
            (41, 40, 1600, 1), (10, 301, 3200, 2), (301, 10, 3200, 2),
        ] {
            let bounded = try Inkling2Processor.fitImage(
                image(width, height), block: 40, maxPixels: budget)
            let (_, frame) = try Inkling2Processor.preprocess(bounded, block: 40)
            #expect((frame.w / 40) * (frame.h / 40) == blocks)
        }
        let large = try Inkling2Processor.fitImage(image(4000, 3000), block: 40, maxPixels: nil)
        let bounds = large.extent.integral
        #expect(ceil(bounds.width / 40) * ceil(bounds.height / 40) <= 4096)
    }

    @Test func visionProducesOneRowPerMergeBlock() throws {
        let model = try tinyModel()
        let features = model.backbone.visionTower(.zeros([8, 12, 3]))
        #expect(features.shape == [6, 32])
        #expect(all(isFinite(features)).item(Bool.self))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"] != nil))
    func publishedCheckpointMatchesReference() throws {
        let path = try #require(ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"])
        let directory = URL(filePath: path)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let model = try Inkling2(JSONDecoder().decode(Inkling2Configuration.self, from: data))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try loadWeights(
            modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization
        )
        #expect(model.kvHeads.count == 48)
        // mlx-lm #2086, thinking effort 0. No sampling, one prefill forward.
        let tokens = [
            200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010,
            200000, 200004, 1775, 149884, 149876, 151451, 149879, 151194, 30, 200010,
        ]
        let cache = try model.newCache(parameters: nil)
        var input = MLXArray(tokens).reshaped(1, -1)
        for expected in [200001, 200008, 371] {
            let logits = model(input, cache: cache)[0, -1]
            #expect(argMax(logits).item(Int.self) == expected)
            input = MLXArray([expected]).reshaped(1, 1)
        }

        // Python reference: red, green, and half-transparent blue bands in an 81×43 image.
        func stripe(_ color: CIColor, y: Int, height: Int) -> CIImage {
            CIImage(color: color).cropped(to: CGRect(x: 0, y: y, width: 81, height: height))
        }
        let image = stripe(.red, y: 23, height: 20).composited(
            over: stripe(.green, y: 3, height: 20).composited(
                over: stripe(
                    CIColor(red: 0, green: 0, blue: 1, alpha: 128.0 / 255), y: 0, height: 3)))
        let (pixels, _) = try Inkling2Processor.preprocess(image, block: 40)
        let features = model.backbone.visionTower(pixels).asType(.float32)
        let samples = features[0..., MLXArray([0, 64, 1024])]
        let reference: [Float] = [
            -0.11572, 0.52734, 0.22754, -0.08545, 0.25977, -0.04248,
            -0.09912, 0.26953, 0.03418, 0.00687, -0.34180, 0.05762,
            -0.11182, -0.20410, 0.05615, -0.16406, 0.16504, 0.16504,
        ]
        #expect(abs(samples - MLXArray(reference).reshaped(6, 3)).max().item(Float.self) < 0.035)
    }
}
