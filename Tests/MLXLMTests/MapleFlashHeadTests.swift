import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tests for the optional approximate FlashHead output head: metadata
/// validation, sanitizer reconciliation, cluster mapping/forced tokens/exact
/// subset scoring, and the exact-head fallback conditions.
final class MapleFlashHeadTests: XCTestCase {
    private func metadataJSON(
        scaledCentroids: Bool = true, nClusters: Int = 4, clusterSize: Int = 8,
        nProbes: Int = 2, forceTokens: [Int] = [31]
    ) -> String {
        """
        {
          "bits": 4,
          "cluster_size": \(clusterSize),
          "force_tokens": \(forceTokens),
          "group_size": 32,
          "head_bits": 4,
          "head_group_size": 32,
          "n_clusters": \(nClusters),
          "n_probes": \(nProbes),
          "scaled_centroids": \(scaledCentroids)
        }
        """
    }

    private func metadata(_ json: String) throws -> MapleFlashHeadMetadata {
        try JSONDecoder().decode(MapleFlashHeadMetadata.self, from: Data(json.utf8))
    }

    /// Tiny flash-bearing configuration: hidden 256, vocab 64, 8 clusters of 8.
    private func flashConfiguration(includeFlash: Bool = true) throws -> MapleConfiguration {
        let flash =
            includeFlash
            ? """
                ,"flash_head": {
                  "bits": 4, "cluster_size": 8, "force_tokens": [3],
                  "group_size": 64, "head_bits": 4, "head_group_size": 64,
                  "n_clusters": 8, "n_probes": 4, "scaled_centroids": true
                }
                """
            : ""
        let json = """
            {
              "model_type": "maple",
              "hidden_size": 256,
              "intermediate_size": 512,
              "moe_intermediate_size": 128,
              "num_hidden_layers": 2,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 128,
              "num_experts": 32,
              "num_experts_per_tok": 8,
              "first_k_dense_replace": 0,
              "partial_rotary_factor": 0.5,
              "vocab_size": 64,
              "sliding_window": 8,
              "layer_types": ["sliding_attention", "full_attention"]
              \(flash)
            }
            """
        return try JSONDecoder().decode(MapleConfiguration.self, from: Data(json.utf8))
    }

    func testMetadataValidation() throws {
        XCTAssertTrue(try metadata(metadataJSON()).isValid)
        XCTAssertFalse(try metadata(metadataJSON(scaledCentroids: false)).isValid)
        XCTAssertFalse(try metadata(metadataJSON(nClusters: 0)).isValid)
        XCTAssertFalse(try metadata(metadataJSON(clusterSize: 0)).isValid)
        XCTAssertFalse(try metadata(metadataJSON(nProbes: 0)).isValid)
        // resolvedProbes clamps to the cluster count
        XCTAssertEqual(try metadata(metadataJSON(nClusters: 4, nProbes: 512)).resolvedProbes, 4)
        // Defaults: force_tokens and n_probes are optional
        let minimal = """
            {"cluster_size": 8, "n_clusters": 4, "scaled_centroids": true}
            """
        let parsed = try metadata(minimal)
        XCTAssertTrue(parsed.isValid)
        XCTAssertEqual(parsed.forceTokens, [])
        XCTAssertEqual(parsed.nProbes, 512)
        XCTAssertEqual(parsed.groupSize, 64)
    }

    func testConfigurationDecodesFlashMetadataAndDefaultsExactMode() throws {
        let config = try flashConfiguration()
        XCTAssertEqual(config.flashHead?.nClusters, 8)
        XCTAssertEqual(config.flashHead?.forceTokens, [3])
        let model = MapleModel(config)
        XCTAssertNotNil(model.lmHeadFlash)
        XCTAssertEqual(model.headMode, .exact)

        let noFlash = MapleModel(try flashConfiguration(includeFlash: false))
        XCTAssertNil(noFlash.lmHeadFlash)
    }

    func testSanitizeRebuildsClusterOrderedHeadAndDropsClusterScale() throws {
        let model = MapleModel(try flashConfiguration())
        // token_map maps cluster row (c, r) to token (63 - (c*8 + r)): reversed.
        let order = MLXArray(stride(from: 63, through: 0, by: -1).map { Int32($0) })
        let tokenMap = order.reshaped(8, 8)
        let headWeights = MLXArray(0 ..< (64 * 32)).asType(.float32).reshaped(64, 32)
        let headScales = MLXArray(0 ..< (64 * 4)).asType(.float32).reshaped(64, 4)
        let headBiases = MLXArray(0 ..< (64 * 4)).asType(.float32).reshaped(64, 4)

        let sanitized = model.sanitize(weights: [
            "lm_head.weight": headWeights,
            "lm_head.scales": headScales,
            "lm_head.biases": headBiases,
            "lm_head_flash.token_map": tokenMap,
            "lm_head_flash.cluster_scale": MLXArray.ones([8]),
        ])

        XCTAssertNil(sanitized["lm_head_flash.cluster_scale"])
        guard let rebuilt = sanitized["lm_head_flash.head.weight"],
            let rebuiltScales = sanitized["lm_head_flash.head.scales"],
            let rebuiltBiases = sanitized["lm_head_flash.head.biases"]
        else {
            return XCTFail("sanitizer must rebuild lm_head_flash.head.*")
        }
        XCTAssertEqual(rebuilt.shape, [8, 8, 32])
        XCTAssertEqual(rebuiltScales.shape, [8, 8, 4])
        XCTAssertEqual(rebuiltBiases.shape, [8, 8, 4])
        eval(rebuilt, rebuiltScales, rebuiltBiases)
        // Row (c, r) of the rebuilt head is lm_head row token_map[c, r].
        let expected = take(headWeights, order, axis: 0).reshaped(8, 8, 32)
        eval(expected)
        XCTAssertEqual(rebuilt.asArray(Float.self), expected.asArray(Float.self))
    }

    func testSanitizeDropsFlashTensorsWithoutMetadata() throws {
        let model = MapleModel(try flashConfiguration(includeFlash: false))
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.zeros([64, 32]),
            "lm_head_flash.token_map": MLXArray.zeros([8, 8], dtype: .int32),
            "lm_head_flash.centroids.weight": MLXArray.zeros([8, 32], dtype: .uint32),
        ])
        XCTAssertFalse(sanitized.keys.contains { $0.hasPrefix("lm_head_flash.") })
        XCTAssertNotNil(sanitized["lm_head.weight"])
    }

    /// Scored tokens must carry exactly the dense-head logits, forced tokens
    /// must always be scored, and everything else must be -inf.
    func testFlashHeadScoresTopClustersAndForcedTokensExactly() throws {
        let hidden = 32
        let meta = try metadata(metadataJSON())
        let flash = MapleFlashHead(hiddenSize: hidden, metadata: meta)
        let lmHead = QuantizedLinear(hidden, 32, bias: false, groupSize: 32, bits: 4)

        // Identity token_map: cluster c covers tokens [8c, 8c+8).
        let tokenMap = MLXArray(0 ..< 32).reshaped(4, 8)
        // Cluster-ordered head is the dense head under the identity permutation.
        let clusteredWeight = lmHead.weight.reshaped(4, 8, -1)
        let clusteredScales = lmHead.scales.reshaped(4, 8, -1)
        let clusteredBiases = lmHead.biases!.reshaped(4, 8, -1)
        // One-hot centroids so centroid scores are C @ h with exact 4-bit rows.
        var centroidRows = [Float](repeating: 0, count: 4 * hidden)
        for c in 0 ..< 4 { centroidRows[c * hidden + c] = 1 }
        let (cw, cs, cb) = MLX.quantized(
            MLXArray(centroidRows).reshaped(4, hidden), groupSize: 32, bits: 4)
        try flash.update(
            parameters: ModuleParameters.unflattened([
                "token_map": tokenMap,
                "head.weight": clusteredWeight,
                "head.scales": clusteredScales,
                "head.biases": clusteredBiases,
                "centroids.weight": cw,
                "centroids.scales": cs,
                "centroids.biases": cb!,
            ]), verify: [])

        // Distinct descending centroid scores: top-2 clusters are {0, 1}.
        var hValues = [Float](repeating: 0, count: hidden)
        hValues[0] = 1.0
        hValues[1] = 0.5
        hValues[2] = 0.25
        hValues[3] = 0.125
        let h = MLXArray(hValues).reshaped(1, 1, hidden)
        let dense = lmHead(h[0..., 0, 0...]).reshaped(-1)  // [32] exact logits

        let out = flash(h, lmHead: lmHead)
        eval(out, dense)
        XCTAssertEqual(out.shape, [1, 1, 32])
        let flat = out.reshaped(-1).asType(.float32)
        let denseFlat = dense.asType(.float32)
        eval(flat, denseFlat)
        let outValues = flat.asArray(Float.self)
        let denseValues = denseFlat.asArray(Float.self)
        for token in 0 ..< 32 {
            let scored = token < 16 || token == 31
            if scored {
                XCTAssertFalse(outValues[token].isInfinite, "token \(token) must be scored")
                XCTAssertEqual(outValues[token], denseValues[token], accuracy: 1e-2)
            } else {
                XCTAssertEqual(outValues[token], -.infinity, "token \(token) must be -inf")
            }
        }
    }

    /// The model must use the exact head by default, for prefill, and whenever
    /// the dense head is not an affine quantized linear.
    func testExactHeadFallbackConditions() throws {
        let model = MapleModel(try flashConfiguration())
        let prompt = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)
        let token = MLXArray([6] as [Int32]).reshaped(1, 1)

        // Unquantized dense head: even in flash mode, decode uses the exact head.
        model.headMode = .flash
        let cache = model.newCache(parameters: nil)
        let prefillOut = model(prompt, cache: cache)
        eval(prefillOut)
        XCTAssertEqual(prefillOut.shape, [1, 5, 64])
        XCTAssertTrue(
            prefillOut.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite),
            "prefill must use the exact head")

        let decodeOut = model(token, cache: cache)
        eval(decodeOut)
        XCTAssertTrue(
            decodeOut.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite),
            "unquantized head must fall back to exact")
    }

    /// With a quantized affine head, flash mode engages for single-token
    /// decode and scores only probed clusters plus forced tokens.
    func testFlashModeEngagesForQuantizedSingleTokenDecode() throws {
        let model = MapleModel(try flashConfiguration())
        quantize(model: model) { path, _ in
            path == "lm_head" ? (groupSize: 64, bits: 4, mode: .affine) : nil
        }
        model.headMode = .flash
        let cache = model.newCache(parameters: nil)
        _ = model(MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3), cache: cache)
        let out = model(MLXArray([4] as [Int32]).reshaped(1, 1), cache: cache)
        eval(out)
        XCTAssertEqual(out.shape, [1, 1, 64])
        let values = out.asType(.float32).asArray(Float.self)
        // 4 probed clusters of 8 tokens plus the forced token: at most 33.
        XCTAssertLessThanOrEqual(values.filter { $0 > -.infinity }.count, 33)
        XCTAssertTrue(values[3] > -.infinity, "forced token must be scored")
    }
}
