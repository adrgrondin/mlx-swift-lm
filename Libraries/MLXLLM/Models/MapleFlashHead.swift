// Copyright © 2026 DeepGrove AI.
//
// Optional approximate FlashHead output head for single-stream Maple decode,
// ported from https://github.com/deepgrove-ai/mlx-lm (commit
// eba96c16158f032821b0bf374ea1421cfddef0a9).
//
// Reference: FlashHead — Efficient Drop-in Replacement for the Classification
// Head in Language Model Inference.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Output-head selection for ``MapleModel``. ``exact`` (the default) scores the
/// full vocabulary; ``flash`` scores cluster centroids and then only the tokens
/// of the top clusters during single-token decode.
public enum MapleHeadMode: String, Codable, Sendable {
    case exact
    case flash
}

/// Typed view of the checkpoint's `flash_head` metadata, written by
/// `mlx_lm.ternary --flash-head`.
public struct MapleFlashHeadMetadata: Codable, Sendable {
    public var bits: Int
    public var clusterSize: Int
    public var forceTokens: [Int]
    public var groupSize: Int
    public var headBits: Int
    public var headGroupSize: Int
    public var nClusters: Int
    public var nProbes: Int
    public var scaledCentroids: Bool

    enum CodingKeys: String, CodingKey {
        case bits
        case clusterSize = "cluster_size"
        case forceTokens = "force_tokens"
        case groupSize = "group_size"
        case headBits = "head_bits"
        case headGroupSize = "head_group_size"
        case nClusters = "n_clusters"
        case nProbes = "n_probes"
        case scaledCentroids = "scaled_centroids"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        bits = try c.decodeIfPresent(Int.self, forKey: .bits) ?? 4
        clusterSize = try c.decode(Int.self, forKey: .clusterSize)
        forceTokens = try c.decodeIfPresent([Int].self, forKey: .forceTokens) ?? []
        groupSize = try c.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
        headBits = try c.decodeIfPresent(Int.self, forKey: .headBits) ?? 4
        headGroupSize = try c.decodeIfPresent(Int.self, forKey: .headGroupSize) ?? 64
        nClusters = try c.decode(Int.self, forKey: .nClusters)
        nProbes = try c.decodeIfPresent(Int.self, forKey: .nProbes) ?? 512
        // Older checkpoints predate scaled centroids and must be regenerated.
        scaledCentroids = try c.decodeIfPresent(Bool.self, forKey: .scaledCentroids) ?? false
    }

    /// Metadata is usable only with scaled centroids, positive dimensions, and
    /// a probe count that selects at least one cluster.
    public var isValid: Bool {
        scaledCentroids && bits > 0 && clusterSize > 0 && groupSize > 0 && headBits > 0
            && headGroupSize > 0 && nClusters > 0 && nProbes > 0
    }

    /// Default matches the converter's `--probes` default; every generated
    /// checkpoint records the value explicitly.
    public var resolvedProbes: Int { min(nProbes, nClusters) }
}

/// Cluster-ordered copy of the quantized `lm_head`: a row-permutation by
/// `token_map`, rebuilt by the sanitizer so checkpoints can omit it.
final class MapleFlashHeadClusteredHead: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray

    init(nClusters: Int, clusterSize: Int, hiddenSize: Int, headBits: Int, headGroupSize: Int) {
        _weight.wrappedValue = MLXArray.zeros(
            [nClusters, clusterSize, hiddenSize * headBits / 32], dtype: .uint32)
        let groupCount = hiddenSize / headGroupSize
        _scales.wrappedValue = MLXArray.zeros(
            [nClusters, clusterSize, groupCount], dtype: .bfloat16)
        _biases.wrappedValue = MLXArray.zeros(
            [nClusters, clusterSize, groupCount], dtype: .bfloat16)
    }
}

/// Two-phase approximate lm_head for single-stream decode.
///
/// Phase one scores quantized cluster centroids of the vocabulary; phase two
/// computes exact logits only for the tokens of the top `nProbes` clusters
/// (plus a fixed set of forced control tokens such as EOS). All other logits
/// are -inf, so greedy decoding is exact whenever the true argmax lies in the
/// probed clusters. Prefill and batched calls must use the exact head;
/// `MapleModel` enforces that.
public final class MapleFlashHead: Module {
    /// Centroids are directions, pre-scaled at generation time by the largest
    /// lm_head row norm in their cluster.
    @ModuleInfo(key: "centroids") var centroids: QuantizedLinear
    @ParameterInfo(key: "token_map") var tokenMap: MLXArray
    @ModuleInfo(key: "head") var head: MapleFlashHeadClusteredHead

    let nProbes: Int
    let headGroupSize: Int
    let headBits: Int
    let forceTokenIDs: [Int]

    var forceIds: MLXArray?
    var forceRows: (weight: MLXArray, scales: MLXArray, biases: MLXArray?)?

    init(hiddenSize: Int, metadata: MapleFlashHeadMetadata) {
        nProbes = metadata.resolvedProbes
        headGroupSize = metadata.headGroupSize
        headBits = metadata.headBits
        forceTokenIDs = metadata.forceTokens
        _centroids.wrappedValue = QuantizedLinear(
            hiddenSize, metadata.nClusters, bias: false,
            groupSize: metadata.groupSize, bits: metadata.bits)
        _tokenMap.wrappedValue = MLXArray.zeros(
            [metadata.nClusters, metadata.clusterSize], dtype: .int32)
        _head.wrappedValue = MapleFlashHeadClusteredHead(
            nClusters: metadata.nClusters, clusterSize: metadata.clusterSize,
            hiddenSize: hiddenSize, headBits: metadata.headBits,
            headGroupSize: metadata.headGroupSize)
    }

    /// Score the last position of `h` `[1, 1, hidden]` approximately. Returns
    /// `[1, 1, vocab]` with -inf for every unscored token.
    func callAsFunction(_ h: MLXArray, lmHead: QuantizedLinear) -> MLXArray {
        let hv = h[0..., h.dim(1) - 1, 0...]
        let top = argPartition(centroids(hv), kth: -nProbes, axis: -1)[
            .ellipsis, (-nProbes)...]  // [1, nProbes]
        var oids = take(tokenMap, top[0], axis: 0).reshaped(-1)

        var logits = gatherQuantizedMM(
            hv.reshaped(1, 1, 1, 1, -1),
            head.weight,
            scales: head.scales,
            biases: head.biases,
            rhsIndices: top[0..., .newAxis, 0...],
            transpose: true,
            groupSize: headGroupSize,
            bits: headBits
        ).reshaped(-1)

        if !forceTokenIDs.isEmpty {
            if forceIds == nil {
                let ids = MLXArray(forceTokenIDs.map { Int32($0) })
                eval(ids)
                forceIds = ids
            }
            if forceRows == nil {
                let rows = (
                    take(lmHead.weight, forceIds!, axis: 0),
                    take(lmHead.scales, forceIds!, axis: 0),
                    lmHead.biases.map { take($0, forceIds!, axis: 0) }
                )
                eval(rows.0, rows.1)
                if let biases = rows.2 { eval(biases) }
                forceRows = rows
            }
            let rows = forceRows!
            let forceLogits = quantizedMatmul(
                hv,
                rows.weight,
                scales: rows.scales,
                biases: rows.biases,
                transpose: true,
                groupSize: lmHead.groupSize,
                bits: lmHead.bits,
                mode: .affine
            )[0]
            oids = concatenated([oids, forceIds!])
            logits = concatenated([logits, forceLogits])
        }

        let vocabularySize = lmHead.weight.dim(0)
        var full = MLXArray.full(
            [1, 1, vocabularySize], values: MLXArray(-Float.infinity), dtype: logits.dtype)
        full[0, 0, oids] = logits
        return full
    }
}
