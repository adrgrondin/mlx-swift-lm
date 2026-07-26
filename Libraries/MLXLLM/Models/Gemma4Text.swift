//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Compiled fusion fragments
//
// Gemma 4 ships with a single rms_norm_eps (1e-6) on every RMSNorm in the
// model (see Gemma4TextConfiguration.rmsNormEps default — all upstream
// Gemma 4 weights use this value). Hardcoding the constant lets one compiled
// graph serve every layer without per-layer specialization. `Gemma4DecoderLayer.init`
// asserts the config matches so a future checkpoint with a different eps fails
// loudly instead of silently using the wrong value.
//
// Mirrors the upstream mlx-lm Python optimization
// (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py)
// which fuses (residual + RMSNorm(x) * weight) and gelu(g) * other into a
// single compiled graph. The Python equivalent measured ~+2.4% decode tps on
// M4 Max for gemma-4-e2b-it-4bit at batch=1; the Swift gain is larger
// (~+23.8% on the same model and hardware) because Swift's per-op MLX
// dispatch has more overhead, so consolidating ops via compile() recovers
// more of that overhead. See PR description for the per-trial numbers.

private let kRMSEps: Float = 1e-6

/// Convert a `RoPEOffset` to an `MLXArray` for the compiled pre-attention segment.
/// The compiled function requires `MLXArray` offset (not a bare `Int`) so that
/// `compile` treats it as a runtime input, not a compile-time constant.
private func ropeOffsetToArray(_ offset: RoPEOffset) -> MLXArray {
    switch offset {
    case .scalar(let i): return MLXArray(Int32(i))
    case .batch(let arr): return arr
    }
}

private let _addRMSNorm: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { residual, x, weight in
    residual + MLXFast.rmsNorm(x, weight: weight, eps: kRMSEps)
}

private let _geluMul: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, other in
    geluApproximate(gate) * other
}

// MARK: - Native compiled pre-attention segments (Phase 6/7 port)
//
// These compile the entire pre-attention chain (input_layernorm → SRQ →
// quantizedMM(fused_qkv or q_proj) → SRQ → q/k/v norms → transpose → RoPE) into
// a single Metal graph. `quantizedMM` is compile-friendly (unlike the custom
// metalKernel), so `compile` fuses the element-wise ops (norms, SRQ, RoPE) with
// the matmul. KV cache + SDPA stay eager between the pre/post compiled segments.
//
// `shapeless: true` is NOT used: the pre-attention functions read `.shape` for
// tensor slicing/reshaping — MLX cannot infer slice output shapes with unknown
// dimensions (confirmed in Python Phase 7). Per-shape compile (default) is used;
// load-time precompilation (Phase 5) eliminates the per-shape JIT cost.

/// Factory + cache for the compiled pre-attention segment of **source layers**
/// (own K/V). Keyed by `(nHeads, headDim, nKvHeads)` — the constants captured in
/// the compiled graph for tensor slicing/reshaping. One compiled function is
/// shared across all source layers with the same head configuration (15 layers
/// for E2B). Mirrors Python `_get_compiled_pre_attn_source`.
private enum CompiledPreAttnSource {
    struct Key: Hashable { let nHeads: Int; let headDim: Int; let nKvHeads: Int }
    nonisolated(unsafe) static var cache: [Key: @Sendable ([MLXArray]) -> [MLXArray]] = [:]

    static func fn(nHeads: Int, headDim: Int, nKvHeads: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
        let key = Key(nHeads: nHeads, headDim: headDim, nKvHeads: nKvHeads)
        if let cached = cache[key] { return cached }
        let qd = nHeads * headDim
        let kvd = nKvHeads * headDim
        let f: @Sendable ([MLXArray]) -> [MLXArray] = compile { args in
            let x = args[0]
            let inputNormW = args[1]
            let qkvWq = args[2]
            let qkvScales = args[3]
            let qkvBiases = args[4]
            let qkvInS = args[5]
            let qkvOutS = args[6]
            let qNormW = args[7]
            let kNormW = args[8]
            let vNormW = args[9]
            let ropeFreqs = args[10]
            let offset = args[11]

            var h = MLXFast.rmsNorm(x, weight: inputNormW, eps: kRMSEps).asType(.float32)
            h = srqF32(h, qkvInS)
            var qkv = quantizedMM(
                h, qkvWq, scales: qkvScales, biases: qkvBiases,
                transpose: true, groupSize: 128, bits: 4, mode: .affine)
            qkv = srqF32(qkv, qkvOutS).asType(x.dtype)

            let B = qkv.dim(0)
            let L = qkv.dim(1)
            var queries = qkv[.ellipsis, 0..<qd].reshaped(B, L, nHeads, headDim)
            var keys = qkv[.ellipsis, qd..<(qd + kvd)].reshaped(B, L, nKvHeads, headDim)
            var values = qkv[.ellipsis, (qd + kvd)..<qkv.dim(-1)].reshaped(B, L, nKvHeads, headDim)

            queries = MLXFast.rmsNorm(queries, weight: qNormW, eps: kRMSEps)
            keys = MLXFast.rmsNorm(keys, weight: kNormW, eps: kRMSEps)
            values = MLXFast.rmsNorm(values, weight: vNormW, eps: kRMSEps)

            keys = keys.transposed(0, 2, 1, 3)
            keys = MLXFast.RoPE(keys, dimensions: headDim, traditional: false, base: nil, scale: 1.0, offset: offset, freqs: ropeFreqs)
            values = values.transposed(0, 2, 1, 3)
            queries = queries.transposed(0, 2, 1, 3)
            queries = MLXFast.RoPE(queries, dimensions: headDim, traditional: false, base: nil, scale: 1.0, offset: offset, freqs: ropeFreqs)

            return [queries, keys, values]
        }
        cache[key] = f
        return f
    }
}

/// Factory + cache for the compiled pre-attention segment of **KV-shared layers**
/// (reuse earlier K/V, own only q_proj). Keyed by `(nHeads, headDim)`. Mirrors
/// Python `_get_compiled_pre_attn_kvshared`.
private enum CompiledPreAttnKvshared {
    struct Key: Hashable { let nHeads: Int; let headDim: Int }
    nonisolated(unsafe) static var cache: [Key: @Sendable ([MLXArray]) -> [MLXArray]] = [:]

    static func fn(nHeads: Int, headDim: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
        let key = Key(nHeads: nHeads, headDim: headDim)
        if let cached = cache[key] { return cached }
        let f: @Sendable ([MLXArray]) -> [MLXArray] = compile { args in
            let x = args[0]
            let inputNormW = args[1]
            let qWq = args[2]
            let qScales = args[3]
            let qBiases = args[4]
            let qInS = args[5]
            let qOutS = args[6]
            let qNormW = args[7]
            let ropeFreqs = args[8]
            let offset = args[9]

            var h = MLXFast.rmsNorm(x, weight: inputNormW, eps: kRMSEps).asType(.float32)
            h = srqF32(h, qInS)
            var q = quantizedMM(
                h, qWq, scales: qScales, biases: qBiases,
                transpose: true, groupSize: 128, bits: 4, mode: .affine)
            q = srqF32(q, qOutS).asType(x.dtype)

            let B = q.dim(0)
            let L = q.dim(1)
            var queries = q.reshaped(B, L, nHeads, headDim)
            queries = MLXFast.rmsNorm(queries, weight: qNormW, eps: kRMSEps)
            queries = queries.transposed(0, 2, 1, 3)
            queries = MLXFast.RoPE(queries, dimensions: headDim, traditional: false, base: nil, scale: 1.0, offset: offset, freqs: ropeFreqs)

            return [queries]
        }
        cache[key] = f
        return f
    }
}

/// Factory + cache for the compiled post-attention + MLP + PLE segment (all
/// layers). Keyed by `(mlpBits, pleBits)` — the `quantizedMM` bit-width
/// constants captured in the compiled graph. o_proj is always 4-bit (hardcoded).
/// One compiled function is shared across all layers with the same bit combo.
/// Mirrors Python `_get_compiled_post_attn_mlp_ple`.
///
/// Input array (38 elements): `[residual, attnOutput, perLayerInput,
/// postAttnW, preFfW, postFfW, pleNormW, layerScalar,
/// oWq, oScales, oBiases, oInS, oOutS,
/// gateWq, gateScales, gateBiases, gateInS, gateOutS,
/// upWq, upScales, upBiases, upInS, upOutS,
/// downWq, downScales, downBiases, downInS, downOutS,
/// pleGWq, pleGScales, pleGBiases, pleGInS, pleGOutS,
/// plePWq, plePScales, plePBiases, plePInS, plePOutS]`.
private enum CompiledPostAttn {
    struct Key: Hashable { let mlpBits: Int; let pleBits: Int }
    nonisolated(unsafe) static var cache: [Key: @Sendable ([MLXArray]) -> [MLXArray]] = [:]

    static func fn(mlpBits: Int, pleBits: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
        let key = Key(mlpBits: mlpBits, pleBits: pleBits)
        if let cached = cache[key] { return cached }
        let f: @Sendable ([MLXArray]) -> [MLXArray] = compile { args in
            let residual = args[0]
            let attnOutput = args[1]
            let perLayerInput = args[2]
            let postAttnW = args[3]
            let preFfW = args[4]
            let postFfW = args[5]
            let pleNormW = args[6]
            let layerScalar = args[7]
            let oWq = args[8], oScales = args[9], oBiases = args[10], oInS = args[11], oOutS = args[12]
            let gateWq = args[13], gateScales = args[14], gateBiases = args[15], gateInS = args[16], gateOutS = args[17]
            let upWq = args[18], upScales = args[19], upBiases = args[20], upInS = args[21], upOutS = args[22]
            let downWq = args[23], downScales = args[24], downBiases = args[25], downInS = args[26], downOutS = args[27]
            let pleGWq = args[28], pleGScales = args[29], pleGBiases = args[30], pleGInS = args[31], pleGOutS = args[32]
            let plePWq = args[33], plePScales = args[34], plePBiases = args[35], plePInS = args[36], plePOutS = args[37]

            let dt = attnOutput.dtype

            // Post-attention: o_proj → norm → residual
            var h = srqF32(attnOutput.asType(.float32), oInS)
            h = quantizedMM(h, oWq, scales: oScales, biases: oBiases,
                transpose: true, groupSize: 128, bits: 4, mode: .affine)
            h = srqF32(h, oOutS).asType(dt)
            h = MLXFast.rmsNorm(h, weight: postAttnW, eps: kRMSEps)
            h = residual + h

            // MLP: pre_ff_norm → gate/up → gelu → down → post_ff_norm → residual
            var residual2 = h
            h = MLXFast.rmsNorm(h, weight: preFfW, eps: kRMSEps).asType(.float32)
            var gate = quantizedMM(srqF32(h, gateInS), gateWq, scales: gateScales, biases: gateBiases,
                transpose: true, groupSize: 128, bits: mlpBits, mode: .affine)
            gate = srqF32(gate, gateOutS).asType(dt)
            var up = quantizedMM(srqF32(h, upInS), upWq, scales: upScales, biases: upBiases,
                transpose: true, groupSize: 128, bits: mlpBits, mode: .affine)
            up = srqF32(up, upOutS).asType(dt)
            h = geluApproximate(gate) * up
            h = h.asType(.float32)
            var down = quantizedMM(srqF32(h, downInS), downWq, scales: downScales, biases: downBiases,
                transpose: true, groupSize: 128, bits: mlpBits, mode: .affine)
            down = srqF32(down, downOutS).asType(dt)
            h = MLXFast.rmsNorm(down, weight: postFfW, eps: kRMSEps)
            h = residual2 + h

            // PLE: gate → gelu → multiply → proj → norm → residual
            residual2 = h
            h = h.asType(.float32)
            var pleGate = quantizedMM(srqF32(h, pleGInS), pleGWq, scales: pleGScales, biases: pleGBiases,
                transpose: true, groupSize: 128, bits: pleBits, mode: .affine)
            pleGate = srqF32(pleGate, pleGOutS).asType(dt)
            pleGate = geluApproximate(pleGate)
            pleGate = pleGate * perLayerInput
            pleGate = pleGate.asType(.float32)
            var pleProj = quantizedMM(srqF32(pleGate, plePInS), plePWq, scales: plePScales, biases: plePBiases,
                transpose: true, groupSize: 128, bits: pleBits, mode: .affine)
            pleProj = srqF32(pleProj, plePOutS).asType(dt)
            pleProj = MLXFast.rmsNorm(pleProj, weight: pleNormW, eps: kRMSEps)
            h = residual2 + pleProj

            h = h * layerScalar
            return [h]
        }
        cache[key] = f
        return f
    }
}

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    var modelType: String = "gemma4_text"
    @_spi(GemmaEncoder) public var hiddenSize: Int = 1536
    @_spi(GemmaEncoder) public var numHiddenLayers: Int = 35
    var intermediateSize: Int = 6144
    var numAttentionHeads: Int = 8
    var headDim: Int = 256
    var globalHeadDim: Int = 512
    var globalPartialRotaryFactor: Float = 0.25
    var rmsNormEps: Float = 1e-6
    var vocabSize: Int = 262144
    var vocabSizePerLayerInput: Int = 262144
    var numKeyValueHeads: Int = 1
    var numGlobalKeyValueHeads: Int?
    var numKvSharedLayers: Int = 20
    var hiddenSizePerLayerInput: Int = 256
    var slidingWindow: Int = 512
    var slidingWindowPattern: Int = 5
    var maxPositionEmbeddings: Int = 131072
    var attentionKeqV: Bool = false
    var finalLogitSoftcapping: Float = 30.0
    var useDoubleWideMlp: Bool = true
    // MoE block (E-series: enable_moe_block=true)
    var enableMoEBlock: Bool = false
    var numExperts: Int?
    var topKExperts: Int?
    var moeIntermediateSize: Int?
    var layerTypes: [String] = []
    var tieWordEmbeddings: Bool = true

    // Gemma 4 QAT mobile (wNa8o8) quantization config (`quant_method: "gemma"`).
    // Present only for the mobile checkpoints; nil for ordinary fp/4-bit Gemma 4.
    var quantizationConfig: GemmaMobileQuantizationConfig?

    // RoPE parameters (nested dict with full_attention/sliding_attention sub-configs)
    var ropeParameters: [String: [String: StringOrNumber]]?

    // Derived properties
    var slidingRopeTheta: Float = 10000.0
    var fullRopeTheta: Float = 1_000_000.0
    var fullPartialRotaryFactor: Float = 1.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case globalPartialRotaryFactor = "global_partial_rotary_factor"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKeqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantizationConfig = "quantization_config"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 35
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        self.globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        self.globalPartialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .globalPartialRotaryFactor) ?? 0.25
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        self.vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 262144
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        self.numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        self.numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 20
        self.hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 256
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.attentionKeqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKeqV) ?? false
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        self.useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? true
        self.enableMoEBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoEBlock) ?? false
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        self.topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = decoded
        } else {
            // Derive layer types from sliding window pattern
            var pattern = [String]()
            for i in 0 ..< slidingWindowPattern {
                pattern.append(
                    i == slidingWindowPattern - 1 ? "full_attention" : "sliding_attention")
            }
            var types = [String]()
            while types.count < numHiddenLayers {
                types.append(contentsOf: pattern)
            }
            self.layerTypes = Array(types.prefix(numHiddenLayers))
        }
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.quantizationConfig = try container.decodeIfPresent(
            GemmaMobileQuantizationConfig.self, forKey: .quantizationConfig)
        self.ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        // Extract RoPE parameters from nested config
        if let ropeParams = ropeParameters {
            if let sliding = ropeParams["sliding_attention"] {
                self.slidingRopeTheta = sliding["rope_theta"]?.asFloat() ?? 10000.0
            }
            if let full = ropeParams["full_attention"] {
                self.fullRopeTheta = full["rope_theta"]?.asFloat() ?? 1_000_000.0
                self.fullPartialRotaryFactor =
                    full["partial_rotary_factor"]?.asFloat() ?? 1.0
            }
        }
    }
}

// MARK: - Helper Modules

private class RMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

// MARK: - Attention

private class Gemma4Attention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let nKvHeads: Int
    let useKeqV: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    // Optional: KV-shared layers reuse an earlier layer's K/V and own no k_proj/v_proj.
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    // Optional: KV-shared layers don't compute K, so they carry no k_norm weight.
    // (v_norm is RMSNormNoScale — parameter-free — so it never appears in checkpoints.)
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    @ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale

    @ModuleInfo var rope: RoPELayer

    // Fused q/k/v cache (plan §9.3): concatenates packed q/k/v weights once and
    // runs a single qmv kernel with per-row output SRQ, replacing three kernel
    // launches with one. Built lazily on the first decode/small-batch call.
    private var _fusedQKVWeight: MLXArray?
    private var _fusedQKVWeightScale: MLXArray?
    private var _fusedQKVInS: MLXArray?
    private var _fusedQKVOutS: MLXArray?
    private var _fusedQKVReady = false
    private var _fusedQKVDisabled = false

    // Precomputed RoPE frequencies for the native compiled path. Shape [headDim/2]
    // with `inf` for non-rotated dims (full attention / ProportionalRoPE). Passed
    // as an input to `MLXFast.RoPE` inside the compiled pre-attention segment.
    // The Metal rope kernel computes `inv_freq = 1/freqs`, so `inf` → 0 → theta=0
    // → identity (no rotation), matching the Python ProportionalRoPE._freqs approach.
    private var _compiledRopeFreqs: MLXArray

    // Fused native q/k/v weights for the compiled pre-attention segment (source
    // layers only). Built lazily; concatenates q/k/v native (uint32) weights +
    // per-row output SRQ scale. Mirrors Python `_build_fused_qkv_native`.
    private var _fusedQKVNative: (
        wq: MLXArray, scales: MLXArray, biases: MLXArray,
        inS: MLXArray, outS: MLXArray
    )?
    private var _fusedQKVNativeReady = false

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"

        // Full attention uses globalHeadDim, sliding uses headDim
        self.effectiveHeadDim =
            isSliding ? config.headDim : config.globalHeadDim

        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads

        // K-eq-V for full attention layers
        self.useKeqV = config.attentionKeqV && !isSliding
        if useKeqV, let globalKvHeads = config.numGlobalKeyValueHeads {
            self.nKvHeads = globalKvHeads
        } else {
            self.nKvHeads = config.numKeyValueHeads
        }

        self.scale = 1.0

        self._qProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        // KV-shared layers (the last `num_kv_shared_layers`) reuse the K/V of an earlier
        // layer of the same attention type, so they own no k_proj/v_proj. Quantized
        // (QAT) checkpoints prune those tensors; create them only for the KV-owning
        // layers so the module tree matches the checkpoint. (Older/PTQ checkpoints that
        // still ship the redundant tensors are dropped in `sanitize`.) Same predicate as
        // the double-wide MLP gate.
        let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        let isKvSharedLayer = layerIdx >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0
        if !isKvSharedLayer {
            self._kProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            if !useKeqV {
                self._vProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            }
        }
        self._oProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        if !isKvSharedLayer {
            self._kNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        }
        self._vNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)

        // RoPE: sliding uses default, full uses proportional with partial rotation
        if isSliding {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.fullRopeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
        }

        // Precompute RoPE frequencies for the native compiled path.
        // Sliding: base^(arange(0, dims, 2) / dims), shape [dims/2].
        // Full (ProportionalRoPE): factor * base^(arange(0, rotatedDims, 2) / dims)
        // padded with `inf` for non-rotated dims, shape [dims/2]. The `inf` freqs
        // produce identity (no rotation) in the Metal rope kernel (inv_freq = 1/inf = 0).
        if isSliding {
            let exponents = MLXArray(
                stride(from: 0, to: effectiveHeadDim, by: 2)
            ).asType(.float32) / Float(effectiveHeadDim)
            self._compiledRopeFreqs = MLX.pow(config.slidingRopeTheta, exponents)
        } else {
            let dims = effectiveHeadDim
            let rotatedDims = 2 * Int(config.fullPartialRotaryFactor * Float(dims) / 2)
            let ropeAngles = rotatedDims / 2
            let nopeAngles = dims / 2 - ropeAngles
            let exponents = MLXArray(
                stride(from: 0, to: rotatedDims, by: 2)
            ).asType(.float32) / Float(dims)
            var freqs = MLX.pow(config.fullRopeTheta, exponents)
            if nopeAngles > 0 {
                let infPad = MLXArray.ones([nopeAngles], dtype: .float32) * Float.infinity
                freqs = MLX.concatenated([freqs, infPad], axis: 0)
            }
            self._compiledRopeFreqs = freqs
        }

        super.init()
    }

    /// Precomputed RoPE frequencies for the native compiled path.
    var compiledRopeFreqs: MLXArray { _compiledRopeFreqs }

    /// Build concatenated native-format q/k/v weights for the compiled pre-attention
    /// segment (source layers only). Returns `(wq, scales, biases, inS, outS)`
    /// where the weights are concatenated along the output dim and `outS` is a
    /// per-row array (q/k/v rows may have different output SRQ scales). Mirrors
    /// Python `_build_fused_qkv_native`.
    func buildFusedQKVNative()
        -> (wq: MLXArray, scales: MLXArray, biases: MLXArray,
            inS: MLXArray, outS: MLXArray)?
    {
        if _fusedQKVNativeReady { return _fusedQKVNative }
        guard let q = qProj as? GemmaQuantizedLinear,
            let k = kProj as? GemmaQuantizedLinear,
            let v = vProj as? GemmaQuantizedLinear,
            let qArgs = q.nativeArgs(),
            let kArgs = k.nativeArgs(),
            let vArgs = v.nativeArgs()
        else {
            return nil
        }
        let wq = MLX.concatenated([qArgs.packed, kArgs.packed, vArgs.packed], axis: 0)
        let scales = MLX.concatenated([qArgs.scales, kArgs.scales, vArgs.scales], axis: 0)
        let biases = MLX.concatenated([qArgs.biases, kArgs.biases, vArgs.biases], axis: 0)
        let inS = qArgs.inScale  // shared input SRQ scale
        let outS = gemmaBuildPerRowOutputScale(q, k, v, dtype: .float32)
        eval([wq, scales, biases, outS])
        _fusedQKVNative = (wq, scales, biases, inS, outS)
        _fusedQKVNativeReady = true
        return _fusedQKVNative
    }

    /// Try fused q/k/v projection (single Metal kernel for all three). Returns
    /// the concatenated `[..., q_out + k_out + v_out]` output, or `nil` when
    /// fusion is not possible (batch too large, unaligned dims, mismatched input
    /// SRQ scales, or projections are not all Gemma-quantized). The caller splits
    /// the output and applies norms separately. Mirrors Python
    /// `gemma_fused_qkv_matmul`.
    private func tryFusedQKV(
        q: GemmaQuantizedLinear, k: GemmaQuantizedLinear, v: GemmaQuantizedLinear,
        x: MLXArray
    ) -> MLXArray? {
        if _fusedQKVDisabled { return nil }
        let batchSize = x.shape.dropLast().reduce(1, *)
        guard q.canUseQMV(batchSize: batchSize) else { return nil }
        guard q.numBits == k.numBits && q.numBits == v.numBits else { return nil }
        guard q.inputDims == k.inputDims && q.inputDims == v.inputDims else { return nil }

        if !_fusedQKVReady {
            guard gemmaQKVInputScalesMatch(q, k, v) else {
                _fusedQKVDisabled = true
                return nil
            }
            _fusedQKVWeight = MLX.concatenated([q.weight, k.weight, v.weight], axis: 0)
            _fusedQKVWeightScale = MLX.concatenated(
                [q.weightScale, k.weightScale, v.weightScale], axis: 0)
            _fusedQKVInS = q.inputActivationScale
            _fusedQKVOutS = gemmaBuildPerRowOutputScale(q, k, v, dtype: x.dtype)
            eval([_fusedQKVWeight!, _fusedQKVWeightScale!, _fusedQKVOutS!])
            _fusedQKVReady = true
        }

        return gemmaFusedQKVMatmul(
            x: x, weight: _fusedQKVWeight!, weightScale: _fusedQKVWeightScale!,
            inputScale: _fusedQKVInS!, outputScale: _fusedQKVOutS!,
            numBits: q.numBits, inputDims: q.inputDims)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        positionOffset: RoPEOffset? = nil
    ) -> (MLXArray, Gemma4SharedKVState, RoPEOffset?) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let activePositionOffset = positionOffset ?? cache?.ropeOffset

        var queries: MLXArray
        let kvState: Gemma4SharedKVState

        if let sharedKV {
            // KV-shared layers use pre-computed KV from an earlier layer.
            queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
            queries = qNorm(queries)
            kvState = sharedKV
        } else {
            // Only KV-owning layers fall here (KV-shared layers always receive `sharedKV`),
            // so k_proj and k_norm are guaranteed to exist.
            guard let kProj, let kNorm else {
                fatalError(
                    "Gemma4Attention layer \(layerIdx) computed its own K/V but has no k_proj/k_norm; "
                        + "KV-shared layers must be passed `sharedKV`.")
            }

            // Try fused q/k/v projection (single Metal kernel for all three).
            // Only when v_proj exists (!useKeqV) and all three are Gemma-quantized
            // with matching input SRQ scales and aligned dims for the qmv kernel.
            let kRaw: MLXArray
            let vRaw: MLXArray

            if !useKeqV,
               let q = qProj as? GemmaQuantizedLinear,
               let k = kProj as? GemmaQuantizedLinear,
               let vQ = vProj as? GemmaQuantizedLinear,
               let fused = tryFusedQKV(q: q, k: k, v: vQ, x: x) {
                let qd = nHeads * effectiveHeadDim
                let kvd = nKvHeads * effectiveHeadDim
                queries = fused[.ellipsis, 0..<qd].reshaped(B, L, nHeads, effectiveHeadDim)
                kRaw = fused[.ellipsis, qd..<qd + kvd].reshaped(B, L, nKvHeads, effectiveHeadDim)
                vRaw = fused[.ellipsis, (qd + kvd)..<fused.dim(-1)].reshaped(B, L, nKvHeads, effectiveHeadDim)
            } else {
                queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
                kRaw = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
                if let vProj {
                    vRaw = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
                } else {
                    vRaw = kRaw
                }
            }

            queries = qNorm(queries)
            var k = kNorm(kRaw)
            k = k.transposed(0, 2, 1, 3)
            k = applyRotaryPosition(rope, to: k, offset: activePositionOffset)

            var v: MLXArray
            if useKeqV {
                v = vNorm(kRaw)
            } else {
                v = vNorm(vRaw)
            }
            v = v.transposed(0, 2, 1, 3)

            if let quantizedCache = cache as? QuantizedKVCacheProtocol {
                let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
                    keys: k, values: v)
                kvState = .quantized(
                    keys: quantizedKeys,
                    values: quantizedValues,
                    groupSize: quantizedCache.groupSize,
                    bits: quantizedCache.bits,
                    mode: quantizedCache.mode
                )
            } else if let cache {
                let (updatedK, updatedV) = cache.update(keys: k, values: v)
                kvState = .regular(keys: updatedK, values: updatedV)
            } else {
                kvState = .regular(keys: k, values: v)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: activePositionOffset)

        // Adjust mask if cache size differs from mask size
        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = kvState.sequenceLength
            if maskArray.dim(-1) != keysSeqLen {
                adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
            }
        }

        let attentionOutput: MLXArray =
            switch kvState {
            case .regular(let keys, let values):
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: adjustedMask ?? .none
                )
            case .quantized(let keys, let values, let groupSize, let bits, let mode):
                quantizedScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: keys,
                    quantizedValues: values,
                    scale: scale,
                    mask: adjustedMask ?? .none,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }

        let output =
            attentionOutput
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

        return (oProj(output), kvState, activePositionOffset)
    }
}

// MARK: - MLP

private class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        let isKvSharedLayer = layerIdx >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0
        let useDoubleWide = config.useDoubleWideMlp && isKvSharedLayer
        let intermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Router + Experts (E-series)

// MoE block ported from MLXVLM/Models/Gemma4.swift (ml-explore PRs #180, #228). yooz-engine.

private class Gemma4TextRouter: Module {
    let topKExperts: Int
    let hiddenSize: Int
    let rmsNormEps: Float
    private let rootSize: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(_ config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts, let topKExperts = config.topKExperts else {
            fatalError("Gemma4 MoE router requires numExperts and topKExperts in config")
        }
        self.topKExperts = topKExperts
        self.hiddenSize = config.hiddenSize
        self.rmsNormEps = config.rmsNormEps
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let normed = MLXFast.rmsNorm(
            x, weight: (scale * rootSize).asType(x.dtype), eps: rmsNormEps)
        let scores = proj(normed)
        let topKIndices = MLX.argPartition(scores, kth: -topKExperts, axis: -1)[
            .ellipsis, (-topKExperts)...,
        ]
        var topKWeights = MLX.takeAlong(scores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

private class Gemma4TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(_ config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts,
            let moeIntermediateSize = config.moeIntermediateSize
        else {
            fatalError("Gemma4 MoE experts require numExperts and moeIntermediateSize in config")
        }
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let topK = topKIndices.dim(-1)

        let expertOutput = switchGLU(
            x.reshaped(batch * length, hidden),
            topKIndices.reshaped(batch * length, topK)
        )
        let weights = topKWeights.reshaped(batch * length, topK).asType(expertOutput.dtype)
        return weightedExpertSum(expertOutput, weights).reshaped(batch, length, hidden)
    }
}

// MARK: - Decoder Layer

/// One Gemma 4 decoder layer.
///
/// Exposed at `@_spi(GemmaEncoder)` scope so opted-in client code can drive the layer
/// stack directly — e.g. an encoder-style tap that collects every hidden state rather
/// than only the final one. Mirrors the Gemma 3 exposure added in #387.
@_spi(GemmaEncoder) public class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let hiddenSizePerLayerInput: Int
    let enableMoE: Bool

    @ModuleInfo(key: "self_attn") fileprivate var selfAttn: Gemma4Attention
    @ModuleInfo fileprivate var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    // MoE block (E-series): router, experts, and their extra norms
    @ModuleInfo(key: "router") fileprivate var router: Gemma4TextRouter?
    @ModuleInfo(key: "experts") fileprivate var experts: Gemma4TextExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?

    // Per-layer input (PLE) gating
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    // Per-layer scalar
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    // Native compiled-path cache (Phase 6/7 port). Lazily extracted on the
    // first forward pass; `nil` means the native path is not usable (MoE,
    // non-gemma-quant, unaligned dims, no PLE) and the eager path is used.
    // `fileprivate` so `Gemma4TextModelInner.precompileNativeFunctions` can read
    // the cached args for the load-time direct-compile path (Phase 5).
    fileprivate struct NativeArgs {
        let isSource: Bool
        let preFn: @Sendable ([MLXArray]) -> [MLXArray]
        let preArgs: [MLXArray]
        let postFn: @Sendable ([MLXArray]) -> [MLXArray]
        let postArgs: [MLXArray]
    }
    private var _nativeArgs: NativeArgs?
    private var _nativeArgsChecked = false

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        // _addRMSNorm bakes kRMSEps into its compiled graph. Catch a future
        // checkpoint that ships a different rms_norm_eps before it reaches
        // the fused path with the wrong constant.
        precondition(
            config.rmsNormEps == kRMSEps,
            "Gemma4 fused decode path requires rmsNormEps == \(kRMSEps), got \(config.rmsNormEps)"
        )

        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.enableMoE = config.enableMoEBlock

        self._selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)

        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4TextRouter(config)
            self._experts.wrappedValue = Gemma4TextExperts(config)
            self._postFeedforwardLayernorm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1], dtype: .float16)

        super.init()
    }

    /// Lazily extract and cache native compiled-path arguments. Returns `nil` if
    /// the native path is not usable (MoE, non-gemma-quantized, unaligned dims,
    /// or no PLE). Mirrors Python `DecoderLayer._get_native_args`.
    ///
    /// `fileprivate` so `Gemma4TextModelInner.precompileNativeFunctions` can
    /// trigger conversion + read the cached args for the load-time direct-compile
    /// path (Phase 5).
    fileprivate func getNativeArgs() -> NativeArgs? {
        if _nativeArgsChecked { return _nativeArgs }
        _nativeArgsChecked = true

        // Guard: no MoE, PLE present.
        guard !enableMoE,
            let pleGate = perLayerInputGate as? GemmaQuantizedLinear,
            let pleProj = perLayerProjection as? GemmaQuantizedLinear,
            let pleNorm = postPerLayerInputNorm
        else { return nil }

        let attn = selfAttn
        let mlp = self.mlp

        // Check all post-attention linears are gemma-quantized with aligned dims.
        guard let oProj = attn.oProj as? GemmaQuantizedLinear,
            let gateProj = mlp.gateProj as? GemmaQuantizedLinear,
            let upProj = mlp.upProj as? GemmaQuantizedLinear,
            let downProj = mlp.downProj as? GemmaQuantizedLinear,
            let oArgs = oProj.nativeArgs(),
            let gateArgs = gateProj.nativeArgs(),
            let upArgs = upProj.nativeArgs(),
            let downArgs = downProj.nativeArgs(),
            let pleGArgs = pleGate.nativeArgs(),
            let plePArgs = pleProj.nativeArgs()
        else { return nil }

        let isSource = attn.kProj != nil

        // Pre-attention compiled function + args.
        let preFn: @Sendable ([MLXArray]) -> [MLXArray]
        let preArgs: [MLXArray]
        if isSource {
            guard let fused = attn.buildFusedQKVNative(),
                let kNorm = attn.kNorm
            else { return nil }
            preFn = CompiledPreAttnSource.fn(
                nHeads: attn.nHeads, headDim: attn.effectiveHeadDim,
                nKvHeads: attn.nKvHeads)
            // vNorm is RMSNormNoScale (no weight) — use ones.
            let vNormW = MLXArray.ones(
                [attn.effectiveHeadDim], dtype: attn.qNorm.weight.dtype)
            preArgs = [
                inputLayernorm.weight,
                fused.wq, fused.scales, fused.biases, fused.inS, fused.outS,
                attn.qNorm.weight, kNorm.weight, vNormW,
                attn.compiledRopeFreqs,
            ]
        } else {
            guard let qProj = attn.qProj as? GemmaQuantizedLinear,
                let qArgs = qProj.nativeArgs()
            else { return nil }
            preFn = CompiledPreAttnKvshared.fn(
                nHeads: attn.nHeads, headDim: attn.effectiveHeadDim)
            preArgs = [
                inputLayernorm.weight,
                qArgs.packed, qArgs.scales, qArgs.biases, qArgs.inScale, qArgs.outScale,
                attn.qNorm.weight,
                attn.compiledRopeFreqs,
            ]
        }

        // Post-attention compiled function + args.
        let mlpBits = gateProj.numBits
        let pleBits = pleGate.numBits
        let postFn = CompiledPostAttn.fn(mlpBits: mlpBits, pleBits: pleBits)
        let postArgs: [MLXArray] = [
            postAttentionLayernorm.weight,
            preFeedforwardLayernorm.weight,
            postFeedforwardLayernorm.weight,
            pleNorm.weight,
            layerScalar,
            // o_proj (always 4-bit)
            oArgs.packed, oArgs.scales, oArgs.biases, oArgs.inScale, oArgs.outScale,
            // MLP gate/up/down
            gateArgs.packed, gateArgs.scales, gateArgs.biases, gateArgs.inScale, gateArgs.outScale,
            upArgs.packed, upArgs.scales, upArgs.biases, upArgs.inScale, upArgs.outScale,
            downArgs.packed, downArgs.scales, downArgs.biases, downArgs.inScale, downArgs.outScale,
            // PLE gate/proj
            pleGArgs.packed, pleGArgs.scales, pleGArgs.biases, pleGArgs.inScale, pleGArgs.outScale,
            plePArgs.packed, plePArgs.scales, plePArgs.biases, plePArgs.inScale, plePArgs.outScale,
        ]

        let result = NativeArgs(
            isSource: isSource, preFn: preFn, preArgs: preArgs,
            postFn: postFn, postArgs: postArgs)
        _nativeArgs = result
        return result
    }

    /// Free the mobile-format weights on every `GemmaQuantizedLinear` in this
    /// layer after native conversion. Called by the load-time precompilation
    /// (Phase 5) once `getNativeArgs()` has returned non-nil (which means every
    /// relevant linear was converted to the native `quantizedMM` format). The
    /// SRQ scales are preserved. Mirrors the per-layer loop in Python
    /// `precompile_native_functions`.
    fileprivate func freeMobileWeightsOnLinears() {
        let attn = selfAttn
        let mlp = self.mlp
        if let q = attn.qProj as? GemmaQuantizedLinear { q.freeMobileWeights() }
        if let k = attn.kProj as? GemmaQuantizedLinear { k.freeMobileWeights() }
        if let v = attn.vProj as? GemmaQuantizedLinear { v.freeMobileWeights() }
        if let o = attn.oProj as? GemmaQuantizedLinear { o.freeMobileWeights() }
        if let g = mlp.gateProj as? GemmaQuantizedLinear { g.freeMobileWeights() }
        if let u = mlp.upProj as? GemmaQuantizedLinear { u.freeMobileWeights() }
        if let d = mlp.downProj as? GemmaQuantizedLinear { d.freeMobileWeights() }
        if let pli = perLayerInputGate as? GemmaQuantizedLinear { pli.freeMobileWeights() }
        if let plp = perLayerProjection as? GemmaQuantizedLinear { plp.freeMobileWeights() }
    }

    /// Run one decoder layer.
    ///
    /// Exposed at `@_spi(GemmaEncoder)` scope. Callers that do not use per-layer inputs,
    /// KV sharing, or a position offset pass `nil` for those and ignore the corresponding
    /// returned values; the sliding-vs-global attention split is resolved internally from
    /// the layer index, so a client tap does not need to know about it.
    @_spi(GemmaEncoder) public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        positionOffset: RoPEOffset? = nil
    ) -> (MLXArray, Gemma4SharedKVState, RoPEOffset?) {
        let residual = x

        // Native compiled path (Phase 6/7 port): compile the entire pre-attention
        // and post-attention segments with native quantizedMM. KV cache + SDPA stay
        // eager between the two compiled segments. Falls back to the eager path
        // when the native path is not usable (MoE, non-gemma-quant, no PLE).
        if Gemma4TextModel.useNativeCompiledPath,
           let native = getNativeArgs(), let perLayerInput {
            let (isSource, preFn, preArgs, postFn, postArgs) = (
                native.isSource, native.preFn, native.preArgs,
                native.postFn, native.postArgs)
            let attn = selfAttn

            // offset must be an MLXArray (not a bare Int) so `compile` treats it as
            // a runtime input, not a compile-time constant. Otherwise precompilation
            // (cache=nil → 0) and real generation (cache → N) compile different
            // versions (Python Phase 7 bug).
            //
            // Capture the RoPEOffset BEFORE the cache update (the eager path reads
            // `cache.ropeOffset` before `cache.update`, and returns it as
            // `attnPositionOffset`). Reading after the update would give the
            // post-update offset (e.g. 3 after a 3-token prefill) instead of the
            // pre-update offset (0), corrupting RoPE for KV-shared layers.
            let ropeOffset: RoPEOffset
            if let cache = cache {
                ropeOffset = cache.ropeOffset
            } else if let positionOffset {
                ropeOffset = positionOffset
            } else {
                ropeOffset = .scalar(0)
            }
            let offset = ropeOffsetToArray(ropeOffset)

            let (B, L) = (x.dim(0), x.dim(1))
            let attnOutput: MLXArray
            let kvState: Gemma4SharedKVState

            if isSource {
                let preOut = preFn([x] + preArgs + [offset])
                let queries = preOut[0]
                let keys = preOut[1]
                let values = preOut[2]

                // KV cache update (eager — outside the compiled graph).
                let updatedK: MLXArray
                let updatedV: MLXArray
                if let cache = cache {
                    let (k, v) = cache.update(keys: keys, values: values)
                    updatedK = k
                    updatedV = v
                } else {
                    updatedK = keys
                    updatedV = values
                }
                kvState = .regular(keys: updatedK, values: updatedV)

                // Adjust mask if cache size differs from mask size.
                var adjustedMask = mask
                if case .array(let maskArray) = mask {
                    let keysSeqLen = updatedK.dim(2)
                    if maskArray.dim(-1) != keysSeqLen {
                        adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
                    }
                }

                let attnOut = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: updatedK,
                    values: updatedV,
                    scale: attn.scale,
                    mask: adjustedMask ?? .none
                )
                attnOutput = attnOut.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            } else {
                let queries = preFn([x] + preArgs + [offset])[0]

                // Use shared KV from the source layer.
                guard let sharedKV else {
                    // KV-shared layer without shared KV — shouldn't happen in normal
                    // generation, but handle gracefully by falling back.
                    return eagerCall(
                        x, mask: mask, cache: cache, perLayerInput: perLayerInput,
                        sharedKV: sharedKV, positionOffset: positionOffset)
                }

                switch sharedKV {
                case .regular(let keys, let values):
                    let attnOut = MLXFast.scaledDotProductAttention(
                        queries: queries, keys: keys, values: values,
                        scale: attn.scale, mask: mask ?? .none)
                    attnOutput = attnOut.transposed(0, 2, 1, 3).reshaped(B, L, -1)
                    kvState = sharedKV
                case .quantized:
                    // Quantized shared KV — fall back to eager for now.
                    return eagerCall(
                        x, mask: mask, cache: cache, perLayerInput: perLayerInput,
                        sharedKV: sharedKV, positionOffset: positionOffset)
                }
            }

            // Post-attention compiled segment (o_proj + norms + MLP + PLE + layer_scalar).
            let h = postFn([residual, attnOutput, perLayerInput] + postArgs)[0]

            // Return the RoPEOffset captured BEFORE the cache update (matches the
            // eager path's `attnPositionOffset`).
            return (h, kvState, ropeOffset)
        }

        return eagerCall(x, mask: mask, cache: cache, perLayerInput: perLayerInput,
            sharedKV: sharedKV, positionOffset: positionOffset)
    }

    /// The existing eager (non-compiled) forward path. Used as a fallback when
    /// the native compiled path is not usable.
    private func eagerCall(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode?,
        cache: KVCache?,
        perLayerInput: MLXArray?,
        sharedKV: Gemma4SharedKVState?,
        positionOffset: RoPEOffset?
    ) -> (MLXArray, Gemma4SharedKVState, RoPEOffset?) {
        let residual = x

        let h = inputLayernorm(x)
        let (attnOut, kvPair, attnPositionOffset) = selfAttn(
            h, mask: mask, cache: cache, sharedKV: sharedKV, positionOffset: positionOffset)
        // Fused: residual + RMSNorm(attnOut) * weight
        var out = _addRMSNorm(residual, attnOut, postAttentionLayernorm.weight)

        let residual2 = out

        if enableMoE,
            let router,
            let experts,
            let pfln1 = postFeedforwardLayernorm1,
            let pfln2 = postFeedforwardLayernorm2,
            let pfln2pre = preFeedforwardLayernorm2
        {
            // Dense branch
            var dense = preFeedforwardLayernorm(out)
            dense = mlp(dense)
            dense = pfln1(dense)

            // Sparse branch
            let (topKIndices, topKWeights) = router(out)
            var sparse = pfln2pre(out)
            sparse = experts(sparse, topKIndices: topKIndices, topKWeights: topKWeights)
            sparse = pfln2(sparse)

            // Combine then apply shared post-ff norm
            out = _addRMSNorm(residual2, dense + sparse, postFeedforwardLayernorm.weight)
        } else {
            out = preFeedforwardLayernorm(out)
            out = mlp(out)
            // Fused: residual + RMSNorm(out) * weight
            out = _addRMSNorm(residual2, out, postFeedforwardLayernorm.weight)
        }

        // PLE gating
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let perLayerInput
        {
            let residual3 = out
            var g = gate(out)
            // Fused: gelu_approx(g) * perLayerInput
            g = _geluMul(g, perLayerInput)
            g = proj(g)
            // Fused: residual + RMSNorm(g) * weight
            out = _addRMSNorm(residual3, g, norm.weight)
        }

        out = out * layerScalar

        return (out, kvPair, attnPositionOffset)
    }
}

// MARK: - Text Model

/// The Gemma 4 text backbone: embeddings, the decoder stack, and the final norm.
///
/// Exposed at `@_spi(GemmaEncoder)` scope for encoder-style client taps.
@_spi(GemmaEncoder) public class Gemma4TextModelInner: Module {
    @_spi(GemmaEncoder) public let config: Gemma4TextConfiguration
    /// Multiplier applied to the token embeddings.
    ///
    /// A plain `Float` — NOT pre-rounded to bfloat16 the way Gemma 3's scale is. Client
    /// taps replicating the forward must apply it in this precision.
    @_spi(GemmaEncoder) public let embedScale: Float
    let perLayerProjectionScale: Float
    let hiddenSizePerLayerInput: Int

    /// Token embedding table. Callers must scale its output by ``embedScale``.
    @ModuleInfo(key: "embed_tokens") @_spi(GemmaEncoder) public var embedTokens: Embedding
    /// The decoder stack, exposed for encoder-style client taps.
    @ModuleInfo(key: "layers") @_spi(GemmaEncoder) public var layers: [Gemma4DecoderLayer]
    /// Final norm. Plain `RMSNorm` — Gemma 4 has NO `1 + weight` shift, unlike Gemma 3.
    @ModuleInfo @_spi(GemmaEncoder) public var norm: RMSNorm

    // Per-layer embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    // KV sharing mapping: for each layer, which earlier layer provides KVs
    let previousKvs: [Int]
    let firstKvSharedLayerIdx: Int

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = Float(config.hiddenSize).squareRoot()
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // PLE
        if config.hiddenSizePerLayerInput > 0 {
            self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput)
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.numHiddenLayers * config.hiddenSizePerLayerInput,
                bias: false)
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            self.perLayerProjectionScale = 1.0
        }

        // Build KV-sharing map
        self.firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        var kvMap = Array(0 ..< config.numHiddenLayers)
        if config.numKvSharedLayers > 0 {
            // Find the last non-shared layer of each type
            var lastByType = [String: Int]()
            for i in 0 ..< firstKvSharedLayerIdx {
                lastByType[config.layerTypes[i]] = i
            }
            // Shared layers reference the last non-shared layer of the same type
            for j in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                if let prev = lastByType[config.layerTypes[j]] {
                    kvMap[j] = prev
                }
            }
        }
        self.previousKvs = kvMap

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let inputEmbeddings = embedTokens(inputs)
        var h = inputEmbeddings * embedScale

        // Compute per-layer inputs (PLE)
        var perLayerInputs: [MLXArray?]
        if hiddenSizePerLayerInput > 0,
            let embedPerLayer = embedTokensPerLayer,
            let modelProj = perLayerModelProjection,
            let projNorm = perLayerProjectionNorm
        {
            // Token-based PLE
            let tokenPLE =
                embedPerLayer(inputs)
                * Float(config.hiddenSizePerLayerInput).squareRoot()

            // [B, L, numLayers * hiddenSizePerLayerInput] -> [B, L, numLayers, hiddenSizePerLayerInput]
            let reshapedTokenPLE = tokenPLE.reshaped(
                tokenPLE.dim(0), tokenPLE.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)

            // Model projection PLE
            let modelPLE = (modelProj(h) * perLayerProjectionScale).reshaped(
                h.dim(0), h.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)
            let normedModelPLE = projNorm(modelPLE)

            // Combine: (model_proj + token_embed) * 2^{-0.5}
            let perLayerInputScale = pow(Float(2.0), -0.5)
            let combined = (normedModelPLE + reshapedTokenPLE) * perLayerInputScale

            perLayerInputs = (0 ..< config.numHiddenLayers).map { i in
                combined[.ellipsis, i, 0...]
            }
        } else {
            perLayerInputs = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // Extend cache array for shared layers (which get nil caches)
        var fullCache: [KVCache?]
        if let cache {
            fullCache = cache.map { Optional($0) }
            while fullCache.count < config.numHiddenLayers {
                fullCache.append(nil)
            }
        } else {
            fullCache = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // Build masks: one per attention type
        var maskByType = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
        for (i, layer) in layers.enumerated() {
            let lt = layer.layerType
            if maskByType[lt] == nil {
                if lt == "sliding_attention" {
                    maskByType[lt] = createAttentionMask(
                        h: h, cache: fullCache[i], windowSize: config.slidingWindow)
                } else {
                    maskByType[lt] = createAttentionMask(h: h, cache: fullCache[i])
                }
            }
        }

        // Forward through layers, tracking intermediate KV pairs for sharing
        var intermediates = [(kv: Gemma4SharedKVState?, positionOffset: RoPEOffset?)](
            repeating: (nil, nil), count: config.numHiddenLayers)

        for (idx, layer) in layers.enumerated() {
            let prevIdx = previousKvs[idx]
            let sharedKV = intermediates[prevIdx].kv
            let sharedPositionOffset = intermediates[prevIdx].positionOffset

            let mask = maskByType[layer.layerType]
            let (out, kvPair, positionOffset) = layer(
                h,
                mask: mask,
                cache: fullCache[idx],
                perLayerInput: perLayerInputs[idx],
                sharedKV: sharedKV,
                positionOffset: sharedPositionOffset
            )
            h = out
            intermediates[idx] = (kvPair, positionOffset)
        }

        return norm(h)
    }

    // MARK: - Load-time precompilation (Phase 5)

    /// Precompile native compiled functions for common prompt lengths and free
    /// mobile-format weights. Called at load time (via `Gemma4TextModel`'s
    /// `NativePrecompilable` conformance) after weights are loaded and modules
    /// are replaced. Mirrors Python `precompile_native_functions`.
    ///
    /// Three steps:
    /// 1. **Convert + free mobile weights layer by layer** — call `getNativeArgs()`
    ///   on each decoder layer (triggering lazy `nativeArgs()` conversion to the
    ///   native `quantizedMM` format), then replace the mobile `weight`/
    ///   `weightScale` with dummy arrays. Keeps the conversion peak at roughly
    ///   half-mobile + half-native per layer (the native weights are `eval`'d
    ///   inside `convertToMLXFormat`).
    /// 2. **Precompile `compile` functions** — run dummy forward passes for
    ///   common prompt lengths with `eval` on the output. Use the hybrid compile
    ///   strategy: full forward pass for shapes ≤ 32 (negligible activations,
    ///   warms up MLX built-in ops), direct compiled-function calls for larger
    ///   shapes (avoids accumulating ~0.8 GB of activations across all layers).
    /// 3. The compiled functions are shared across layers via the factory caches,
    ///   so only the first source layer and first KV-shared layer are needed for
    ///   the direct-compile path.
    ///
    /// No-op if no layer uses the native path (e.g. unaligned dims, MoE, no PLE).
    fileprivate func precompileNativeFunctions(shapes: [Int] = [1, 16, 32, 64, 128, 256]) {
        let layers = self.layers
        guard !layers.isEmpty else { return }

        // 1. Convert + free mobile weights layer by layer.
        var anyNative = false
        for layer in layers {
            guard layer.getNativeArgs() != nil else { continue }
            anyNative = true
            layer.freeMobileWeightsOnLinears()
        }
        guard anyNative else { return }

        // 2. Precompile compile functions for common prompt lengths.
        let hiddenSize = config.hiddenSize
        let perLayerDim = config.hiddenSizePerLayerInput
        let dtype = layers[0].inputLayernorm.weight.dtype

        // Find the first source and first KV-shared layer (different compiled
        // pre-attention functions). Their compiled functions are shared across
        // all layers via the factory caches, so only these are needed for the
        // direct-compile path.
        var firstSource: Gemma4DecoderLayer?
        var firstKvshared: Gemma4DecoderLayer?
        for layer in layers {
            guard let native = layer.getNativeArgs() else { continue }
            if native.isSource {
                if firstSource == nil { firstSource = layer }
            } else {
                if firstKvshared == nil { firstKvshared = layer }
            }
            if firstSource != nil && firstKvshared != nil { break }
        }
        let compileLayers = [firstSource, firstKvshared].compactMap { $0 }

        // offset must be an MLXArray (not a bare Int) so `compile` treats it as a
        // runtime input, not a compile-time constant (Python Phase 7 bug).
        let dummyOffset = MLXArray(Int32(0))
        let fullPassShapes = Set(shapes.filter { $0 <= 32 })

        for seqLen in shapes {
            if fullPassShapes.contains(seqLen) {
                // Full forward pass: warms up the compiled functions AND the MLX
                // built-in ops (RMSNorm, SDPA, RoPE, embeddings, PLE) with
                // negligible activation memory (~0.1 GB for seq_len=32).
                let dummyIds = MLXArray(
                    Array(repeating: Int32(2), count: seqLen)
                ).reshaped([1, seqLen])
                let out = self(dummyIds, cache: nil)
                eval(out)
            } else {
                // Direct compile: call the compiled functions with tiny dummy
                // inputs (~4 MB) to avoid accumulating ~0.8 GB of activations
                // across all layers for large seq_len.
                //
                // `dummyAttnOut` must match the o_proj input dim
                // (`nHeads * effectiveHeadDim`), which differs from `hiddenSize`
                // for Gemma 4 (sliding=2048, full=4096 vs hiddenSize=1536). The
                // Python uses `hidden_size` here and relies on `try/except` to
                // swallow the resulting `quantized_matmul` shape error, leaving the
                // post-attention segment un-precompiled for large seq_len. Swift's
                // `quantized_matmul` aborts on shape mismatch, so we use the
                // correct per-layer attention output dim and actually precompile it.
                let dummyX = MLXArray.zeros([1, seqLen, hiddenSize], dtype: dtype)
                let dummyResidual = MLXArray.zeros([1, seqLen, hiddenSize], dtype: dtype)
                let dummyPli = MLXArray.zeros([1, seqLen, perLayerDim], dtype: dtype)

                for layer in compileLayers {
                    guard let native = layer.getNativeArgs() else { continue }
                    let preOut = native.preFn([dummyX] + native.preArgs + [dummyOffset])
                    eval(preOut)
                    let attnDim = layer.selfAttn.nHeads * layer.selfAttn.effectiveHeadDim
                    let dummyAttnOut = MLXArray.zeros([1, seqLen, attnDim], dtype: dtype)
                    let postOut = native.postFn(
                        [dummyResidual, dummyAttnOut, dummyPli] + native.postArgs)
                    eval(postOut)
                }
            }
        }
    }
}

// MARK: - Public Model

public class Gemma4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    /// Set to `false` to force the eager (non-compiled) decoder path for A/B
    /// testing. Defaults to `true` (native compiled path enabled).
    nonisolated(unsafe) public static var useNativeCompiledPath = true

    /// Set to `false` to skip load-time precompilation + weight freeing (the
    /// `NativePrecompilable` hook in `loadWeights`). Defaults to `true`.
    ///
    /// The A/B equivalence test sets this to `false` so the eager path (which
    /// needs the mobile-format weights) still works after `loadWeights` — once
    /// weights are freed the eager path can no longer dequantize them. In
    /// production this stays `true` so load converts to the native `quantizedMM`
    /// format, frees the mobile weights, and warms up the per-shape `compile`
    /// graphs.
    nonisolated(unsafe) public static var precompileAtLoad = true

    fileprivate let config: Gemma4TextConfiguration
    /// The text backbone, exposed at `@_spi(GemmaEncoder)` scope for client taps.
    @_spi(GemmaEncoder) public let model: Gemma4TextModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = Gemma4TextModelInner(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        out = tanh(out / config.finalLogitSoftcapping) * config.finalLogitSoftcapping
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let weights = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: config.tieWordEmbeddings)

        // MoE expert weight remapping ported from MLXVLM/Models/Gemma4.swift. yooz-engine.
        // HuggingFace stores expert weights as fused gate_up_proj; SwitchGLU expects
        // separate gate_proj and up_proj, each shaped [numExperts, hiddenDims, inputDims].
        let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            let k = key
            // Skip vision/audio/rotary weights
            if k.contains("self_attn.rotary_emb")
                || k.contains("input_max")
                || k.contains("input_min")
                || k.contains("output_max")
                || k.contains("output_min")
            {
                continue
            }
            // Drop redundant k_proj/v_proj/k_norm for KV-shared layers: they reuse an
            // earlier layer's K/V and own no K projection or K norm, so the module tree
            // has none. QAT checkpoints already omit these; some (PTQ) checkpoints still
            // ship them, and keeping them would be an unexpected weight. Dropping here
            // makes both load against the same tree. (v_norm is parameter-free.)
            if firstKvSharedLayerIdx > 0,
                k.contains("self_attn.k_proj")
                    || k.contains("self_attn.v_proj")
                    || k.contains("self_attn.k_norm"),
                let layerIdx = Self.decoderLayerIndex(in: k),
                layerIdx >= firstKvSharedLayerIdx
            {
                continue
            }

            // Remap .experts.down_proj -> .experts.switch_glu.down_proj.weight
            if k.hasSuffix(".experts.down_proj") {
                sanitized[
                    k.replacingOccurrences(
                        of: ".experts.down_proj",
                        with: ".experts.switch_glu.down_proj.weight"
                    )
                ] = value
                continue
            }

            // Remap .experts.gate_up_proj -> split into gate_proj + up_proj
            if k.hasSuffix(".experts.gate_up_proj") {
                let mid = value.dim(-2) / 2
                sanitized[
                    k.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.gate_proj.weight"
                    )
                ] = value[.ellipsis, ..<mid, 0...]
                sanitized[
                    k.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.up_proj.weight"
                    )
                ] = value[.ellipsis, mid..., 0...]
                continue
            }

            sanitized[k] = value
        }
        return sanitized
    }

    /// Sanitize with access to safetensor metadata. Wraps the existing MoE
    /// expert remap (`sanitize(weights:)`) and, for Gemma 4 QAT mobile
    /// checkpoints (`quant_method: "gemma"`), remaps `embedding_quantized` →
    /// `weight` and swaps `Linear`/`Embedding` leaves for their Gemma quantized
    /// counterparts. This is the top-level entry for `gemma4_text` checkpoints;
    /// for `gemma4` the wrapper `Gemma4Model.sanitize(weights:metadata:)` drives
    /// the mobile path on the full `language_model.*` namespace instead.
    public func sanitize(weights: [String: MLXArray], metadata: [String: String])
        -> [String: MLXArray]
    {
        var sanitized = sanitize(weights: weights)
        if let qc = config.quantizationConfig, qc.isGemmaMobile {
            sanitized = applyGemmaMobileQuantization(
                model: self, weights: sanitized, config: qc)
        }
        return sanitized
    }

    /// Extract `N` from a weight key shaped like `…layers.N.…`, else nil.
    private static func decoderLayerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let digits = key[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers

        return try (0 ..< firstKvShared).map { i in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: config.slidingWindow,
                usesSlidingWindow: config.layerTypes[i] != "full_attention")
        }
    }
}

// MARK: - Load-time precompilation (Phase 5)

extension Gemma4TextModel: NativePrecompilable {
    /// Precompile native compiled functions and free mobile-format weights.
    ///
    /// Called by `loadWeights` after weights are loaded and modules are replaced.
    /// Honors ``precompileAtLoad``: when `false` (e.g. the A/B equivalence test),
    /// this is a no-op so the eager path keeps the mobile-format weights.
    public func precompileNativeFunctions() {
        guard Self.precompileAtLoad else { return }
        model.precompileNativeFunctions()
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    /// Decoder layers, not just attention: LoRA keys are matched against the
    /// children of each returned module, so `mlp.*` targets only resolve when
    /// the layer itself is returned (as Gemma3Text / Qwen35 / Llama do).
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

extension Gemma4TextModel {
    public var toolCallFormat: ToolCallFormat? { .gemma4 }
}
