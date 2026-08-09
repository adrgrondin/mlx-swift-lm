// Copyright © 2026 DeepGrove AI.
//
// Portable Swift port of Maple from
// https://github.com/deepgrove-ai/mlx-lm (commit eba96c16158f032821b0bf374ea1421cfddef0a9).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct MapleQuantizationConfiguration: Codable, Sendable {
    public var groupSize: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 128
    }
}

public struct MapleConfiguration: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var numExperts: Int
    public var numExpertsPerToken: Int
    public var firstKDenseReplace: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var partialRotaryFactor: Float
    public var maxPositionEmbeddings: Int
    public var vocabularySize: Int
    public var slidingWindow: Int
    public var layerTypes: [String]
    public var useQKNorm: Bool
    public var useBias: Bool
    public var tieWordEmbeddings: Bool
    public var flashHead: MapleFlashHeadMetadata?
    public var quantization: MapleQuantizationConfiguration?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case vocabularySize = "vocab_size"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case useQKNorm = "use_qk_norm"
        case useBias = "use_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case flashHead = "flash_head"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "maple"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 256
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 140_000
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        let decodedTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        layerTypes =
            decodedTypes?.isEmpty == false
            ? decodedTypes! : Array(repeating: "full_attention", count: hiddenLayers)
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        flashHead = try c.decodeIfPresent(MapleFlashHeadMetadata.self, forKey: .flashHead)
        quantization = try c.decodeIfPresent(
            MapleQuantizationConfiguration.self, forKey: .quantization)
    }
}

extension MapleConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard hiddenSize > 0, intermediateSize > 0, moeIntermediateSize > 0,
            hiddenLayers > 0, attentionHeads > 0, kvHeads > 0, headDim > 0,
            vocabularySize > 0, slidingWindow > 0
        else {
            throw ModelFactoryError.invalidConfiguration("Maple dimensions must be positive")
        }
        guard hiddenSize == attentionHeads * headDim else {
            throw ModelFactoryError.invalidConfiguration(
                "Maple hidden_size must equal num_attention_heads * head_dim")
        }
        guard attentionHeads.isMultiple(of: kvHeads) else {
            throw ModelFactoryError.invalidConfiguration(
                "Maple num_attention_heads must be divisible by num_key_value_heads")
        }
        guard numExperts > 0, numExpertsPerToken > 0, numExpertsPerToken <= numExperts else {
            throw ModelFactoryError.invalidConfiguration("Maple expert counts are invalid")
        }
        guard firstKDenseReplace >= 0, firstKDenseReplace <= hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration("Maple first_k_dense_replace is invalid")
        }
        guard layerTypes.count == hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "Maple layer_types count must equal num_hidden_layers")
        }
        guard layerTypes.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" }) else {
            throw ModelFactoryError.invalidConfiguration(
                "Maple layer_types contains an unknown type")
        }
        if layerTypes.contains("sliding_attention") {
            let rotaryDimensions = Int(Float(headDim) * partialRotaryFactor)
            guard rotaryDimensions > 0, rotaryDimensions.isMultiple(of: 2) else {
                throw ModelFactoryError.invalidConfiguration(
                    "Maple sliding attention requires a positive even rotary dimension")
            }
        }
        try validateRoPEConfiguration(ropeScaling, context: "MapleConfiguration.rope_scaling")
    }
}

final class MapleRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(
            x.asType(.float32), weight: weight.asType(.float32), eps: eps
        ).asType(x.dtype)
    }
}

final class MapleAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let useRoPE: Bool
    let rotaryDimensions: Int
    let eps: Float
    let ropeTheta: Float

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MapleRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: MapleRMSNorm?

    let rope: RoPELayer

    /// Fused Q/K norm+RoPE state: nil = unprobed, then true/false (latched).
    var fusedQK: Bool?
    /// Per-head norm weights broadcast to `[heads, headDim]`, and the rope
    /// inverse frequencies, built lazily for the fused kernel.
    var fusedQKWeights: MLXArray?
    var fusedInvFreq: MLXArray?

    /// The kernel pairs rope dims directly from `rope_theta`, so it only
    /// matches the portable path when there is no rope scaling.
    let fusedQKEligible: Bool

    init(_ config: MapleConfiguration, layerType: String) {
        attentionHeads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        useRoPE = layerType == "sliding_attention"
        eps = config.rmsNormEps
        ropeTheta = config.ropeTheta

        _qkvProj.wrappedValue = Linear(
            config.hiddenSize,
            (config.attentionHeads + 2 * config.kvHeads) * config.headDim,
            bias: config.useBias)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: config.useBias)
        if config.useQKNorm {
            _qNorm.wrappedValue = MapleRMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = MapleRMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        }

        rotaryDimensions = max(2, Int(Float(config.headDim) * config.partialRotaryFactor))
        fusedQKEligible =
            mapleFusedKernelsEnabled
            && config.useQKNorm && config.ropeScaling == nil
            && config.headDim > 0 && config.headDim % 32 == 0
            && (!useRoPE
                || (rotaryDimensions > 0 && rotaryDimensions % 2 == 0
                    && rotaryDimensions <= config.headDim))
        rope = initializeRope(
            dims: rotaryDimensions, base: config.ropeTheta, traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    /// Build the concatenated per-head norm weights and rope frequencies once.
    func prepareFusedQK() {
        guard fusedQKWeights == nil else { return }
        let weights = contiguous(
            concatenated([
                broadcast(
                    qNorm!.weight.expandedDimensions(axis: 0),
                    to: [attentionHeads, headDim]),
                broadcast(
                    kNorm!.weight.expandedDimensions(axis: 0),
                    to: [kvHeads, headDim]),
            ]))
        let invFreq: MLXArray
        if useRoPE {
            let half = rotaryDimensions / 2
            let exponents = MLXArray(0 ..< half).asType(.float32) / Float(half)
            invFreq = pow(MLXArray(ropeTheta), -exponents)
        } else {
            invFreq = MLXArray.ones([1], dtype: .float32)
        }
        fusedQKWeights = weights
        fusedInvFreq = invFreq
        eval(weights, invFreq)
    }

    func fusedQKDecode(_ qk: MLXArray, offset: Int) -> MLXArray {
        prepareFusedQK()
        // The kernel reads x, w, and out through one templated pointer type,
        // so the norm weights must share the activation dtype (float32 math
        // inside the kernel is unaffected by the storage cast).
        if fusedQKWeights!.dtype != qk.dtype {
            let cast = fusedQKWeights!.asType(qk.dtype)
            eval(cast)
            fusedQKWeights = cast
        }
        return MapleQKNormRoPEKernel.callAsFunction(
            qk, weights: fusedQKWeights!, invFreq: fusedInvFreq!, offset: offset, eps: eps,
            headDim: headDim, ropeDim: useRoPE ? rotaryDimensions : 0)
    }

    /// The same result from stock ops: fallback, and the yardstick the fused
    /// kernel is checked against.
    func referenceQKDecode(_ qk: MLXArray, offset: Int) -> MLXArray {
        var queries = qk[0 ..< attentionHeads].reshaped(1, attentionHeads, 1, headDim)
        var keys = qk[attentionHeads...].reshaped(1, kvHeads, 1, headDim)
        if let qNorm { queries = qNorm(queries) }
        if let kNorm { keys = kNorm(keys) }
        if useRoPE {
            queries = applyRotaryPosition(rope, to: queries, offset: .scalar(offset))
            keys = applyRotaryPosition(rope, to: keys, offset: .scalar(offset))
        }
        return concatenated([queries, keys], axis: 1).reshaped(qk.shape)
    }

    /// Probe the fused kernel against the reference at a nonzero position, so
    /// a broken rotation cannot pass.
    private func probeFusedQK(_ qk: MLXArray) -> Bool {
        mapleKernelMatches(
            fast: { [fusedQKDecode(qk, offset: 7)] },
            reference: { [referenceQKDecode(qk, offset: 7)] })
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let qkv = qkvProj(x)

        if batch == 1 && length == 1 && fusedQKEligible && !(cache is BatchPositionedKVCache) {
            // Single-token decode: one dispatch for both Q/K norms and both
            // rope applications instead of four.
            let nQ = attentionHeads
            let qkSize = (attentionHeads + kvHeads) * headDim
            let flat = qkv.reshaped(-1)
            let qk = flat[0 ..< qkSize].reshaped(attentionHeads + kvHeads, headDim)
            if fusedQK == nil {
                fusedQK = probeFusedQK(qk)
            }
            let offset = cache?.offset ?? 0
            let out =
                fusedQK == true
                ? fusedQKDecode(qk, offset: offset)
                : referenceQKDecode(qk, offset: offset)
            let queries = out[0 ..< nQ].reshaped(1, nQ, 1, headDim)
            let keys = out[nQ...].reshaped(1, kvHeads, 1, headDim)
            let values = flat[qkSize...].reshaped(1, kvHeads, 1, headDim)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache,
                scale: scale, mask: mask)
            return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }

        let qSize = attentionHeads * headDim
        let kSize = kvHeads * headDim
        let parts = split(qkv, indices: [qSize, qSize + kSize], axis: -1)

        var queries = parts[0].reshaped(batch, length, attentionHeads, headDim)
        var keys = parts[1].reshaped(batch, length, kvHeads, headDim)
        let values = parts[2].reshaped(batch, length, kvHeads, headDim)
        if let qNorm { queries = qNorm(queries) }
        if let kNorm { keys = kNorm(keys) }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        let transposedValues = values.transposed(0, 2, 1, 3)
        if useRoPE {
            queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
            keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: transposedValues, cache: cache,
            scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class MapleMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: MapleConfiguration) {
        _gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.useBias)
        _upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.useBias)
        _downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class MapleGate: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let topK: Int
    let numExperts: Int
    let hiddenSize: Int

    /// Fused-router state: nil = unprobed, then true/false (latched).
    var fusedRouter: Bool?
    /// Persistent arrival counter for the fused router; private to this gate.
    var routerCounter: MLXArray?

    init(_ config: MapleConfiguration) {
        topK = config.numExpertsPerToken
        numExperts = config.numExperts
        hiddenSize = config.hiddenSize
        _weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])
    }

    private var fusedRouterEligible: Bool {
        mapleFusedKernelsEnabled
            && MapleFusedRouterKernel.supports(
                topK: topK, numExperts: numExperts, hiddenSize: hiddenSize)
    }

    func fusedCall(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        if routerCounter == nil {
            let counter = MLXArray.zeros([8], dtype: .uint32)
            eval(counter)
            routerCounter = counter
        }
        return MapleFusedRouterKernel.callAsFunction(
            x, weight: weight, counter: routerCounter!, topK: topK,
            numExperts: numExperts, hiddenSize: hiddenSize)
    }

    func referenceCall(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let logits = x.asType(.float32).matmul(weight.asType(.float32).T)
        let probabilities = softmax(logits, axis: -1, precise: true)
        let indices = stopGradient(
            argPartition(probabilities, kth: -topK, axis: -1)[.ellipsis, (-topK)...])
        var scores = takeAlong(probabilities, indices, axis: -1)
        scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        return (indices, scores)
    }

    /// Probe the fused router on a live single-token activation. The two paths
    /// may order the selected experts differently, and an exact tie at the
    /// top-k boundary may legitimately pick either expert, so compare sorted
    /// score vectors and bound-check the ids (a bad id indexes the expert
    /// gather).
    private func probeFusedRouter(_ x: MLXArray) -> Bool {
        do {
            return try withError {
                let (indices, scores) = fusedCall(x)
                let (referenceIndices, referenceScores) = referenceCall(x)
                eval(indices, scores, referenceIndices, referenceScores)
                guard indices.shape == referenceIndices.shape else { return false }
                let inBounds = ((indices .>= 0) .&& (indices .< numExperts)).all()
                eval(inBounds)
                guard inBounds.item(Bool.self) else { return false }
                let close = sorted(scores, axis: -1).allClose(
                    sorted(referenceScores, axis: -1), rtol: 1e-5, atol: 1e-5)
                eval(close)
                return close.item(Bool.self)
            }
        } catch {
            return false
        }
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        if fusedRouter != false, fusedRouterEligible, x.size == hiddenSize {
            if fusedRouter == nil {
                fusedRouter = probeFusedRouter(x)
            }
            if fusedRouter == true {
                return fusedCall(x)
            }
        }
        return referenceCall(x)
    }
}

// Compile the elementwise expert work into one graph per operation. Stock MLX already
// supplies Maple's fast gathered 2-bit QMV; these remove dispatches around those QMV calls.
let mapleClampedSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    let gateMaximum = MLXArray(Float(7)).asType(gate.dtype)
    let upMinimum = MLXArray(Float(-7)).asType(up.dtype)
    let upMaximum = MLXArray(Float(7)).asType(up.dtype)
    return silu(minimum(gate, gateMaximum)) * clip(up, min: upMinimum, max: upMaximum)
}

let mapleAggregateExpertOutputs: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, scores in
    (outputs.asType(.float32) * expandedDimensions(scores, axis: -1))
        .sum(axis: -2).asType(outputs.dtype)
}

final class MapleSwitchGLU: Module {
    @ModuleInfo(key: "up_gate_proj") var upGateProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(_ config: MapleConfiguration) {
        _upGateProj.wrappedValue = SwitchLinear(
            inputDims: config.hiddenSize, outputDims: 2 * config.moeIntermediateSize,
            numExperts: config.numExperts, bias: config.useBias)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: config.moeIntermediateSize, outputDims: config.hiddenSize,
            numExperts: config.numExperts, bias: config.useBias)
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = expandedDimensions(x, axes: [-2, -3])
        let shouldSort = indices.size >= 64
        var selected = indices
        var inverseOrder = MLXArray()
        if shouldSort {
            (input, selected, inverseOrder) = gatherSort(x: input, indices: indices)
        }

        let projected = upGateProj(input, selected, sortedIndices: shouldSort)
        let parts = split(projected, parts: 2, axis: -1)
        let activated = mapleClampedSwiGLU(parts[1], parts[0])
        var output = downProj(activated, selected, sortedIndices: shouldSort)
        if shouldSort {
            output = scatterUnsort(x: output, invOrder: inverseOrder, shape: indices.shape)
        }
        return output.squeezed(axis: -2)
    }
}

final class MapleSparseMoeBlock: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: MapleGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: MapleSwitchGLU

    init(_ config: MapleConfiguration) {
        _gate.wrappedValue = MapleGate(config)
        _switchMLP.wrappedValue = MapleSwitchGLU(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        let outputs = switchMLP(x, indices: indices)
        return mapleAggregateExpertOutputs(outputs, scores)
    }
}

final class MapleDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MapleAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MapleRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MapleRMSNorm

    init(_ config: MapleConfiguration, index: Int) {
        _selfAttention.wrappedValue = MapleAttention(config, layerType: config.layerTypes[index])
        _inputLayerNorm.wrappedValue = MapleRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = MapleRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if index < config.firstKDenseReplace {
            _mlp.wrappedValue = MapleMLP(config)
        } else {
            _mlp.wrappedValue = MapleSparseMoeBlock(config)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let hidden = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        return hidden + mlp(postAttentionLayerNorm(hidden))
    }
}

public final class MapleModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    let layers: [MapleDecoderLayer]
    @ModuleInfo(key: "norm") var norm: MapleRMSNorm

    let layerTypes: [String]
    let slidingWindow: Int
    let slidingAttentionIndex: Int?
    let fullAttentionIndex: Int?

    /// Fused add+norm decode state: nil = unprobed, then true/false (latched).
    var fusedAddNorm: Bool?
    /// Initial residual for the fused decode loop; `x + 0` is exact in bf16.
    var fusedZero: MLXArray?

    init(_ config: MapleConfiguration) {
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map { MapleDecoderLayer(config, index: $0) }
        _norm.wrappedValue = MapleRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        layerTypes = config.layerTypes
        slidingWindow = config.slidingWindow
        slidingAttentionIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention")
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = wordEmbeddings(inputs)
        let caches: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        if let index = fullAttentionIndex {
            fullMask = createAttentionMask(h: hidden, cache: caches[index])
        }
        if let index = slidingAttentionIndex {
            slidingMask = createAttentionMask(
                h: hidden, cache: caches[index], windowSize: slidingWindow)
        }

        if hidden.size == hidden.dim(-1) {
            // Single-token decode is bounded by the serial dispatch chain:
            // fold each residual add into the following norm (one dispatch per
            // add+norm pair) when the kernel matches the portable semantics.
            if fusedAddNorm == nil {
                fusedAddNorm =
                    mapleFusedKernelsEnabled
                    && MapleAddRMSNormKernel.supports(dimensions: hidden.dim(-1))
                    && MapleAddRMSNormKernel.probe(
                        dimensions: hidden.dim(-1), dtype: hidden.dtype,
                        weight: norm.weight, eps: norm.eps)
            }
            if fusedAddNorm == true {
                return decodeFused(
                    hidden, caches: caches, fullMask: fullMask, slidingMask: slidingMask)
            }
        }

        for (index, layer) in layers.enumerated() {
            let mask = layerTypes[index] == "sliding_attention" ? slidingMask! : fullMask!
            hidden = layer(hidden, mask: mask, cache: caches[index])
        }
        return norm(hidden)
    }

    /// Decode loop with residual adds folded into the norms. Carries `(h, r)`
    /// instead of adding `r` back each step; identical arithmetic: the kernel
    /// rounds the sum once (as the portable add did) and norms the rounded
    /// stream with a float32 weight multiply.
    private func decodeFused(
        _ h0: MLXArray, caches: [KVCache?],
        fullMask: MLXFast.ScaledDotProductAttentionMaskMode?,
        slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?
    ) -> MLXArray {
        if fusedZero == nil || fusedZero!.dtype != h0.dtype || fusedZero!.shape != h0.shape {
            let zero = MLXArray.zeros(h0.shape, dtype: h0.dtype)
            eval(zero)
            fusedZero = zero
        }
        var h = h0
        var r = fusedZero!
        for (index, layer) in layers.enumerated() {
            let mask = layerTypes[index] == "sliding_attention" ? slidingMask! : fullMask!
            let inputNorm = layer.inputLayerNorm
            let afterAttention = MapleAddRMSNormKernel.callAsFunction(
                h, r, inputNorm.weight, eps: inputNorm.eps)
            h = afterAttention.h
            r = layer.selfAttention(afterAttention.hn, mask: mask, cache: caches[index])
            let postNorm = layer.postAttentionLayerNorm
            let afterMLPInput = MapleAddRMSNormKernel.callAsFunction(
                h, r, postNorm.weight, eps: postNorm.eps)
            h = afterMLPInput.h
            r = layer.mlp(afterMLPInput.hn)
        }
        return MapleAddRMSNormKernel.callAsFunction(h, r, norm.weight, eps: norm.eps).hn
    }
}

public final class MapleModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: MapleModelInner
    let configuration: MapleConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    /// Present when the checkpoint carries usable `flash_head` metadata.
    @ModuleInfo(key: "lm_head_flash") public var lmHeadFlash: MapleFlashHead?

    /// Permanently force the portable single-token decode path (no fused
    /// Metal kernels). Useful for diagnostics and A/B benchmarking; the
    /// `MLX_MAPLE_FUSED_KERNELS=0` environment variable has the same effect
    /// from process start.
    public func disableFusedDecodeKernels() {
        model.fusedAddNorm = false
        for layer in model.layers {
            layer.selfAttention.fusedQK = false
            (layer.mlp as? MapleSparseMoeBlock)?.gate.fusedRouter = false
        }
    }

    /// Output-head selection: ``MapleHeadMode/exact`` (default) or the
    /// approximate ``MapleHeadMode/flash`` head for single-token decode.
    /// Callers may set this on the model inside a loaded container; the exact
    /// head is always used for prefill, batches, and unsupported heads.
    public var headMode: MapleHeadMode = .exact

    public init(_ config: MapleConfiguration) {
        configuration = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        model = MapleModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
            // Instantiate FlashHead tensors whenever valid metadata exists so a
            // FlashHead-bearing checkpoint loads in one pass; it stays unused
            // until headMode is set to .flash.
            if let metadata = config.flashHead, metadata.isValid {
                _lmHeadFlash.wrappedValue = MapleFlashHead(
                    hiddenSize: config.hiddenSize, metadata: metadata)
            }
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        if headMode == .flash, let flash = lmHeadFlash,
            hidden.dim(0) == 1, hidden.dim(1) == 1,
            let head = lmHead as? QuantizedLinear, head.mode == .affine
        {
            return flash(hidden, lmHead: head)
        }
        return lmHead?(hidden) ?? model.wordEmbeddings.asLinear(hidden)
    }

    public func sanitize(weights original: [String: MLXArray]) -> [String: MLXArray] {
        var weights = original.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
                && !(configuration.tieWordEmbeddings && key.hasPrefix("lm_head."))
        }

        if lmHeadFlash == nil {
            // No usable FlashHead metadata: drop its tensors so checkpoints
            // that carry them still load.
            weights = weights.filter { !$0.key.hasPrefix("lm_head_flash.") }
        } else {
            // Folded into the centroid rows at generation time; older shards
            // still carry the tensor.
            weights.removeValue(forKey: "lm_head_flash.cluster_scale")
            // `lm_head_flash.head.*` is lm_head permuted by token_map, so it is
            // pure redundancy on disk. Checkpoints may ship it or omit it;
            // reconcile both here.
            if weights["lm_head_flash.head.weight"] == nil,
                let tokenMap = weights["lm_head_flash.token_map"]
            {
                let order = tokenMap.reshaped(-1)
                for suffix in ["weight", "scales", "biases"] {
                    if let source = weights["lm_head.\(suffix)"] {
                        weights["lm_head_flash.head.\(suffix)"] =
                            take(source, order, axis: 0)
                            .reshaped(tokenMap.dim(0), tokenMap.dim(1), -1)
                    }
                }
            }
        }

        let rowAlphaKeys = weights.keys.filter { $0.hasSuffix(".row_alpha") }
        let groupSize = configuration.quantization?.groupSize ?? 128
        for key in rowAlphaKeys {
            let prefix = String(key.dropLast(".row_alpha".count))
            guard let alpha = weights.removeValue(forKey: key),
                weights["\(prefix).scales"] == nil,
                weights["\(prefix).biases"] == nil,
                let packedWeight = weights["\(prefix).weight"]
            else { continue }
            let groupCount = packedWeight.dim(-1) * 16 / groupSize
            let shape = alpha.shape + [groupCount]
            let scales = contiguous(broadcast(alpha.expandedDimensions(axis: -1), to: shape))
            weights["\(prefix).scales"] = scales
            weights["\(prefix).biases"] = contiguous(-scales)
        }

        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["weight", "scales", "biases", "bias"] {
                    let first = "\(prefix).mlp.experts.0.\(projection).\(suffix)"
                    guard weights[first] != nil else { continue }
                    var expertWeights: [MLXArray] = []
                    for expert in 0 ..< configuration.numExperts {
                        let key = "\(prefix).mlp.experts.\(expert).\(projection).\(suffix)"
                        if let value = weights.removeValue(forKey: key) {
                            expertWeights.append(value)
                        }
                    }
                    if expertWeights.count == configuration.numExperts {
                        weights["\(prefix).mlp.switch_mlp.\(projection).\(suffix)"] =
                            stacked(expertWeights)
                    }
                }
            }

            for suffix in ["weight", "scales", "biases", "bias"] {
                let qkvKeys = ["q_proj", "k_proj", "v_proj"].map {
                    "\(prefix).self_attn.\($0).\(suffix)"
                }
                if qkvKeys.allSatisfy({ weights[$0] != nil }) {
                    weights["\(prefix).self_attn.qkv_proj.\(suffix)"] = concatenated(
                        qkvKeys.map { weights.removeValue(forKey: $0)! }, axis: 0)
                }

                let up = "\(prefix).mlp.switch_mlp.up_proj.\(suffix)"
                let gate = "\(prefix).mlp.switch_mlp.gate_proj.\(suffix)"
                if let upValue = weights.removeValue(forKey: up) {
                    guard let gateValue = weights.removeValue(forKey: gate) else {
                        weights[up] = upValue
                        continue
                    }
                    weights["\(prefix).mlp.switch_mlp.up_gate_proj.\(suffix)"] = concatenated(
                        [upValue, gateValue], axis: 1)
                }
            }
        }
        return weights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map {
            $0 == "sliding_attention"
                ? RotatingKVCache(maxSize: configuration.slidingWindow)
                : KVCacheSimple()
        }
    }
}

extension MapleModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
