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
    public var flashHead: [String: JSONValue]?
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
        flashHead = try c.decodeIfPresent([String: JSONValue].self, forKey: .flashHead)
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

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MapleRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: MapleRMSNorm?

    let rope: RoPELayer

    init(_ config: MapleConfiguration, layerType: String) {
        attentionHeads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        useRoPE = layerType == "sliding_attention"

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

        let rotaryDimensions = max(2, Int(Float(config.headDim) * config.partialRotaryFactor))
        rope = initializeRope(
            dims: rotaryDimensions, base: config.ropeTheta, traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let qSize = attentionHeads * headDim
        let kSize = kvHeads * headDim
        let parts = split(qkvProj(x), indices: [qSize, qSize + kSize], axis: -1)

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

    init(_ config: MapleConfiguration) {
        topK = config.numExpertsPerToken
        _weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let logits = x.asType(.float32).matmul(weight.asType(.float32).T)
        let probabilities = softmax(logits, axis: -1, precise: true)
        let indices = stopGradient(
            argPartition(probabilities, kth: -topK, axis: -1)[.ellipsis, (-topK)...])
        var scores = takeAlong(probabilities, indices, axis: -1)
        scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        return (indices, scores)
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

        for (index, layer) in layers.enumerated() {
            let mask = layerTypes[index] == "sliding_attention" ? slidingMask! : fullMask!
            hidden = layer(hidden, mask: mask, cache: caches[index])
        }
        return norm(hidden)
    }
}

public final class MapleModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: MapleModelInner
    let configuration: MapleConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: MapleConfiguration) {
        configuration = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        model = MapleModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead?(hidden) ?? model.wordEmbeddings.asLinear(hidden)
    }

    public func sanitize(weights original: [String: MLXArray]) -> [String: MLXArray] {
        var weights = original.filter { key, _ in
            !key.hasPrefix("lm_head_flash.")
                && !key.contains("rotary_emb.inv_freq")
                && !(configuration.tieWordEmbeddings && key.hasPrefix("lm_head."))
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
