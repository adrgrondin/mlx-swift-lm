// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func getLlama4AttentionScale(
    offset: Int, count: Int, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    let positions = MLXArray(stride(from: 0, to: count, by: 1)).asType(.float32)
    let offsetArray = MLXArray(repeating: Float(offset), shape: positions.shape)
    let values = positions + offsetArray
    let scaling = 1 + beta
        * MLX.log(1 + MLX.floor(values / Float(maxPositionEmbeddings)))
    return scaling[0..., .newAxis]
}

private func stringOrNumberToFloat(_ value: StringOrNumber?, default defaultValue: Float)
    -> Float
{
    if case .float(let number) = value {
        return number
    }
    if case .string(let stringValue) = value, let parsed = Float(stringValue) {
        return parsed
    }
    return defaultValue
}

private class Llama3RoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    let freqs: MLXArray

    init(dims: Int, maxPositionEmbeddings: Int, traditional: Bool, base: Float,
         scalingConfig: [String: StringOrNumber])
    {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional

        let factor = stringOrNumberToFloat(scalingConfig["factor"], default: 1.0)
        let lowFreqFactor = stringOrNumberToFloat(scalingConfig["low_freq_factor"], default: 1.0)
        let highFreqFactor = stringOrNumberToFloat(scalingConfig["high_freq_factor"], default: 4.0)
        let oldContextLen = stringOrNumberToFloat(
            scalingConfig["original_max_position_embeddings"], default: 8192)

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        var frequencies = MLX.pow(
            base,
            MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims)
        )
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen), frequencies * factor, frequencies)
        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen), wavelens .< MLXArray(lowFreqWavelen))
        let smoothFactors = (oldContextLen / wavelens - lowFreqFactor)
            / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }
}

private func makeRoPE(
    dimensions: Int, parameters: [String: StringOrNumber]?, maxPositionEmbeddings: Int?
) -> Module {
    let ropeType: String = {
        if let typeValue = parameters?["type"], case .string(let typeString) = typeValue {
            return typeString
        }
        if let typeValue = parameters?["rope_type"], case .string(let typeString) = typeValue {
            return typeString
        }
        return "default"
    }()

    let base = stringOrNumberToFloat(parameters?["rope_theta"], default: 10000)
    let maxPos = maxPositionEmbeddings ?? 2048

    switch ropeType {
    case "llama3":
        return Llama3RoPE(
            dims: dimensions,
            maxPositionEmbeddings: maxPos,
            traditional: false,
            base: base,
            scalingConfig: parameters ?? [:]
        )
    case "yarn":
        let scalingFactor = stringOrNumberToFloat(parameters?["factor"], default: 1.0)
        let originalMax = Int(stringOrNumberToFloat(
            parameters?["original_max_position_embeddings"], default: 4096))
        let betaFast = stringOrNumberToFloat(parameters?["beta_fast"], default: 32)
        let betaSlow = stringOrNumberToFloat(parameters?["beta_slow"], default: 1)
        let mscale = stringOrNumberToFloat(parameters?["mscale"], default: 1)
        let mscaleAllDim = stringOrNumberToFloat(parameters?["mscale_all_dim"], default: 0)
        return YarnRoPE(
            dimensions: dimensions,
            traditional: false,
            maxPositionEmbeddings: maxPos,
            base: base,
            scalingFactor: scalingFactor,
            originalMaxPositionEmbeddings: originalMax,
            betaFast: betaFast,
            betaSlow: betaSlow,
            mscale: mscale,
            mscaleAllDim: mscaleAllDim
        )
    case "linear":
        let factor = stringOrNumberToFloat(parameters?["factor"], default: 1.0)
        return RoPE(dimensions: dimensions, traditional: false, base: base, scale: 1 / factor)
    default:
        return RoPE(dimensions: dimensions, traditional: false, base: base, scale: 1.0)
    }
}

public struct Ministral3Configuration: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var headDim: Int
    public var maxPositionEmbeddings: Int?
    public var kvHeads: Int
    public var ropeParameters: [String: StringOrNumber]?
    public var tieWordEmbeddings: Bool
    public var layerTypes: [String]
    public var slidingWindow: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let outerContainer = try decoder.container(keyedBy: CodingKeys.self)
        let container: KeyedDecodingContainer<CodingKeys>
        if outerContainer.contains(.textConfig) {
            container = try outerContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            container = outerContainer
        }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? false
        if let layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = layerTypes
        } else {
            self.layerTypes = Array(repeating: "full_attention", count: hiddenLayers)
        }
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
    }

    public init(
        modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
        attentionHeads: Int, rmsNormEps: Float, vocabularySize: Int,
        headDim: Int? = nil, maxPositionEmbeddings: Int? = nil, kvHeads: Int? = nil,
        ropeParameters: [String: StringOrNumber]? = nil, tieWordEmbeddings: Bool = false,
        layerTypes: [String]? = nil, slidingWindow: Int? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.headDim = headDim ?? hiddenSize / attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeParameters = ropeParameters
        self.tieWordEmbeddings = tieWordEmbeddings
        self.layerTypes = layerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
        self.slidingWindow = slidingWindow
    }
}

private class Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeParameters: [String: StringOrNumber]?
    let rope: Module

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear

    init(_ config: Ministral3Configuration) {
        nHeads = config.attentionHeads
        nKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        ropeParameters = config.ropeParameters

        _query.wrappedValue = Linear(config.hiddenSize, nHeads * headDim, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, nKVHeads * headDim, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, nKVHeads * headDim, bias: false)
        _outputProjection.wrappedValue = Linear(
            nHeads * headDim, config.hiddenSize, bias: false)

        rope = makeRoPE(
            dimensions: headDim,
            parameters: config.ropeParameters,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = query(x)
        var keys = key(x)
        var values = value(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let scaleArray = attnScale[.newAxis, .newAxis, 0..., .newAxis]
        queries = queries * scaleArray

        let attnOutput = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProjection(attnOutput)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: Ministral3Configuration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let useSliding: Bool

    init(_ config: Ministral3Configuration, useSliding: Bool) {
        self.useSliding = useSliding
        _attention.wrappedValue = Attention(config)
        _mlp.wrappedValue = MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attnOut = attention(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        let h = x + attnOut
        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut
    }
}

private class Ministral3LanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm
    let layerTypes: [String]
    let slidingWindow: Int?
    let config: Ministral3Configuration

    init(_ config: Ministral3Configuration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layerTypes = config.layerTypes
        slidingWindow = config.slidingWindow
        layers = layerTypes.map { TransformerBlock(config, useSliding: $0 == "sliding_attention") }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray?
    ) -> MLXArray {
        var h = inputEmbeddings ?? embedTokens(inputs)

        let offset = cache?.first?.offset ?? 0
        let seqLength = inputs.dim(1)
        let beta = stringOrNumberToFloat(
            ropeParameters?["llama_4_scaling_beta"], default: 0.0)
        let maxPos = Int(stringOrNumberToFloat(
            ropeParameters?["original_max_position_embeddings"],
            default: Float(config.maxPositionEmbeddings ?? 2048)))
        var attnScale = getLlama4AttentionScale(
            offset: offset,
            count: seqLength,
            beta: beta,
            maxPositionEmbeddings: maxPos
        )
        attnScale = attnScale.asType(h.dtype)

        let faMask = createAttentionMask(h: h, cache: cache)
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            if let slidingWindow, let cache, let slidingIndex =
                layerTypes.firstIndex(of: "sliding_attention")
            {
                return .array(createCausalMask(
                    n: seqLength,
                    offset: cache[slidingIndex].offset,
                    windowSize: slidingWindow
                ))
            }
            return .none
        }()

        for (idx, layer) in layers.enumerated() {
            let mask = layer.useSliding ? slidingMask : faMask
            h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[idx])
        }

        return norm(h)
    }

    private var ropeParameters: [String: StringOrNumber]? { config.ropeParameters }
}

public class Ministral3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: Ministral3LanguageModel
    let configuration: Ministral3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Ministral3Configuration) {
        configuration = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)

        model = Ministral3LanguageModel(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let output = callAsFunction(inputs, cache: cache, inputEmbeddings: nil)
        return .init(logits: output)
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        if let slidingWindow = configuration.slidingWindow {
            return configuration.layerTypes.map { layerType in
                if layerType == "sliding_attention" {
                    return RotatingKVCache(maxSize: slidingWindow, keep: 0)
                }
                return KVCacheSimple()
            }
        }
        return (0 ..< configuration.hiddenLayers).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var filtered = weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
        if configuration.tieWordEmbeddings {
            filtered.removeValue(forKey: "lm_head.weight")
        }

        var newWeights: [String: MLXArray] = [:]
        for (key, value) in filtered {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = filtered[weightKey] {
                    newWeights[weightKey] = weight * value
                }
            } else if key.contains("activation_scale") {
                continue
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        return newWeights
    }
}
