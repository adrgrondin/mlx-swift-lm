// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mistral3/language.py

// MARK: - Llama 4 Attention Scaling

/// Computes the Llama 4 attention scale based on position
private func getLlama4AttentionScale(
    start: Int,
    stop: Int,
    beta: Float,
    maxPositionEmbeddings: Int
) -> MLXArray {
    let positions = MLXArray(start ..< stop).asType(.float32)
    let scaling = 1 + beta * MLX.log(1 + MLX.floor(positions / Float(maxPositionEmbeddings)))
    return expandedDimensions(scaling, axis: -1)
}

// MARK: - Attention

private class Attention: Module {

    let config: Mistral3Configuration
    let scale: Float
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ config: Mistral3Configuration) {
        self.config = config

        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads

        self.headDim = config.headDimensions ?? (config.hiddenSize / nHeads)
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        // Initialize RoPE with rope_theta from rope_parameters
        let ropeTheta = config.ropeParameters?["rope_theta"]?.asFloat() ?? config.ropeTheta
        self.rope = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            base: ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Apply Llama 4 attention scaling
        queries = queries * attentionScale

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MLP

private class MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: Mistral3Configuration) {
        let dim = config.hiddenSize
        let hiddenDim = config.intermediateSize

        self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - TransformerBlock

private class TransformerBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let useSliding: Bool

    init(_ config: Mistral3Configuration, useSliding: Bool = false) {
        self.useSliding = useSliding
        self._attention.wrappedValue = Attention(config)
        self.mlp = MLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), attentionScale: attentionScale, mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Ministral3ModelInner

private class Ministral3ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TransformerBlock]
    let norm: RMSNorm
    let config: Mistral3Configuration
    let layerTypes: [String]
    let slidingWindow: Int?
    let faIndex: Int  // Index of first full attention layer
    let swaIndex: Int?  // Index of first sliding window attention layer

    init(_ config: Mistral3Configuration) {
        precondition(config.vocabularySize > 0)
        self.config = config
        self.slidingWindow = config.slidingWindow

        // Determine layer types
        self.layerTypes = config.layerTypes ?? Array(repeating: "full_attention", count: config.hiddenLayers)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = layerTypes.map { layerType in
            TransformerBlock(config, useSliding: layerType == "sliding_attention")
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Find indices for mask creation
        self.faIndex = layerTypes.firstIndex(of: "full_attention") ?? 0
        self.swaIndex = layers.firstIndex { $0.useSliding }
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputsEmbeds {
            h = inputsEmbeds
        } else {
            h = embedTokens(inputs)
        }

        let cache = cache ?? [KVCache?](repeating: nil, count: layers.count).compactMap { $0 }
        let offset = cache.first?.offset ?? 0

        // Create full attention mask
        let faMask = createAttentionMask(h: h, cache: [cache[faIndex]])

        // Create sliding window attention mask if needed
        var swaMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if let swaIndex, let slidingWindow {
            let t = h.dim(1)
            if t > 1 {
                let swaOffset = min(slidingWindow, cache[swaIndex].offset)
                swaMask = .array(
                    createCausalMask(n: t, offset: swaOffset, windowSize: slidingWindow))
            }
        }

        // Compute Llama 4 attention scale
        let beta = config.ropeParameters?["llama_4_scaling_beta"]?.asFloat() ?? 0.0
        let originalMaxPos = config.ropeParameters?["original_max_position_embeddings"]?.asInt()
            ?? config.maxPositionEmbeddings ?? 4096
        let attentionScale = getLlama4AttentionScale(
            start: offset,
            stop: offset + inputs.dim(1),
            beta: beta,
            maxPositionEmbeddings: originalMaxPos
        ).asType(h.dtype)

        for (i, layer) in layers.enumerated() {
            let mask = layer.useSliding ? swaMask : faMask
            h = layer(h, attentionScale: attentionScale, mask: mask, cache: cache[i])
        }

        return norm(h)
    }
}

// MARK: - Mistral3Model

public class Mistral3Model: Module, LLMModel {

    public let vocabularySize: Int

    private let model: Ministral3ModelInner
    private let config: Mistral3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Mistral3Configuration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.model = Ministral3ModelInner(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if config.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else if let lmHead {
            out = lmHead(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        // Remove unused precomputed rotary freqs
        weights = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        // Remove lm_head if using tied embeddings
        if config.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Handle weight_scale_inv and activation_scale patterns (for quantized models)
        var newWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let scaleInv = value
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    newWeights[weightKey] = weight * scaleInv
                }
            } else if key.contains("activation_scale") {
                continue
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        return newWeights
    }

    /// Creates the cache for this model with appropriate cache types per layer
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let layerTypes = config.layerTypes ?? Array(repeating: "full_attention", count: config.hiddenLayers)

        return layerTypes.map { layerType in
            if layerType == "sliding_attention", let slidingWindow = config.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow)
            } else if let maxKVSize = parameters?.maxKVSize {
                return RotatingKVCache(maxSize: maxKVSize, keep: 4)
            } else {
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - Configuration

public struct Mistral3Configuration: Codable, Sendable {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeParameters: [String: StringOrNumber]?
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = false
    var layerTypes: [String]?
    var slidingWindow: Int?
    var useQKNorm: Bool = false

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useQKNorm = "use_qk_norm"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeParameters = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        useQKNorm = try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false

        // Default layer types if not specified
        if layerTypes == nil {
            layerTypes = Array(repeating: "full_attention", count: hiddenLayers)
        }
    }
}

// MARK: - LoRA Support

extension Mistral3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}


