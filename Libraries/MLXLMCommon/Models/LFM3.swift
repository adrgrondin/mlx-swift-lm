// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Configuration shared by the standalone and multimodal LFM3 decoders.
public struct LFM3Configuration: Codable, Sendable, ModelConfigurationValidating {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDimensions: Int
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let normEps: Float
    public let convBias: Bool
    public let convLCache: Int
    public let layerTypes: [String]
    public let slidingWindow: Int
    public let useSandwichNorm: Bool
    public let tieWordEmbeddings: Bool
    public let ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 7680
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDimensions =
            try c.decodeIfPresent(Int.self, forKey: .headDimensions)
            ?? hiddenSize / max(attentionHeads, 1)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128000
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 128000
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 7
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
        useSandwichNorm = try c.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? true
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< max(hiddenLayers, 0)).map {
                $0 % 4 == 2
                    ? "full_attention" : ($0 > 0 && $0 % 4 == 0 ? "sliding_attention" : "conv")
            }
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            ?? [
                "full_attention": [
                    "rope_type": .string("default"), "rope_theta": .float(1e6),
                    "partial_rotary_factor": .float(0.5),
                ],
                "sliding_attention": ["rope_type": .string("default"), "rope_theta": .float(1e4)],
            ]
    }

    public func validateModelConfiguration() throws {
        func require(_ condition: Bool, _ message: String) throws {
            guard condition else {
                throw ModelFactoryError.invalidConfiguration("LFM3: \(message)")
            }
        }
        try require(
            hiddenSize > 0 && intermediateSize > 0 && vocabularySize > 0,
            "dimensions must be positive")
        try require(
            hiddenLayers > 0 && layerTypes.count == hiddenLayers,
            "layer_types must match num_hidden_layers")
        try require(
            attentionHeads > 0 && kvHeads > 0 && attentionHeads % kvHeads == 0,
            "invalid attention heads")
        try require(
            headDimensions > 0 && headDimensions % 2 == 0, "head_dim must be positive and even")
        try require(
            convLCache > 1 && slidingWindow > 0 && normEps > 0,
            "invalid convolution, window, or norm")
        for type in layerTypes {
            try require(
                ["conv", "full_attention", "sliding_attention"].contains(type),
                "unknown layer type \(type)")
            if type != "conv" {
                guard let rope = ropeParameters[type] else {
                    throw ModelFactoryError.invalidConfiguration("LFM3: missing RoPE for \(type)")
                }
                try require(
                    rope["rope_type"] == nil || rope["rope_type"] == .string("default"),
                    "unsupported RoPE type")
                let base = rope["rope_theta"]?.asFloat() ?? 1e6
                let partial = rope["partial_rotary_factor"]?.asFloat() ?? 1
                try require(
                    base.isFinite && base > 0 && partial.isFinite && partial > 0 && partial <= 1,
                    "invalid RoPE parameters")
            }
        }
    }
}

// The zero-frequency tail preserves the original half-head pairing and frequency denominator.
package final class LFM3RotaryEmbedding: Module, OffsetLayer, ArrayOffsetLayer {
    private let _inverseFrequencies: MLXArray

    package init(dimensions: Int, base: Float, partial: Float) {
        let angles = Int(partial * Float(dimensions)) / 2
        let frequencies =
            1
            / pow(
                base,
                MLXArray(stride(from: 0, to: angles * 2, by: 2)).asType(.float32)
                    / Float(dimensions))
        _inverseFrequencies = concatenated([frequencies, MLXArray.zeros([dimensions / 2 - angles])])
    }

    package func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        callAsFunction(x, offset: MLXArray(Int32(offset)))
    }

    package func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        let positions = MLXArray(0 ..< x.dim(-2)) + offset.reshaped(-1, 1)
        let angles = positions.asType(.float32).expandedDimensions(axis: -1) * _inverseFrequencies
        let frequencies = concatenated([angles, angles], axis: -1).expandedDimensions(axis: 1)
        let cosine = cos(frequencies).asType(x.dtype).asType(.float32)
        let sine = sin(frequencies).asType(x.dtype).asType(.float32)
        let halves = split(x.asType(.float32), parts: 2, axis: -1)
        return
            (x.asType(.float32) * cosine + concatenated([-halves[1], halves[0]], axis: -1) * sine)
            .asType(x.dtype)
    }
}

package final class LFM3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: LFM3RotaryEmbedding
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int?

    init(_ c: LFM3Configuration, type: String) {
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        headDim = c.headDimensions
        window = type == "sliding_attention" ? c.slidingWindow : nil
        _qProj.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _gateProj.wrappedValue = Linear(c.hiddenSize, heads, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.normEps)
        let p = c.ropeParameters[type] ?? [:]
        rope = LFM3RotaryEmbedding(
            dimensions: headDim, base: p["rope_theta"]?.asFloat() ?? 1e6,
            partial: p["partial_rotary_factor"]?.asFloat() ?? 1)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        let offset = cache?.ropeOffset
        let q = applyRotaryPosition(
            rope, to: qNorm(qProj(x).reshaped(b, n, heads, headDim)).transposed(0, 2, 1, 3),
            offset: offset)
        let k = applyRotaryPosition(
            rope, to: kNorm(kProj(x).reshaped(b, n, kvHeads, headDim)).transposed(0, 2, 1, 3),
            offset: offset)
        let v = vProj(x).reshaped(b, n, kvHeads, headDim).transposed(0, 2, 1, 3)
        let mask = createAttentionMask(h: x, cache: cache, windowSize: window)
        let attended = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache,
            scale: pow(Float(headDim), -0.5), mask: mask
        ).transposed(0, 2, 1, 3)
        return oProj(
            (attended * sigmoid(gateProj(x)).expandedDimensions(axis: -1)).reshaped(b, n, -1))
    }
}

package final class LFM3ShortConv: Module {
    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let width: Int
    let hiddenSize: Int

    init(_ c: LFM3Configuration) {
        width = c.convLCache
        hiddenSize = c.hiddenSize
        _conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize, outputChannels: hiddenSize,
            kernelSize: width, groups: hiddenSize, bias: c.convBias)
        _inProj.wrappedValue = Linear(hiddenSize, 3 * hiddenSize, bias: c.convBias)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: c.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let n = x.dim(1)
        let mask = cache?.makeMask(N: n)?.expandedDimensions(axis: -1)
        let input = mask.map { MLX.where($0, x, zeros(like: x)) } ?? x
        let parts = inProj(input).split(parts: 3, axis: -1)
        var gated = parts[0] * parts[2]
        if let mask { gated = MLX.where(mask, gated, zeros(like: gated)) }
        let previous =
            cache?[0] ?? MLXArray.zeros([x.dim(0), width - 1, hiddenSize], dtype: x.dtype)
        let joined = concatenated([previous, gated], axis: 1)
        if let cache {
            if let lengths = cache.currentLengths {
                let ends = clip(lengths, min: 0, max: n).asType(.int32)
                let indices = (ends[0..., .newAxis] + MLXArray(0 ..< (width - 1)))[
                    0..., 0..., .newAxis]
                cache[0] = contiguous(takeAlong(joined, indices, axis: 1))
            } else {
                cache[0] = contiguous(joined[0..., (joined.dim(1) - width + 1)..., 0...])
            }
            cache.advance(n)
        }
        return outProj(parts[1] * conv(joined))
    }
}

package final class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    package init(hiddenSize: Int, intermediateSize: Int) {
        _w1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _w2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _w3.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

package final class LFM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ c: LFM3Configuration, type: String) {
        if type == "conv" {
            _conv.wrappedValue = LFM3ShortConv(c)
        } else {
            _attention.wrappedValue = LFM3Attention(c, type: type)
        }
        _feedForward.wrappedValue = LFM3MLP(
            hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        if c.useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        }
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let normalized = operatorNorm(x)
        let operated =
            attention?(normalized, cache: cache) ?? conv!(normalized, cache: cache as? MambaCache)
        let h = x + (postOperatorNorm?(operated) ?? operated)
        let ff = feedForward(ffnNorm(h))
        return h + (postFFNNorm?(ff) ?? ff)
    }
}

package final class LFM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") package var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    let layers: [LFM3DecoderLayer]

    init(_ c: LFM3Configuration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        layers = c.layerTypes.map { LFM3DecoderLayer(c, type: $0) }
    }

    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        for (i, layer) in layers.enumerated() { h = layer(h, cache: cache?[i]) }
        return embeddingNorm(h)
    }
}

/// Shared LFM3 causal decoder, also used inside LFM3-VL.
open class LFM3LanguageModel: Module, LanguageModel, LoRAModel, KVCacheDimensionProvider {
    public let configuration: LFM3Configuration
    package let model: LFM3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    public var vocabularySize: Int { configuration.vocabularySize }
    public var kvHeads: [Int] {
        configuration.layerTypes.map { $0 == "conv" ? 0 : configuration.kvHeads }
    }
    public var loraLayers: [Module] { model.layers }
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }

    public init(_ configuration: LFM3Configuration) {
        self.configuration = configuration
        model = LFM3ModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
    }

    package func embed(_ tokens: MLXArray) -> MLXArray { model.embedTokens(tokens) }

    package func logits(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(embeddings, cache: cache)
        return lmHead?(h) ?? model.embedTokens.asLinear(h)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        return logits(embeddings: embed(tokens), cache: cache)
    }

    package func prefill(
        embeddings: MLXArray, lengths: [Int]?, cache: [KVCache], prefill: PrefillParameters
    ) throws -> LMOutput {
        try withPreparedCache(cache, lengths: lengths) {
            let total = embeddings.dim(1)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = model(embeddings[0..., range, 0...], cache: cache.isEmpty ? nil : cache)
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
            let output = logits(
                embeddings: embeddings[0..., processed..., 0...], cache: cache.isEmpty ? nil : cache
            )
            prefill.progress?(total, total)
            return LMOutput(logits: output)
        }
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        return .logits(
            try self.prefill(
                embeddings: embed(tokens), lengths: input.text.sequenceLengths, cache: cache,
                prefill: prefill))
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.layerTypes.map { type in
            if type == "conv" { return MambaCache() }
            return try makeHybridAttentionKVCache(
                parameters: parameters, slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: type == "sliding_attention")
        }
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = try checkpoint.mapNames(
            using: .init([
                .replacePrefix("language_model", with: "")
            ]))
        checkpoint = try prepareLFM3Checkpoint(
            checkpoint, tiedEmbeddings: configuration.tieWordEmbeddings)
        return checkpoint
    }
}

package func prepareLFM3Checkpoint(_ checkpoint: ModelCheckpoint, tiedEmbeddings: Bool) throws
    -> ModelCheckpoint
{
    var checkpoint = try checkpoint.mapNames(
        using: .init(tiedEmbeddings ? [.excludeModule("lm_head")] : []))
    for (name, value) in checkpoint.weights where name.hasSuffix("conv.conv.weight") {
        if value.ndim == 3 && value.dim(1) == 1 && value.dim(2) > 1 {
            checkpoint.weights[name] = value.transposed(0, 2, 1)
        }
    }
    return checkpoint
}

/// LFM3 chat history, including reasoning supplied by the generation prompt.
public struct LFM3MessageGenerator: MessageGenerator {
    private let multimodal: Bool
    public init(multimodal: Bool = false) { self.multimodal = multimodal }

    public func generate(message: Chat.Message) -> Message {
        var message = message
        var reasoning: String?
        if message.role == .assistant {
            let start = "<think>"
            let content: Substring?
            if message.prefilledReasoningStartDelimiter == start {
                content = message.content[...]
            } else if message.content.hasPrefix(start) {
                content = message.content.dropFirst(start.count)
            } else {
                content = nil
            }
            if let content {
                if let end = content.range(of: "</think>") {
                    reasoning = String(content[..<end.lowerBound])
                    message.content = String(content[end.upperBound...])
                } else {
                    reasoning = String(content)
                    message.content = ""
                }
            }
        }
        var result: Message = ["role": message.role.rawValue]
        if multimodal {
            result["content"] = contentParts(for: message, layout: .imagesThenVideosThenText)
        } else {
            result["content"] = message.content
        }
        if let reasoning { result["reasoning_content"] = reasoning }
        addToolMetadata(to: &result, for: message)
        return result
    }
}
