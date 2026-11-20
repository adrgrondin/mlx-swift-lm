// Copyright © 2026 Apple Inc.
// Decoder architecture adapted from ml-explore/mlx-lm#2086 (Apache-2.0).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct Inkling2Configuration: Decodable, Sendable {
    struct Loop: Decodable, Sendable {
        let coreStart: Int
        let coreEnd: Int
        let numPasses: Int
        let gateRank: Int
        let isVarloop: Bool?

        enum CodingKeys: String, CodingKey {
            case coreStart = "core_start"
            case coreEnd = "core_end"
            case numPasses = "num_passes"
            case gateRank = "gate_rank"
            case isVarloop = "is_varloop"
        }
    }

    struct Text: Decodable, Sendable {
        let hiddenSize: Int
        let numHiddenLayers: Int
        let vocabSize: Int
        let unpaddedVocabSize: Int
        let numHeads: Int
        let kvHeads: Int
        let headDim: Int
        let slidingHeads: Int
        let slidingKVHeads: Int
        let slidingHeadDim: Int
        let slidingWindow: Int
        let localLayerIDs: [Int]
        let relativeDim: Int
        let relativeExtent: Int
        let logScalingFloor: Float
        let logScalingAlpha: Float
        let normEps: Float
        let qkClip: Float
        let relativeClip: Float
        let convSize: Int
        let blockSize: Int
        let experts: Int
        let expertsPerToken: Int
        let intermediateSize: Int
        let gateCap: Float
        let upCap: Float
        let routeScale: Float
        let logitCap: Float
        let loop: Loop

        var core: Range<Int> { loop.coreStart ..< loop.coreEnd }
        var visits: [Int] {
            Array(0 ..< core.lowerBound)
                + Array(repeating: Array(core), count: loop.numPasses).flatMap { $0 }
                + Array(core.upperBound ..< numHiddenLayers)
        }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case vocabSize = "vocab_size"
            case unpaddedVocabSize = "unpadded_vocab_size"
            case numHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case slidingHeads = "swa_num_attention_heads"
            case slidingKVHeads = "swa_num_key_value_heads"
            case slidingHeadDim = "swa_head_dim"
            case slidingWindow = "sliding_window_size"
            case localLayerIDs = "local_layer_ids"
            case relativeDim = "d_rel"
            case relativeExtent = "rel_extent"
            case logScalingFloor = "log_scaling_n_floor"
            case logScalingAlpha = "log_scaling_alpha"
            case normEps = "rms_norm_eps"
            case qkClip = "qk_l2_clip_threshold"
            case relativeClip = "rel_l2_clip_threshold"
            case convSize = "sconv_kernel_size"
            case blockSize = "n_layers_per_attn_res_block"
            case experts = "n_routed_experts"
            case expertsPerToken = "num_experts_per_tok"
            case intermediateSize = "intermediate_size"
            case gateCap = "mlp_gate_softcap"
            case upCap = "mlp_up_softcap"
            case routeScale = "route_scale"
            case logitCap = "final_logit_softcapping"
            case loop = "loop_config"
        }
    }

    let text: Text
    let vision: Inkling2VisionConfiguration
    let imageTokenID: Int?
    var imageToken: Int { imageTokenID ?? 200054 }

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case vision = "vision_config"
        case imageTokenID = "image_token_id"
    }
}

public final class Inkling2: Module, LanguageModel, KVCacheDimensionProvider {
    let configuration: Inkling2Configuration
    @ModuleInfo(key: "model") var backbone: Inkling2Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ configuration: Inkling2Configuration) throws {
        let text = configuration.text
        let loop = text.loop
        guard loop.isVarloop != true, text.blockSize > 0,
            loop.coreStart > 0, loop.coreEnd > loop.coreStart,
            loop.coreEnd <= text.numHiddenLayers, loop.numPasses > 0, loop.gateRank > 0,
            loop.coreStart.isMultiple(of: text.blockSize),
            loop.coreEnd.isMultiple(of: text.blockSize),
            text.kvHeads > 0, text.slidingKVHeads > 0,
            text.numHeads.isMultiple(of: text.kvHeads),
            text.slidingHeads.isMultiple(of: text.slidingKVHeads),
            text.expertsPerToken > 0, text.expertsPerToken <= text.experts,
            text.slidingWindow > 0, text.convSize > 1,
            text.relativeExtent > 0, text.unpaddedVocabSize <= text.vocabSize,
            configuration.vision.isValid
        else {
            throw ModelFactoryError.invalidConfiguration("Unsupported Inkling 2 Mini configuration")
        }
        self.configuration = configuration
        _backbone.wrappedValue = Inkling2Backbone(configuration)
        _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabSize, bias: false)
    }

    public var vocabularySize: Int { configuration.text.unpaddedVocabSize }
    public var reasoningConfig: ReasoningConfig? { .inkling2 }
    public var toolCallFormat: ToolCallFormat? { .inkling2 }

    public var kvHeads: [Int] {
        let text = configuration.text
        return text.visits.map {
            text.localLayerIDs.contains($0) ? text.slidingKVHeads : text.kvHeads
        }
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let text = configuration.text
        return try text.visits.map { layer in
            let kv = try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: text.slidingWindow,
                usesSlidingWindow: text.localLayerIDs.contains(layer))
            return CacheList(kv, ArraysCache(size: 4))
        }
    }

    private func logits(_ embeddings: MLXArray, cache: [KVCache]?, lastOnly: Bool = false)
        -> MLXArray
    {
        let states = backbone.languageModel(embeddings, cache: cache)
        let values = lmHead(lastOnly ? states[0..., (-1)..., 0...] : states).asType(.float32)
        let cap = configuration.text.logitCap
        return (tanh(values / cap) * cap)[.ellipsis, ..<vocabularySize]
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(backbone.languageModel.embed(inputs), cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.video == nil, input.audio == nil else {
            throw VLMError.processing("Inkling 2 Mini accepts text and still images only")
        }
        let tokens =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        guard tokens.dim(0) == 1, tokens.dim(1) > 0 else {
            throw VLMError.processing("Inkling 2 Mini requires one nonempty prompt")
        }
        let embeddings = backbone.languageModel.embed(tokens)
        let positions = tokens.asArray(Int.self).enumerated().compactMap {
            $0.element == configuration.imageToken ? $0.offset : nil
        }
        if let image = input.image, let frames = image.frames, !frames.isEmpty {
            var cursor = 0
            var features = [MLXArray]()
            let block = configuration.vision.blockSize
            for frame in frames {
                guard frame.t == 1, frame.h > 0, frame.w > 0,
                    frame.h.isMultiple(of: block), frame.w.isMultiple(of: block)
                else { throw VLMError.processing("Invalid Inkling 2 image grid") }
                let size = 3 * frame.h * frame.w
                guard cursor + size <= image.pixels.size else {
                    throw VLMError.processing("Invalid Inkling 2 pixel count")
                }
                let pixels = image.pixels[cursor ..< (cursor + size)].reshaped(frame.h, frame.w, 3)
                features.append(backbone.visionTower(pixels))
                cursor += size
            }
            let rows = concatenated(features, axis: 0)
            guard cursor == image.pixels.size, positions.count == rows.dim(0) else {
                throw VLMError.processing("Inkling 2 image features and placeholder counts differ")
            }
            embeddings[0, MLXArray(positions), 0...] = rows.asType(embeddings.dtype)
        } else if input.image != nil || !positions.isEmpty {
            throw VLMError.processing("Inkling 2 image placeholders have no image grid")
        }

        // Relative bias and per-query sinks make the prefill attention mask dense.
        let total = tokens.dim(1)
        prefill.prefix?((cache.first as? CacheList)?[0].offset ?? 0)
        let start = try prefill.forEachChunk(
            total: total, defaultStepSize: 256, maximumStepSize: 256
        ) { range in
            eval(backbone.languageModel(embeddings[0..., range, 0...], cache: cache))
        }
        let output = logits(embeddings[0..., start..., 0...], cache: cache, lastOnly: true)
        prefill.progress?(total, total)
        return .logits(.init(logits: output))
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        for (index, layer) in backbone.languageModel.layers.enumerated() {
            try layer.mlp.experts.prepare(
                checkpoint.weights, prefix: "model.language_model.layers.\(index).mlp.experts")
        }
        return checkpoint
    }
}

final class Inkling2Backbone: Module {
    @ModuleInfo(key: "language_model") var languageModel: Inkling2TextModel
    @ModuleInfo(key: "vision_tower") var visionTower: Inkling2Vision

    init(_ configuration: Inkling2Configuration) {
        _languageModel.wrappedValue = Inkling2TextModel(configuration.text)
        _visionTower.wrappedValue = Inkling2Vision(
            configuration.vision, outputSize: configuration.text.hiddenSize)
    }
}

final class Inkling2TextModel: Module {
    let configuration: Inkling2Configuration.Text
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    @ModuleInfo var layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2DepthAttention
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput
    @ModuleInfo var norm: RMSNorm

    init(_ c: Inkling2Configuration.Text) {
        configuration = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _embedNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { Inkling2DecoderLayer(c, index: $0) }
        _finalInput.wrappedValue = Inkling2DepthAttention(dimensions: c.hiddenSize)
        _messageInput.wrappedValue = Inkling2MessageInput(
            dimensions: c.hiddenSize, layers: c.core.count)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
    }

    func embed(_ tokens: MLXArray) -> MLXArray { embedNorm(embedTokens(tokens)) }

    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        let core = configuration.core
        var visit = 0
        var hidden = embeddings
        var blocks = [MLXArray]()

        func run(_ indices: Range<Int>, bank: [MLXArray]? = nil) {
            for index in indices {
                let messages = bank.map { messageInput($0, layer: index - core.lowerBound) }
                let layerCache = cache.map { $0[visit] as! CacheList }
                hidden = layers[index](
                    hidden, blocks: &blocks, cache: layerCache, messages: messages)
                visit += 1
            }
        }

        run(0 ..< core.lowerBound)
        let prelude = (hidden, blocks)
        run(core)
        for _ in 1 ..< configuration.loop.numPasses {
            // Skip the repeated prelude and the first core block's prelude state.
            let bank = Array(blocks.dropFirst(prelude.1.count + 1)) + [inkling2RMS(hidden)]
            (hidden, blocks) = prelude
            run(core, bank: bank)
        }
        run(core.upperBound ..< layers.count)
        return norm(finalInput(blocks, inkling2RMS(hidden))[.ellipsis, 0, 0...]).asType(
            embeddings.dtype)
    }
}
