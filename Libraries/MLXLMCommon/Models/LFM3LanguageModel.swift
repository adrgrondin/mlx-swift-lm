// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// The shared causal language head over the LFM3 backbone, used by text and vision models.
open class LFM3LanguageModel: Module, LoRAModel, KVCacheDimensionProvider {
    public let model: LFM3ModelInner
    public let configuration: LFM3Configuration
    @ModuleInfo(key: "lm_head") package var lmHead: Linear?

    public var vocabularySize: Int { configuration.vocabularySize }
    public var kvHeads: [Int] {
        configuration.layerTypes.map { $0 == .conv ? 0 : configuration.kvHeads }
    }
    public var loraLayers: [Module] { model.layers }
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }

    public init(_ configuration: LFM3Configuration) {
        self.configuration = configuration
        self.model = LFM3ModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
    }

    package func embed(_ tokens: MLXArray) -> MLXArray { model.embed(tokens) }

    package func logits(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(model(embeddings: embeddings, cache: cache))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        return projectLogits(model(tokens, cache: cache))
    }

    public func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let hidden = hiddenStates(input.tokens, cache: cache)
        let projection: Module = lmHead ?? model.embedTokens
        return .init(
            logits: projectLogits(
                quantizedVocabularyProjectionInput(hidden, projection: projection)))
    }

    package func prefill(
        embeddings: MLXArray, lengths: [Int]?, cache: [KVCache], prefill: PrefillParameters
    ) throws -> LMOutput {
        try withPreparedCache(cache, lengths: lengths) {
            let total = embeddings.dim(1)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = model(
                    embeddings: embeddings[0..., range, 0...], cache: cache.isEmpty ? nil : cache)
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

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.newCache(parameters: parameters)
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let checkpoint = try checkpoint.mapNames { name in
            var name = name
            if name.hasPrefix("language_model.") {
                name.removeFirst("language_model.".count)
            }
            let root = name.prefix { $0 != "." }
            return root == "model" || root == "lm_head" ? name : nil
        }
        return try prepareLFM3Checkpoint(
            checkpoint, tiedEmbeddings: configuration.tieWordEmbeddings)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }
}

extension LFM3LanguageModel {
    package func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache)
    }

    package func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        model.projectLogits(hiddenStates, lmHead: lmHead)
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
