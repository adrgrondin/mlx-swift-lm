// Copyright © 2024 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mistral3/mistral3.py

// MARK: - Vision Configuration

public struct Mistral3VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let patchSize: Int
    public let imageSize: Int
    
    public var numChannels: Int { _numChannels ?? 3 }
    public var layerNormEps: Float { _layerNormEps ?? 1e-5 }
    
    private let _numChannels: Int?
    private let _layerNormEps: Float?
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case patchSize = "patch_size"
        case imageSize = "image_size"
        case _numChannels = "num_channels"
        case _layerNormEps = "layer_norm_eps"
    }
}

// MARK: - Text Configuration

public struct Mistral3VLMTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    
    public var headDim: Int? { _headDim }
    public var maxPositionEmbeddings: Int? { _maxPositionEmbeddings }
    public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }
    public var ropeTheta: Float { _ropeTheta ?? 10_000 }
    public var ropeParameters: [String: StringOrNumber]? { _ropeParameters }
    public var ropeTraditional: Bool { _ropeTraditional ?? false }
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }
    public var layerTypes: [String]? { _layerTypes }
    public var slidingWindow: Int? { _slidingWindow }
    
    private let _headDim: Int?
    private let _maxPositionEmbeddings: Int?
    private let _numKeyValueHeads: Int?
    private let _ropeTheta: Float?
    private let _ropeParameters: [String: StringOrNumber]?
    private let _ropeTraditional: Bool?
    private let _tieWordEmbeddings: Bool?
    private let _layerTypes: [String]?
    private let _slidingWindow: Int?
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case _headDim = "head_dim"
        case _maxPositionEmbeddings = "max_position_embeddings"
        case _numKeyValueHeads = "num_key_value_heads"
        case _ropeTheta = "rope_theta"
        case _ropeParameters = "rope_parameters"
        case _ropeTraditional = "rope_traditional"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _layerTypes = "layer_types"
        case _slidingWindow = "sliding_window"
    }
}

// MARK: - Model Configuration

public struct Mistral3VLMConfiguration: Codable, Sendable {
    public let textConfig: Mistral3VLMTextConfiguration
    public let visionConfig: Mistral3VisionConfiguration
    public let modelType: String
    
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    public var imageTokenIndex: Int { _imageTokenIndex ?? _imageTokenId ?? 10 }
    public var visionFeatureSelectStrategy: String { _visionFeatureSelectStrategy ?? "full" }
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    public var vocabSize: Int { _vocabSize ?? 32000 }
    public var spatialMergeSize: Int { _spatialMergeSize ?? 2 }
    public var multimodalProjectorBias: Bool { _multimodalProjectorBias ?? false }
    
    private let _ignoreIndex: Int?
    private let _imageTokenIndex: Int?
    private let _imageTokenId: Int?
    private let _visionFeatureSelectStrategy: String?
    private let _visionFeatureLayer: Int?
    private let _vocabSize: Int?
    private let _spatialMergeSize: Int?
    private let _multimodalProjectorBias: Bool?
    
    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenIndex = "image_token_index"
        case _imageTokenId = "image_token_id"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _visionFeatureLayer = "vision_feature_layer"
        case _vocabSize = "vocab_size"
        case _spatialMergeSize = "spatial_merge_size"
        case _multimodalProjectorBias = "multimodal_projector_bias"
    }
}

// MARK: - Vision Model

private enum Vision {
    
    static func checkArrayShape(_ arr: MLXArray) -> Bool {
        if arr.ndim != 4 { return false }
        let (o, h, w, _) = (arr.dim(0), arr.dim(1), arr.dim(2), arr.dim(3))
        return (o >= h && o >= w && h == w)
    }
    
    // MARK: Vision Attention
    
    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float
        
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        
        init(_ config: Mistral3VisionConfiguration) {
            self.numHeads = config.numAttentionHeads
            let headDim = config.hiddenSize / config.numAttentionHeads
            self.scale = pow(Float(headDim), -0.5)
            
            self._qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            self._kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            self._vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            self._oProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        }
        
        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none) -> MLXArray {
            let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
            
            let q = qProj(x).reshaped(B, L, numHeads, D / numHeads).transposed(0, 2, 1, 3)
            let k = kProj(x).reshaped(B, L, numHeads, D / numHeads).transposed(0, 2, 1, 3)
            let v = vProj(x).reshaped(B, L, numHeads, D / numHeads).transposed(0, 2, 1, 3)
            
            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, D)
            
            return oProj(output)
        }
    }
    
    // MARK: Vision MLP
    
    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .precise)
        
        init(_ config: Mistral3VisionConfiguration) {
            self.fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            self.fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }
        
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }
    
    // MARK: Vision Encoder Layer
    
    fileprivate class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        
        init(_ config: Mistral3VisionConfiguration) {
            self._selfAttn.wrappedValue = Attention(config)
            self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            self.mlp = MLP(config)
            self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        }
        
        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none) -> MLXArray {
            let h = x + selfAttn(layerNorm1(x), mask: mask)
            return h + mlp(layerNorm2(h))
        }
    }
    
    // MARK: Vision Encoder
    
    fileprivate class Encoder: Module {
        var layers: [EncoderLayer]
        
        init(_ config: Mistral3VisionConfiguration) {
            self.layers = (0 ..< config.numHiddenLayers).map { _ in EncoderLayer(config) }
        }
        
        func callAsFunction(
            _ x: MLXArray,
            outputHiddenStates: Bool = false,
            mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        ) -> (MLXArray, [MLXArray]?) {
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            var h = x
            
            for layer in layers {
                h = layer(h, mask: mask)
                if outputHiddenStates {
                    encoderStates?.append(h)
                }
            }
            
            return (h, encoderStates)
        }
    }
    
    // MARK: Vision Embeddings
    
    fileprivate class VisionEmbeddings: Module, UnaryLayer {
        @ModuleInfo(key: "patch_conv") var patchConv: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        let numPositions: Int
        
        init(_ config: Mistral3VisionConfiguration) {
            self._patchConv.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize),
                stride: .init(config.patchSize)
            )
            let numPatches = (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            self.numPositions = numPatches
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPatches,
                dimensions: config.hiddenSize
            )
        }
        
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var patchEmbeddings = patchConv(x)
            patchEmbeddings = patchEmbeddings.flattened(start: 1, end: 2)
            let positionIds = MLXArray(0 ..< numPositions)[.newAxis, 0...]
            let posEmbedding = positionEmbedding(positionIds)
            return patchEmbeddings + posEmbedding
        }
    }
    
    // MARK: Vision Model (Pixtral-style)
    
    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "patch_conv") var patchConv: Conv2d
        @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
        @ModuleInfo(key: "transformer") var transformer: Encoder
        let config: Mistral3VisionConfiguration
        
        init(_ config: Mistral3VisionConfiguration) {
            self.config = config
            self._patchConv.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize),
                stride: .init(config.patchSize),
                bias: false
            )
            self._lnPre.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            self._transformer.wrappedValue = Encoder(config)
        }
        
        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (MLXArray, MLXArray, [MLXArray]?) {
            // x is expected in NHWC format
            var embeddings = patchConv(x)
            embeddings = embeddings.flattened(start: 1, end: 2)
            embeddings = lnPre(embeddings)
            
            let (encoded, hiddenStates) = transformer(embeddings, outputHiddenStates: outputHiddenStates)
            
            return (encoded, embeddings, hiddenStates)
        }
        
        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()
            for (k, v) in weights {
                if k.contains("position_ids") {
                    continue
                } else if k.contains("patch_conv.weight") || k.contains("patch_embedding.weight") {
                    if Vision.checkArrayShape(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }
            return sanitizedWeights
        }
    }
}

// MARK: - Unfold (im2col)

/// Extract sliding local blocks from a batched input tensor.
/// Equivalent to PyTorch's nn.functional.unfold / im2col operation.
private func unfold(
    _ input: MLXArray,
    kernelSize: Int,
    dilation: Int = 1,
    padding: Int = 0,
    stride: Int = 1
) -> MLXArray {
    var x = input
    let (batchSize, channels, height, width) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    
    // Add padding if needed
    if padding > 0 {
        x = MLX.padded(x, widths: [
            0,  // batch
            0,  // channels
            .init((padding, padding)),  // height
            .init((padding, padding))   // width
        ])
    }
    
    let paddedH = height + 2 * padding
    let paddedW = width + 2 * padding
    
    // Calculate output dimensions
    let heightOut = (paddedH - dilation * (kernelSize - 1) - 1) / stride + 1
    let widthOut = (paddedW - dilation * (kernelSize - 1) - 1) / stride + 1
    
    // Extract blocks using array indexing
    var blocks: [MLXArray] = []
    
    for i in Swift.stride(from: 0, to: paddedH - kernelSize * dilation + 1, by: stride) {
        for j in Swift.stride(from: 0, to: paddedW - kernelSize * dilation + 1, by: stride) {
            var block: [MLXArray] = []
            for di in 0 ..< kernelSize {
                for dj in 0 ..< kernelSize {
                    let hIdx = i + di * dilation
                    let wIdx = j + dj * dilation
                    block.append(x[0..., 0..., hIdx, wIdx])
                }
            }
            // Stack the channel-blocks: (B, C, k*k)
            let stackedBlock = MLX.stacked(block, axis: 1).transposed(0, 2, 1)
            blocks.append(stackedBlock)
        }
    }
    
    // Stack all blocks: (B, C, k*k, L)
    let result = MLX.stacked(blocks, axis: -1)
    
    // Reshape to (B, C*k*k, L)
    return result.reshaped(batchSize, channels * kernelSize * kernelSize, heightOut * widthOut)
}

// MARK: - Mistral3 Patch Merger

private class Mistral3PatchMerger: Module {
    let spatialMergeSize: Int
    let patchSize: Int
    
    @ModuleInfo(key: "merging_layer") var mergingLayer: Linear
    
    init(_ config: Mistral3VLMConfiguration) {
        self.spatialMergeSize = config.spatialMergeSize
        self.patchSize = config.visionConfig.patchSize
        
        let hiddenSize = config.visionConfig.hiddenSize
        self._mergingLayer.wrappedValue = Linear(
            hiddenSize * spatialMergeSize * spatialMergeSize,
            hiddenSize,
            bias: false
        )
    }
    
    func callAsFunction(_ imageFeatures: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        // Convert image sizes to patch sizes
        let patchSizes = imageSizes.map { (h, w) in
            (h / patchSize, w / patchSize)
        }
        
        let tokensPerImage = patchSizes.map { $0.0 * $0.1 }
        let d = imageFeatures.dim(-1)
        var features = imageFeatures.asType(.bfloat16)
        
        // Split the image features into chunks based on tokens per image
        var splitIndices: [Int] = []
        var currentIndex = 0
        for tokens in tokensPerImage.dropLast() {
            currentIndex += tokens
            splitIndices.append(currentIndex)
        }
        
        let chunks: [MLXArray]
        if splitIndices.isEmpty {
            chunks = [features[0, 0..., 0...]]
        } else {
            chunks = MLX.split(features[0], indices: splitIndices, axis: 0)
        }
        
        var permutedTensors: [MLXArray] = []
        
        for (imageIndex, imageTokens) in chunks.enumerated() {
            if imageTokens.dim(0) > 0 {
                let (h, w) = patchSizes[imageIndex]
                
                // Reshape to grid: (h, w, d) -> (1, d, h, w)
                var imageGrid = imageTokens.reshaped(h, w, d).transposed(2, 0, 1)[.newAxis, 0..., 0..., 0...]
                
                // Apply unfold
                var grid = unfold(imageGrid, kernelSize: spatialMergeSize, stride: spatialMergeSize)
                
                // Reshape: (d * spatial_merge_size^2, -1).T
                grid = grid.reshaped(d * spatialMergeSize * spatialMergeSize, -1).transposed()
                permutedTensors.append(grid)
            }
        }
        
        features = MLX.concatenated(permutedTensors, axis: 0)
        features = mergingLayer(features)
        
        return features[.newAxis, 0..., 0...]
    }
}

// MARK: - Mistral3 MultiModal Projector

private class Mistral3MultiModalProjector: Module {
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "patch_merger") var patchMerger: Mistral3PatchMerger
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo var gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear
    
    init(_ config: Mistral3VLMConfiguration) {
        self._norm.wrappedValue = RMSNorm(dimensions: config.visionConfig.hiddenSize)
        self._patchMerger.wrappedValue = Mistral3PatchMerger(config)
        self._linear1.wrappedValue = Linear(
            config.visionConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: config.multimodalProjectorBias
        )
        self.gelu = GELU()
        self._linear2.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: config.multimodalProjectorBias
        )
    }
    
    func callAsFunction(_ x: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        var result = norm(x)
        result = patchMerger(result, imageSizes: imageSizes)
        result = linear1(result)
        result = gelu(result)
        result = linear2(result)
        return result
    }
}

// MARK: - Language Model Components

private enum Language {
    
    // MARK: Llama 4 Attention Scaling
    
    static func getLlama4AttentionScale(start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int) -> MLXArray {
        let positions = MLXArray(start ..< stop).asType(.float32)
        let scaling = 1 + beta * MLX.log(1 + MLX.floor(positions / Float(maxPositionEmbeddings)))
        return expandedDimensions(scaling, axis: -1)
    }
    
    // MARK: Language Attention
    
    fileprivate class Attention: Module {
        let config: Mistral3VLMTextConfiguration
        let scale: Float
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int
        
        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear
        
        let rope: RoPE
        
        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config
            
            let dim = config.hiddenSize
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numKeyValueHeads
            
            self.headDim = config.headDim ?? (config.hiddenSize / nHeads)
            self.scale = pow(Float(headDim), -0.5)
            
            self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
            
            let ropeTheta = config.ropeParameters?["rope_theta"]?.asFloat() ?? config.ropeTheta
            self.rope = RoPE(dimensions: headDim, traditional: config.ropeTraditional, base: ropeTheta)
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
            
            queries = queries * attentionScale
            
            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            
            return wo(output)
        }
    }
    
    // MARK: Language MLP
    
    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        
        init(_ config: Mistral3VLMTextConfiguration) {
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
    
    // MARK: Language Transformer Block
    
    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP
        
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        
        let useSliding: Bool
        
        init(_ config: Mistral3VLMTextConfiguration, useSliding: Bool = false) {
            self.useSliding = useSliding
            self._attention.wrappedValue = Attention(config)
            self.mlp = MLP(config)
            self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
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
    
    // MARK: Language Model Inner
    
    fileprivate class LanguageModelInner: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        
        let layers: [TransformerBlock]
        let norm: RMSNorm
        let config: Mistral3VLMTextConfiguration
        let layerTypes: [String]
        let slidingWindow: Int?
        let faIndex: Int
        let swaIndex: Int?
        
        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config
            self.slidingWindow = config.slidingWindow
            self.layerTypes = config.layerTypes ?? Array(repeating: "full_attention", count: config.numHiddenLayers)
            
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize
            )
            
            self.layers = layerTypes.map { layerType in
                TransformerBlock(config, useSliding: layerType == "sliding_attention")
            }
            
            self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            
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
            
            let cache = cache ?? []
            let offset = cache.first?.offset ?? 0
            
            let faMask = createAttentionMask(h: h, cache: cache.isEmpty ? nil : [cache[faIndex]])
            
            var swaMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
            if let swaIndex, let slidingWindow, !cache.isEmpty {
                let t = h.dim(1)
                if t > 1 {
                    let swaOffset = min(slidingWindow, cache[swaIndex].offset)
                    swaMask = .array(createCausalMask(n: t, offset: swaOffset, windowSize: slidingWindow))
                }
            }
            
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
                h = layer(h, attentionScale: attentionScale, mask: mask, cache: cache.isEmpty ? nil : cache[i])
            }
            
            return norm(h)
        }
    }
    
    // MARK: Language Model
    
    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: Mistral3VLMTextConfiguration
        @ModuleInfo(key: "model") var model: LanguageModelInner
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        
        var kvHeads: [Int] {
            let layerTypes = config.layerTypes ?? Array(repeating: "full_attention", count: config.numHiddenLayers)
            return layerTypes.map { _ in config.numKeyValueHeads }
        }
        
        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config
            self._model.wrappedValue = LanguageModelInner(config)
            
            if !config.tieWordEmbeddings {
                self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
            }
        }
        
        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            inputsEmbeds: MLXArray? = nil
        ) -> MLXArray {
            var out = model(inputs, cache: cache, inputsEmbeds: inputsEmbeds)
            if config.tieWordEmbeddings {
                out = model.embedTokens.asLinear(out)
            } else if let lmHead {
                out = lmHead(out)
            }
            return out
        }
        
        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            let layerTypes = config.layerTypes ?? Array(repeating: "full_attention", count: config.numHiddenLayers)
            
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
}

// MARK: - Mistral3 VLM Model

public class Mistral3VLM: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector: Mistral3MultiModalProjector
    
    public let config: Mistral3VLMConfiguration
    let visionFeatureLayer: Int
    
    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    
    public init(_ config: Mistral3VLMConfiguration) {
        self.config = config
        self.visionFeatureLayer = config.visionFeatureLayer
        
        self._visionTower.wrappedValue = Vision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfig)
        self._multiModalProjector.wrappedValue = Mistral3MultiModalProjector(config)
    }
    
    private func getInputEmbeddings(
        inputIds: MLXArray?,
        pixelValues: MLXArray?,
        imageSizes: [(Int, Int)]?
    ) -> MLXArray {
        guard let pixelValues, let imageSizes else {
            guard let inputIds else {
                fatalError("Either inputIds or pixelValues must be provided")
            }
            return languageModel.model.embedTokens(inputIds)
        }
        
        guard let inputIds else {
            fatalError("inputIds required when pixelValues provided")
        }
        
        let inputsEmbeds = languageModel.model.embedTokens(inputIds)
        
        // Process through vision tower
        let (_, _, hiddenStates) = visionTower(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: true
        )
        
        // Select features from specified layer
        guard let hiddenStates else {
            fatalError("Vision model must return hidden states")
        }
        
        let layerIndex = visionFeatureLayer < 0 
            ? hiddenStates.count + visionFeatureLayer 
            : visionFeatureLayer
        let selectedFeatures = hiddenStates[layerIndex]
        
        // Project to text space
        let imageFeatures = multiModalProjector(selectedFeatures, imageSizes: imageSizes)
        
        // Merge embeddings
        return mergeInputIdsWithImageFeatures(
            imageTokenIndex: config.imageTokenIndex,
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds
        )
    }
    
    private func mergeInputIdsWithImageFeatures(
        imageTokenIndex: Int,
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let (numImages, numImagePatches, embedDim) = (
            imageFeatures.dim(0),
            imageFeatures.dim(1),
            imageFeatures.dim(2)
        )
        
        // Find image token positions
        let inputIdArray: [Int32] = inputIds[0].asArray(Int32.self)
        let imagePositions = inputIdArray.enumerated().compactMap { 
            $1 == Int32(imageTokenIndex) ? $0 : nil 
        }
        
        var textSegments: [MLXArray] = []
        var startIdx = 0
        
        for position in imagePositions {
            if position > startIdx {
                textSegments.append(inputsEmbeds[0..., startIdx ..< position, 0...])
            }
            startIdx = position + 1
        }
        
        // Split image features
        let imageEmbeddings = (0 ..< numImages).map { i in
            imageFeatures[i, 0..., 0...]
        }
        
        // Interleave text and image embeddings
        var finalEmbeddings: [MLXArray] = []
        for (text, image) in zip(textSegments, imageEmbeddings) {
            finalEmbeddings.append(text[0])
            finalEmbeddings.append(image)
        }
        
        // Add remaining text
        if startIdx < inputsEmbeds.dim(1) {
            finalEmbeddings.append(inputsEmbeds[0, startIdx..., 0...])
        }
        
        return MLX.concatenated(finalEmbeddings, axis: 0)[.newAxis, 0..., 0...]
    }
    
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels
        
        // For now, assume fixed image sizes based on config
        let imageSizes: [(Int, Int)]? = pixelValues != nil 
            ? [(config.visionConfig.imageSize, config.visionConfig.imageSize)]
            : nil
        
        let embeddings = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues,
            imageSizes: imageSizes
        )
        
        let logits = languageModel(inputIds, cache: cache, inputsEmbeds: embeddings)
        return .logits(.init(logits: logits))
    }
    
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }
    
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        
        for (key, value) in weights {
            var newKey = key
            
            // Transform keys to match model structure
            if key.contains("vision_tower") && !key.contains("vision_model") {
                if key.contains("transformer") || key.contains("patch_conv") || key.contains("ln_pre") {
                    newKey = key.replacingOccurrences(of: "vision_tower", with: "vision_tower.vision_model")
                }
            } else if key.contains("vision_encoder") && !key.contains("vision_tower") {
                if key.contains("transformer") || key.contains("patch_conv") || key.contains("ln_pre") {
                    newKey = key.replacingOccurrences(of: "model.vision_encoder", with: "vision_tower.vision_model")
                }
            } else if key.contains("model.language_model") && !key.contains("language_model.model") {
                newKey = key.replacingOccurrences(of: "model.language_model", with: "language_model.model")
            } else if key.contains("lm_head") && !key.contains("language_model") {
                newKey = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            } else if key.contains("model.vision_projection") {
                newKey = key.replacingOccurrences(of: "model.vision_projection", with: "multi_modal_projector")
            }
            
            // Skip rotary embeddings
            if newKey.contains("self_attn.rotary_emb.inv_freq") {
                continue
            }
            
            // Handle weight scale patterns
            if newKey.contains("weight_scale_inv") {
                let scaleInv = value
                let weightKey = newKey.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[key.replacingOccurrences(of: "_scale_inv", with: "")] {
                    newWeights[weightKey] = weight * scaleInv
                }
            } else if newKey.contains("activation_scale") {
                continue
            } else if newWeights[newKey] == nil {
                newWeights[newKey] = value
            }
        }
        
        return newWeights
    }
    
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }
}

// MARK: - LoRA Support

extension Mistral3VLM: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Processor Configuration

public struct Mistral3VLMProcessorConfiguration: Codable, Sendable {
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: ProcessorSize
    public let imageSequenceLength: Int?
    
    public struct ProcessorSize: Codable, Sendable {
        public let width: Int?
        public let height: Int?
        public let longestEdge: Int?
        
        enum CodingKeys: String, CodingKey {
            case width
            case height
            case longestEdge = "longest_edge"
        }
    }
    
    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }
    
    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case imageSequenceLength = "image_seq_len"
    }
}

// MARK: - Processor

public class Mistral3VLMProcessor: UserInputProcessor {
    private let config: Mistral3VLMProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenId: Int
    
    public init(_ config: Mistral3VLMProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        // Default image token ID for Mistral3
        self.imageTokenId = 10
    }
    
    private func prompt(from userInput: UserInput) -> String {
        switch userInput.prompt {
        case .text(let text):
            return text
        case .messages(let messages):
            return messages.last?["content"] as? String ?? ""
        case .chat(let messages):
            return messages.last?.content ?? ""
        }
    }
    
    public func prepare(input: UserInput) throws -> LMInput {
        let prompt = prompt(from: input)
        
        if input.images.isEmpty {
            let tokens = try tokenizer.encode(text: prompt)
            let tokensArray = MLXArray(tokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        } else {
            guard input.images.count == 1 else {
                throw VLMError.singleImageAllowed
            }
            
            var promptTokens = try tokenizer.encode(text: prompt)
            
            // Insert image token
            let imageTokenIndex = promptTokens.count / 2
            promptTokens.insert(imageTokenId, at: imageTokenIndex)
            
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray)
            
            // Process image
            let targetSize = config.size.longestEdge ?? config.size.width ?? 384
            var image = try input.images[0].asCIImage()
            image = MediaProcessing.inSRGBToneCurveSpace(image)
            image = MediaProcessing.apply(image, processing: input.processing)
            image = try MediaProcessing.resampleBicubic(
                image,
                to: CGSize(width: targetSize, height: targetSize)
            )
            image = MediaProcessing.normalize(
                image,
                mean: config.imageMeanTuple,
                std: config.imageStdTuple
            )
            
            var pixels = MediaProcessing.asMLXArray(image)
            
            if pixels.ndim == 2 {
                pixels = pixels.expandedDimensions(axis: -1)
            }
            if pixels.ndim == 3 {
                pixels = pixels.expandedDimensions(axis: 0)
            }
            
            // Ensure BHWC format
            if pixels.dim(1) == 3 && pixels.dim(2) == targetSize && pixels.dim(3) == targetSize {
                pixels = pixels.transposed(0, 2, 3, 1)
            }
            
            return LMInput(
                text: .init(tokens: promptArray, mask: mask),
                image: .init(pixels: pixels)
            )
        }
    }
}

