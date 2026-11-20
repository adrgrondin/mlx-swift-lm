// Copyright © 2026 Apple Inc.
// Vision architecture adapted from ml-explore/mlx-lm#2086 (Apache-2.0).

import Foundation
import MLX
import MLXNN

struct Inkling2VisionConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numHeads: Int
    let patchSize: Int
    let mergeSize: Int
    let positionGrid: Int
    let projectionSize: Int
    let projectionBias: Bool
    let mlpRatio: Float
    let normEps: Float
    let ropeTheta: Float?

    var headDim: Int { hiddenSize / numHeads }
    var blockSize: Int { patchSize * mergeSize }
    var isValid: Bool {
        numHeads > 0 && hiddenSize > 0 && hiddenSize.isMultiple(of: numHeads)
            && headDim.isMultiple(of: 4) && patchSize > 0 && mergeSize > 0 && positionGrid > 0
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numHeads = "num_attention_heads"
        case patchSize = "native_patch_size"
        case mergeSize = "adapter_merge_size"
        case positionGrid = "posemb_grid_size"
        case projectionSize = "projection_hidden_dim"
        case projectionBias = "projector_bias"
        case mlpRatio = "mlp_ratio"
        case normEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
    }
}

final class Inkling2VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patch: Linear
    @ModuleInfo(key: "position_embedding") var position: Embedding
    @ParameterInfo(key: "class_embedding") var cls: MLXArray
    @ParameterInfo(key: "cls_position_embedding") var clsPosition: MLXArray
    let grid: Int

    init(_ c: Inkling2VisionConfiguration) {
        grid = c.positionGrid
        _patch.wrappedValue = Linear(3 * c.patchSize * c.patchSize, c.hiddenSize, bias: false)
        _position.wrappedValue = Embedding(embeddingCount: grid * grid, dimensions: c.hiddenSize)
        _cls.wrappedValue = .zeros([c.hiddenSize])
        _clsPosition.wrappedValue = .zeros([c.hiddenSize])
    }

    func callAsFunction(_ patches: MLXArray, rows: Int, cols: Int, row: MLXArray, col: MLXArray)
        -> MLXArray
    {
        func taps(_ index: MLXArray, size: Int) -> (MLXArray, MLXArray) {
            let source = (index.asType(.float32) + 0.5) * Float(grid) / Float(size) - 0.5
            let points = floor(source).expandedDimensions(axis: -1) + MLXArray([0, 1])
            let weights = maximum(1 - abs(source.expandedDimensions(axis: -1) - points), 0)
            return (clip(points, min: 0, max: grid - 1).asType(.int32), weights)
        }
        let (r, rw) = taps(row, size: rows)
        let (c, cw) = taps(col, size: cols)
        let indices = r.expandedDimensions(axis: -1) * grid + c.expandedDimensions(axis: -2)
        let weights = rw.expandedDimensions(axis: -1) * cw.expandedDimensions(axis: -2)
        let p = patch(patches.asType(patch.weight.dtype))
        let interpolated =
            (position(indices.reshaped(-1, 4)).asType(.float32) * weights.reshaped(-1, 4, 1))
            .sum(axis: 1).asType(p.dtype)
        return concatenated([
            (cls + clsPosition).asType(p.dtype).expandedDimensions(axis: 0), p + interpolated,
        ])
    }
}

final class Inkling2VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    let heads: Int
    let scale: Float

    init(_ c: Inkling2VisionConfiguration) {
        heads = c.numHeads
        scale = 1 / sqrt(Float(c.headDim))
        let dims = c.hiddenSize
        _q.wrappedValue = Linear(dims, dims)
        _k.wrappedValue = Linear(dims, dims)
        _v.wrappedValue = Linear(dims, dims)
        _out.wrappedValue = Linear(dims, dims)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = x.dim(0)
        func rotate(_ x: MLXArray) -> MLXArray {
            let states = x.asType(.float32)
            let pairs = states.reshaped(length, heads, -1, 2)
            let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1).reshaped(
                x.shape)
            return (states * cos + rotated * sin).asType(x.dtype)
        }
        func pack(_ x: MLXArray) -> MLXArray { x.transposed(1, 0, 2).expandedDimensions(axis: 0) }
        let query = pack(rotate(q(x).reshaped(length, heads, -1)))
        let key = pack(rotate(k(x).reshaped(length, heads, -1)))
        let value = pack(v(x).reshaped(length, heads, -1))
        let result = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: scale, mask: nil)
        return out(result[0].transposed(1, 0, 2).reshaped(x.shape))
    }
}

private func inkling2GELU(_ x: MLXArray) -> MLXArray { gelu(x.asType(.float32)).asType(x.dtype) }

final class Inkling2VisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(_ c: Inkling2VisionConfiguration) {
        let hidden = Int(Float(c.hiddenSize) * c.mlpRatio)
        _fc1.wrappedValue = Linear(c.hiddenSize, hidden)
        _fc2.wrappedValue = Linear(hidden, c.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(inkling2GELU(fc1(x))) }
}

final class Inkling2VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2VisionAttention
    @ModuleInfo(key: "layernorm_before") var before: LayerNorm
    @ModuleInfo(key: "layernorm_after") var after: LayerNorm
    @ParameterInfo(key: "lambda_1") var scale1: MLXArray
    @ParameterInfo(key: "lambda_2") var scale2: MLXArray
    @ModuleInfo var mlp: Inkling2VisionMLP

    init(_ c: Inkling2VisionConfiguration) {
        _attention.wrappedValue = Inkling2VisionAttention(c)
        _before.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _after.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _scale1.wrappedValue = .ones([c.hiddenSize])
        _scale2.wrappedValue = .ones([c.hiddenSize])
        _mlp.wrappedValue = Inkling2VisionMLP(c)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + scale1 * attention(before(x), cos: cos, sin: sin)
        return h + scale2 * mlp(after(h))
    }
}

final class Inkling2VisionMerger: Module {
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear

    init(_ c: Inkling2VisionConfiguration, outputSize: Int) {
        _first.wrappedValue = Linear(
            c.mergeSize * c.mergeSize * c.hiddenSize, c.projectionSize, bias: c.projectionBias)
        _second.wrappedValue = Linear(c.projectionSize, outputSize, bias: c.projectionBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { second(inkling2GELU(first(x))) }
}

final class Inkling2Vision: Module {
    @ModuleInfo var embeddings: Inkling2VisionEmbeddings
    @ModuleInfo(key: "pre_layernorm") var preNorm: LayerNorm
    @ModuleInfo var layers: [Inkling2VisionLayer]
    @ModuleInfo var merger: Inkling2VisionMerger
    let c: Inkling2VisionConfiguration

    init(_ c: Inkling2VisionConfiguration, outputSize: Int) {
        self.c = c
        _embeddings.wrappedValue = Inkling2VisionEmbeddings(c)
        _preNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in Inkling2VisionLayer(c) }
        _merger.wrappedValue = Inkling2VisionMerger(c, outputSize: outputSize)
    }

    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let (p, m) = (c.patchSize, c.mergeSize)
        let (rows, cols) = (pixels.dim(0) / c.blockSize, pixels.dim(1) / c.blockSize)
        let patches = pixels.reshaped(rows, m, p, cols, m, p, 3)
            .transposed(0, 3, 1, 4, 2, 5, 6).reshaped(-1, p * p * 3)
        // The merge blocks are in raster order, then the patches within each block.
        let i = MLXArray(0 ..< (rows * cols * m * m))
        let row = floorDivide(i, m * m * cols) * m + floorDivide(i, m) % m
        let col = floorDivide(i, m * m) % cols * m + i % m
        var h = preNorm(embeddings(patches, rows: rows * m, cols: cols * m, row: row, col: col))
        let dims = c.headDim / 2
        let indices = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
        let inverse = 1 / pow(c.ropeTheta ?? 10000, indices / Float(dims))
        let r =
            concatenated([MLXArray([0]), row + 1]).asType(.float32).expandedDimensions(axis: 1)
            * inverse
        let w =
            concatenated([MLXArray([0]), col + 1]).asType(.float32).expandedDimensions(axis: 1)
            * inverse
        let angles = repeated(concatenated([w, r], axis: -1), count: 2, axis: -1)
            .expandedDimensions(axis: 1)
        let cos = MLX.cos(angles)
        let sin = MLX.sin(angles)
        for layer in layers { h = layer(h, cos: cos, sin: sin) }
        return merger(h[1...].reshaped(rows * cols, -1))
    }
}
