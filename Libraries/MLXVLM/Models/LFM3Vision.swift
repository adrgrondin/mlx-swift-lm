// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// LFM3-VL's hybrid spatial encoder configuration.
public struct LFM3VisionConfiguration: Codable, Sendable, ModelConfigurationValidating {
    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let channels: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let mergeSize: Int
    let kernelSize: Int
    let normEps: Float
    let layerTypes: [String]
    let ropeParameters: [String: StringOrNumber]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case channels = "in_channels"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case mergeSize = "spatial_merge_size"
        case kernelSize = "conv_kernel_size"
        case normEps = "norm_eps"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2560
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 14
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 3
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< max(hiddenLayers, 0)).map {
                $0 >= 2 && $0 % 2 == 0 ? "full_attention" : "conv"
            }
        ropeParameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? ["rope_type": .string("axial"), "rope_theta": .float(1e6)]
    }

    public func validateModelConfiguration() throws {
        guard hiddenSize > 0, intermediateSize > 0, hiddenLayers > 0,
            attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0,
            hiddenSize % attentionHeads == 0, (hiddenSize / attentionHeads) % 4 == 0,
            channels == 3, patchSize > 0, temporalPatchSize > 0, mergeSize > 0,
            kernelSize > 0, kernelSize % 2 == 1, normEps.isFinite, normEps > 0,
            layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ ["conv", "full_attention"].contains($0) }),
            ropeParameters["rope_type"] == .string("axial"),
            let theta = ropeParameters["rope_theta"]?.asFloat(), theta.isFinite, theta > 0
        else {
            throw ModelFactoryError.invalidConfiguration("Unsupported LFM3 vision configuration")
        }
    }
}

final class LFM3VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm
    let heads: Int
    let kvHeads: Int
    let headDim: Int

    init(_ c: LFM3VisionConfiguration) {
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        headDim = c.hiddenSize / heads
        _qProj.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.normEps)
    }

    private func rotate(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let halves = x.asType(.float32).split(parts: 2, axis: -1)
        return
            (x.asType(.float32) * cosine + concatenated([-halves[1], halves[0]], axis: -1) * sine)
            .asType(x.dtype)
    }

    func callAsFunction(_ x: MLXArray, shapes: [THW], cosine: MLXArray, sine: MLXArray) -> MLXArray
    {
        let q = rotate(qNorm(qProj(x).reshaped(-1, heads, headDim)), cosine: cosine, sine: sine)
        let k = rotate(kNorm(kProj(x).reshaped(-1, kvHeads, headDim)), cosine: cosine, sine: sine)
        let v = vProj(x).reshaped(-1, kvHeads, headDim)
        func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, frames: Int, length: Int)
            -> MLXArray
        {
            MLXFast.scaledDotProductAttention(
                queries: q.reshaped(frames, length, heads, headDim).transposed(0, 2, 1, 3),
                keys: k.reshaped(frames, length, kvHeads, headDim).transposed(0, 2, 1, 3),
                values: v.reshaped(frames, length, kvHeads, headDim).transposed(0, 2, 1, 3),
                scale: pow(Float(headDim), -0.5), mask: .none
            )
            .transposed(0, 2, 1, 3).reshaped(-1, heads * headDim)
        }
        let lengths = shapes.map { $0.h * $0.w }
        if Set(lengths).count == 1 {
            return outProj(attend(q, k, v, frames: shapes.count, length: lengths[0]))
        }
        var offset = 0
        let output = lengths.map { length in
            defer { offset += length }
            return attend(
                q[offset ..< offset + length], k[offset ..< offset + length],
                v[offset ..< offset + length], frames: 1, length: length)
        }
        return outProj(concatenated(output))
    }
}

final class LFM3VisionConv: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "spatial_conv") var spatialConv: Conv2d
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let merge: Int

    init(_ c: LFM3VisionConfiguration) {
        merge = c.mergeSize
        _inProj.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: false)
        _spatialConv.wrappedValue = Conv2d(
            inputChannels: c.hiddenSize, outputChannels: c.hiddenSize,
            kernelSize: .init(c.kernelSize), padding: .init(c.kernelSize / 2), groups: c.hiddenSize,
            bias: false)
        _outProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, shapes: [THW]) -> MLXArray {
        let parts = inProj(x).split(parts: 3, axis: -1)
        let gated = parts[0] * parts[2]
        func convolve(_ input: MLXArray, frames: Int, h: Int, w: Int) -> MLXArray {
            let raster = input.reshaped(frames, h / merge, w / merge, merge, merge, -1)
                .transposed(0, 1, 3, 2, 4, 5).reshaped(frames, h, w, -1)
            return spatialConv(raster).reshaped(frames, h / merge, merge, w / merge, merge, -1)
                .transposed(0, 1, 3, 2, 4, 5).reshaped(-1, x.dim(-1))
        }
        let first = shapes[0]
        let output: MLXArray
        if shapes.allSatisfy({ $0.h == first.h && $0.w == first.w }) {
            output = convolve(gated, frames: shapes.count, h: first.h, w: first.w)
        } else {
            var offset = 0
            output = concatenated(
                shapes.map { shape in
                    defer { offset += shape.h * shape.w }
                    return convolve(
                        gated[offset ..< offset + shape.h * shape.w], frames: 1, h: shape.h,
                        w: shape.w)
                })
        }
        return outProj(parts[1] * output)
    }
}

final class LFM3VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM3VisionAttention?
    @ModuleInfo var conv: LFM3VisionConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ c: LFM3VisionConfiguration, type: String) {
        if type == "conv" {
            _conv.wrappedValue = LFM3VisionConv(c)
        } else {
            _attention.wrappedValue = LFM3VisionAttention(c)
        }
        _feedForward.wrappedValue = LFM3MLP(
            hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
    }

    func callAsFunction(_ x: MLXArray, shapes: [THW], cosine: MLXArray, sine: MLXArray) -> MLXArray
    {
        let normalized = operatorNorm(x)
        let h =
            x
            + (attention?(normalized, shapes: shapes, cosine: cosine, sine: sine)
                ?? conv!(normalized, shapes: shapes))
        return h + feedForward(ffnNorm(h))
    }
}

final class LFM3VisionModel: Module {
    @ModuleInfo var patchifier: Linear
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    let layers: [LFM3VisionLayer]
    let config: LFM3VisionConfiguration

    init(_ c: LFM3VisionConfiguration) {
        config = c
        _patchifier.wrappedValue = Linear(
            c.channels * c.temporalPatchSize * c.patchSize * c.patchSize, c.hiddenSize, bias: false)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        layers = c.layerTypes.map { LFM3VisionLayer(c, type: $0) }
    }

    func callAsFunction(_ pixels: MLXArray, grids: [THW]) throws -> MLXArray {
        guard !grids.isEmpty,
            grids.allSatisfy({
                $0.t > 0 && $0.h > 0 && $0.w > 0 && $0.h % config.mergeSize == 0
                    && $0.w % config.mergeSize == 0
            }),
            pixels.ndim == 2, grids.reduce(0, { $0 + $1.product }) == pixels.dim(0)
        else { throw VLMError.processing("Invalid LFM3 vision patch grids") }
        let shapes = grids.flatMap { grid in Array(repeating: THW(1, grid.h, grid.w), count: grid.t)
        }
        var rows: [Int32] = []
        var columns: [Int32] = []
        let m = config.mergeSize
        for shape in shapes {
            for h in stride(from: 0, to: shape.h, by: m) {
                for w in stride(from: 0, to: shape.w, by: m) {
                    for dh in 0 ..< m {
                        for dw in 0 ..< m {
                            rows.append(Int32(h + dh))
                            columns.append(Int32(w + dw))
                        }
                    }
                }
            }
        }
        let spatialDim = config.hiddenSize / config.attentionHeads / 2
        let inverse =
            1
            / pow(
                config.ropeParameters["rope_theta"]?.asFloat() ?? 1e6,
                MLXArray(stride(from: 0, to: spatialDim, by: 2)).asType(.float32)
                    / Float(spatialDim))
        let hAngles = MLXArray(rows).asType(.float32)[0..., .newAxis] * inverse
        let wAngles = MLXArray(columns).asType(.float32)[0..., .newAxis] * inverse
        let angles = concatenated([hAngles, wAngles, hAngles, wAngles], axis: -1)
        let cosine = cos(angles)[0..., .newAxis, 0...]
        let sine = sin(angles)[0..., .newAxis, 0...]
        var h = embeddingNorm(patchifier(pixels.asType(embeddingNorm.weight.dtype)))
        for layer in layers { h = layer(h, shapes: shapes, cosine: cosine, sine: sine) }
        return h
    }
}

final class LFM3MultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let mergedSize: Int

    init(_ c: LFM3VLConfiguration) {
        mergedSize = c.visionConfiguration.hiddenSize * c.downsampleFactor * c.downsampleFactor
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: c.visionConfiguration.hiddenSize, eps: c.projectorNormEps)
        _linear1.wrappedValue = Linear(mergedSize, c.projectorHiddenSize, bias: c.projectorBias)
        _linear2.wrappedValue = Linear(
            c.projectorHiddenSize, c.textConfiguration.hiddenSize, bias: c.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(layerNorm(x).reshaped(-1, mergedSize))))
    }
}
