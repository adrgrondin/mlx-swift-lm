// Copyright © 2026 Apple Inc.
// Decoder architecture adapted from ml-explore/mlx-lm#2086 (Apache-2.0).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

func inkling2RMS(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x.asType(.float32), weight: .ones([x.dim(-1)]), eps: eps)
}

private func depthAttention(_ states: [MLXArray], query: MLXArray) -> MLXArray {
    let dims = query.dim(-1)
    let values = stacked(states, axis: -2)
    let scores = matmul(values, query.reshaped(-1, dims).T) / sqrt(Float(dims))
    return matmul(softmax(scores, axis: -2).swappedAxes(-1, -2), values)
}

final class Inkling2DepthAttention: Module {
    let query: MLXArray

    init(dimensions: Int, queries: Int = 1) {
        query = .zeros(queries == 1 ? [dimensions] : [queries, dimensions])
    }

    func callAsFunction(_ blocks: [MLXArray], _ partial: MLXArray) -> MLXArray {
        depthAttention(blocks + [partial], query: query)
    }
}

final class Inkling2MessageInput: Module {
    let query: MLXArray

    init(dimensions: Int, layers: Int) {
        query = .zeros([layers, 4, dimensions])
    }

    func callAsFunction(_ bank: [MLXArray], layer: Int) -> MLXArray {
        depthAttention(bank, query: query[layer])
    }
}

final class Inkling2MessageGate: Module {
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "forget_gate_proj") var forget: Linear
    @ModuleInfo(key: "message_gate_proj") var message: Linear

    init(_ dims: Int, rank: Int) {
        _down.wrappedValue = Linear(2 * dims, rank, bias: false)
        _forget.wrappedValue = Linear(rank, dims, bias: false)
        _message.wrappedValue = Linear(rank, dims, bias: false)
    }

    func callAsFunction(_ x: MLXArray, _ msg: MLXArray) -> MLXArray {
        let reduced = down(
            concatenated([inkling2RMS(x), inkling2RMS(msg)], axis: -1).asType(down.weight.dtype)
        )
        .asType(.float32)
        return (1 - tanh(forget(reduced))) * x + tanh(message(reduced)) * msg
    }
}

final class Inkling2ShortConvolution: Module {
    @ModuleInfo var conv1d: Conv1d
    let size: Int

    init(_ dims: Int, size: Int) {
        self.size = size
        _conv1d.wrappedValue = Conv1d(
            inputChannels: dims, outputChannels: dims, kernelSize: size, groups: dims, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?, index: Int) -> MLXArray {
        let window = inkling2ConvolutionWindow(x, size: size, cache: cache, index: index)
        let out = conv1d(window.asType(conv1d.weight.dtype)).asType(.float32)
        return ((out + x.asType(.float32)) / sqrt(Float(size + 1))).asType(x.dtype)
    }
}

private func inkling2ConvolutionWindow(_ x: MLXArray, size: Int, cache: ArraysCache?, index: Int)
    -> MLXArray
{
    let history = cache?[index] ?? MLXArray.zeros([x.dim(0), size - 1, x.dim(2)])
    let window = concatenated([history, x.asType(.float32)], axis: 1)
    if let cache {
        // Gather a copy so a short state does not retain the whole prefill buffer.
        cache[index] = take(window, MLXArray(x.dim(1) ..< window.dim(1)), axis: 1)
    }
    return window
}

final class Inkling2DynamicConvolution: Module {
    @ParameterInfo(key: "static_weight") var staticWeight: MLXArray
    @ModuleInfo(key: "weight_proj") var weightProjection: Linear
    let heads: Int
    let size: Int

    init(_ dims: Int, inputSize: Int, heads: Int, size: Int) {
        self.heads = heads
        self.size = size
        _staticWeight.wrappedValue = .zeros([size, dims])
        _weightProjection.wrappedValue = Linear(inputSize, heads * size, bias: false)
    }

    func callAsFunction(_ x: MLXArray, input: MLXArray, cache: ArraysCache?, index: Int) -> MLXArray
    {
        let (b, length, channels) = (x.dim(0), x.dim(1), x.dim(2))
        let dynamic = weightProjection(input).asType(.float32).reshaped(b, length, heads, 1, size)
        let fixed = staticWeight.asType(.float32).T.reshaped(heads, channels / heads, size)
        let window = inkling2ConvolutionWindow(x, size: size, cache: cache, index: index)
        let taps = (0 ..< size).map { lag in
            let start = size - 1 - lag
            return window[0..., start ..< (start + length), 0...]
        }
        let values = stacked(taps, axis: -1).reshaped(b, length, heads, channels / heads, size)
        let filtered = (values * (fixed + dynamic)).sum(axis: -1).reshaped(x.shape)
        return ((filtered + x.asType(.float32)) / sqrt(Float(2 * size + 1))).asType(x.dtype)
    }
}

final class Inkling2Clip: Module {
    let scale: MLXArray
    let threshold: Float

    init(_ heads: Int, _ dims: Int, threshold: Float) {
        self.scale = .ones([heads, dims])
        self.threshold = threshold
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        Self.clip(x, scale: scale, threshold: threshold)
    }

    static func clip(_ x: MLXArray, scale: MLXArray? = nil, threshold: Float) -> MLXArray {
        let values = x.asType(.float32)
        let scaled = scale.map { values * $0 } ?? values
        let norm = sqrt(square(scaled).sum(axis: -1, keepDims: true))
        return (scaled * minimum(threshold / (norm + 1e-6), 1)).asType(x.dtype)
    }
}

final class Inkling2RelativeLogits: Module {
    let proj: MLXArray

    init(_ dims: Int, extent: Int) { proj = .zeros([dims, extent]) }

    func callAsFunction(_ x: MLXArray) -> MLXArray { matmul(x, proj.asType(x.dtype)) }
}

final class Inkling2Attention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "r_proj") var r: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "sink_proj") var sink: Linear
    @ModuleInfo(key: "k_sconv") var kConv: Module
    @ModuleInfo(key: "v_sconv") var vConv: Module
    @ModuleInfo(key: "q_l2_clip") var qClip: Inkling2Clip
    @ModuleInfo(key: "rel_l2_clip") var rClip: Inkling2Clip
    @ModuleInfo(key: "rel_logits_proj") var relative: Inkling2RelativeLogits

    let c: Inkling2Configuration.Text
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int
    let extent: Int

    init(_ c: Inkling2Configuration.Text, sliding: Bool) {
        self.c = c
        heads = sliding ? c.slidingHeads : c.numHeads
        kvHeads = sliding ? c.slidingKVHeads : c.kvHeads
        headDim = sliding ? c.slidingHeadDim : c.headDim
        window = sliding ? c.slidingWindow : 0
        extent = sliding ? c.slidingWindow : c.relativeExtent
        let hidden = c.hiddenSize
        let qSize = heads * headDim
        let kvSize = kvHeads * headDim
        _q.wrappedValue = Linear(hidden, qSize, bias: false)
        _k.wrappedValue = Linear(hidden, kvSize, bias: false)
        _v.wrappedValue = Linear(hidden, kvSize, bias: false)
        _r.wrappedValue = Linear(hidden, heads * c.relativeDim, bias: false)
        _o.wrappedValue = Linear(qSize, hidden, bias: false)
        _gate.wrappedValue = Linear(hidden, qSize, bias: false)
        _sink.wrappedValue = Linear(hidden, heads, bias: false)
        if sliding {
            _kConv.wrappedValue = Inkling2ShortConvolution(kvSize, size: c.convSize)
            _vConv.wrappedValue = Inkling2ShortConvolution(kvSize, size: c.convSize)
        } else {
            _kConv.wrappedValue = Inkling2DynamicConvolution(
                kvSize, inputSize: hidden, heads: kvHeads, size: c.convSize)
            _vConv.wrappedValue = Inkling2DynamicConvolution(
                kvSize, inputSize: hidden, heads: kvHeads, size: c.convSize)
        }
        _qClip.wrappedValue = Inkling2Clip(heads, headDim, threshold: c.qkClip)
        _rClip.wrappedValue = Inkling2Clip(heads, c.relativeDim, threshold: c.relativeClip)
        _relative.wrappedValue = Inkling2RelativeLogits(c.relativeDim, extent: extent)
    }

    func callAsFunction(
        _ qInput: MLXArray, _ kInput: MLXArray, _ vInput: MLXArray, cache: CacheList?
    ) -> MLXArray {
        let kv = cache?[0]
        let conv = cache?[1] as? ArraysCache
        let (b, length) = (qInput.dim(0), qInput.dim(1))
        let offset = kv?.offset ?? 0
        var queries = qClip(q(qInput).reshaped(b, length, heads, headDim))
        var rel = rClip(r(qInput).reshaped(b, length, heads, c.relativeDim))
        var keys = k(kInput)
        var values = v(vInput)
        if window > 0 {
            keys = (kConv as! Inkling2ShortConvolution)(keys, cache: conv, index: 0)
            values = (vConv as! Inkling2ShortConvolution)(values, cache: conv, index: 1)
        } else {
            keys = (kConv as! Inkling2DynamicConvolution)(
                keys, input: kInput, cache: conv, index: 0)
            values = (vConv as! Inkling2DynamicConvolution)(
                values, input: vInput, cache: conv, index: 1)
            let positions = MLXArray((offset + 1) ... (offset + length)).asType(.float32)
            let tau = (1 + c.logScalingAlpha * log(maximum(positions / c.logScalingFloor, 1)))
                .reshaped(length, 1, 1)
            queries = (queries.asType(.float32) * tau).asType(queries.dtype)
            rel = (rel.asType(.float32) * tau).asType(rel.dtype)
        }
        queries = queries.transposed(0, 2, 1, 3)
        keys = Inkling2Clip.clip(keys.reshaped(b, length, kvHeads, headDim), threshold: c.qkClip)
            .transposed(0, 2, 1, 3)
        values = inkling2RMS(values.reshaped(b, length, kvHeads, headDim), eps: c.normEps)
            .asType(values.dtype).transposed(0, 2, 1, 3)
        let cacheState = AttentionKVState.update(keys: keys, values: values, cache: kv)
        let rotating = kv as? RotatingKVCache
        switch cacheState {
        case .regular(let cachedKeys, let cachedValues):
            let ordered = length == 1 ? rotating?.currentKeyValues() : nil
            (keys, values) = ordered ?? (cachedKeys, cachedValues)
        case .quantized(let packedKeys, let packedValues, let groupSize, let bits, let mode):
            // The compressed attention kernels do not take per-query sinks.
            keys = dequantized(
                packedKeys.0, scales: packedKeys.1, biases: packedKeys.2,
                groupSize: groupSize, bits: bits, mode: mode, dtype: keys.dtype)
            values = dequantized(
                packedValues.0, scales: packedValues.1, biases: packedValues.2,
                groupSize: groupSize, bits: bits, mode: mode, dtype: values.dtype)
        case .native(let cache, _):
            (keys, values) = cache.currentKeyValues()!
        }
        let bias = relative(rel).transposed(0, 2, 1, 3)
        let out = attend(
            queries, keys, values, bias: bias, sinks: sink(qInput), offset: offset,
            capacity: rotating?.maxSize, keep: rotating?.keepCount ?? 0)
        let gates = gate(qInput)
        let gated =
            out.transposed(0, 2, 1, 3).reshaped(gates.shape).asType(.float32)
            * sigmoid(gates.asType(.float32))
        return o(gated.asType(gates.dtype))
    }

    private func attend(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray, sinks: MLXArray,
        offset: Int, capacity: Int?, keep: Int
    ) -> MLXArray {
        let (b, length, keys) = (q.dim(0), q.dim(2), k.dim(2))
        let end = offset + length
        let keyIndices = MLXArray(0 ..< keys)
        let pinned = keyIndices .< keep
        // Pinned keys keep their original positions even when the recent tail has wrapped.
        let keyPositions =
            keep == 0
            ? keyIndices + (end - keys) : which(pinned, keyIndices, keyIndices + (end - keys))
        let distance = MLXArray(offset ..< end).expandedDimensions(axis: -1) - keyPositions
        var visible = distance .>= 0
        let recentWindow = capacity.map { $0 - keep } ?? window
        if recentWindow > 0 { visible = visible .&& (pinned .|| (distance .< recentWindow)) }
        let index = broadcast(
            clip(distance, min: 0, max: extent - 1), to: [b, heads, length, keys])
        let relativeBias = takeAlong(bias, index, axis: -1)
        let mask = which(visible, which(distance .< extent, relativeBias, 0), -Float.infinity)
        let scale = 1 / sqrt(Float(headDim))
        if b == 1 && length == 1 {
            return MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask.asType(q.dtype),
                sinks: sinks.reshaped(heads).asType(q.dtype))
        }
        let group = heads / kvHeads
        let queries = q.asType(.float32).reshaped(b, kvHeads, group, length, headDim)
        let keysT = k.asType(.float32).swappedAxes(-1, -2).expandedDimensions(axis: 2)
        let scores =
            matmul(queries, keysT) * scale
            + mask.asType(.float32).reshaped(b, kvHeads, group, length, keys)
        let sinkScores = sinks.asType(.float32).transposed(0, 2, 1).reshaped(
            b, kvHeads, group, length, 1)
        let probs = softmax(concatenated([scores, sinkScores], axis: -1), axis: -1)[
            .ellipsis, ..<keys]
        return matmul(probs.asType(v.dtype), v.expandedDimensions(axis: 2)).reshaped(
            b, heads, length, headDim)
    }
}

final class Inkling2DecoderLayer: Module {
    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2DepthAttention?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2DepthAttention
    @ModuleInfo(key: "self_attn") var attention: Inkling2Attention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm?
    @ModuleInfo(key: "q_input_layernorm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_input_layernorm") var kNorm: RMSNorm?
    @ModuleInfo(key: "v_input_layernorm") var vNorm: RMSNorm?
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "post_attention_convolution") var postAttentionConv: Inkling2ShortConvolution
    @ModuleInfo(key: "post_mlp_convolution") var postMLPConv: Inkling2ShortConvolution
    @ModuleInfo(key: "q_message_gate") var qMessage: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") var kMessage: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") var vMessage: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") var mlpMessage: Inkling2MessageGate?
    @ModuleInfo var mlp: Inkling2MoE

    let boundary: Bool
    let sliding: Bool

    init(_ c: Inkling2Configuration.Text, index: Int) {
        boundary = index.isMultiple(of: c.blockSize)
        sliding = c.localLayerIDs.contains(index)
        let dims = c.hiddenSize
        if index > 0 {
            _attentionInput.wrappedValue = Inkling2DepthAttention(
                dimensions: dims, queries: sliding ? 1 : 3)
        }
        _mlpInput.wrappedValue = Inkling2DepthAttention(dimensions: dims)
        _attention.wrappedValue = Inkling2Attention(c, sliding: sliding)
        if sliding {
            _inputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: c.normEps)
        } else {
            _qNorm.wrappedValue = RMSNorm(dimensions: dims, eps: c.normEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: dims, eps: c.normEps)
            _vNorm.wrappedValue = RMSNorm(dimensions: dims, eps: c.normEps)
        }
        _postNorm.wrappedValue = RMSNorm(dimensions: dims, eps: c.normEps)
        _postAttentionConv.wrappedValue = Inkling2ShortConvolution(dims, size: c.convSize)
        _postMLPConv.wrappedValue = Inkling2ShortConvolution(dims, size: c.convSize)
        _mlp.wrappedValue = Inkling2MoE(c)
        if c.core.contains(index) {
            let rank = c.loop.gateRank
            _qMessage.wrappedValue = Inkling2MessageGate(dims, rank: rank)
            _kMessage.wrappedValue = Inkling2MessageGate(dims, rank: rank)
            _vMessage.wrappedValue = Inkling2MessageGate(dims, rank: rank)
            _mlpMessage.wrappedValue = Inkling2MessageGate(dims, rank: rank)
        }
    }

    func callAsFunction(
        _ x: MLXArray, blocks: inout [MLXArray], cache: CacheList?, messages: MLXArray?
    ) -> MLXArray {
        let normed = inkling2RMS(x)
        let inputs = attentionInput?(blocks, normed)
        var q = inputs?[.ellipsis, 0, 0...] ?? x
        var k = sliding ? q : (inputs?[.ellipsis, 1, 0...] ?? x)
        var v = sliding ? q : (inputs?[.ellipsis, 2, 0...] ?? x)
        if let messages, let qMessage, let kMessage, let vMessage {
            q = qMessage(q, messages[.ellipsis, 0, 0...])
            k = kMessage(k, messages[.ellipsis, 1, 0...])
            v = vMessage(v, messages[.ellipsis, 2, 0...])
        }
        q = (inputNorm ?? qNorm!)(q).asType(x.dtype)
        if sliding && messages == nil {
            k = q
            v = q
        } else {
            k = (inputNorm ?? kNorm!)(k).asType(x.dtype)
            v = (inputNorm ?? vNorm!)(v).asType(x.dtype)
        }
        let conv = cache?[1] as? ArraysCache
        let attended = attention(q, k, v, cache: cache)
        var residual = postAttentionConv(attended, cache: conv, index: 2).asType(.float32)
        if boundary {
            blocks.append(normed)
        } else {
            residual = residual + x.asType(.float32)
        }
        var mlpStates = mlpInput(blocks, inkling2RMS(residual))[.ellipsis, 0, 0...]
        if let messages, let mlpMessage {
            mlpStates = mlpMessage(mlpStates, messages[.ellipsis, 3, 0...])
        }
        let out = postMLPConv(mlp(postNorm(mlpStates).asType(x.dtype)), cache: conv, index: 3)
        return (residual + out.asType(.float32)).asType(x.dtype)
    }
}

final class Inkling2ExpertLinear: Module {
    let weight: MLXArray
    @ParameterInfo var scales: MLXArray?
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray?
    let mode: QuantizationMode?

    init(
        weight: MLXArray, scales: MLXArray? = nil, globalScale: MLXArray? = nil,
        mode: QuantizationMode? = nil
    ) {
        self.weight = weight
        self.mode = mode
        self._scales.wrappedValue = scales
        self._globalScale.wrappedValue = globalScale
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        if let scales, let mode {
            return gatherQuantizedMM(
                x, weight, scales: scales, rhsIndices: indices, mode: mode,
                globalScale: globalScale, sortedIndices: sorted)
        }
        return gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
    }
}

final class Inkling2Experts: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUp: Inkling2ExpertLinear
    @ModuleInfo(key: "down_proj") var down: Inkling2ExpertLinear
    let c: Inkling2Configuration.Text

    init(_ c: Inkling2Configuration.Text) {
        self.c = c
        _gateUp.wrappedValue = Inkling2ExpertLinear(
            weight: .zeros([c.experts, 2 * c.intermediateSize, c.hiddenSize]))
        _down.wrappedValue = Inkling2ExpertLinear(
            weight: .zeros([c.experts, c.hiddenSize, c.intermediateSize]))
    }

    func prepare(_ weights: [String: MLXArray], prefix: String) throws {
        var replacements = [(String, Module)]()
        for (name, input, output) in [
            ("gate_up_proj", c.hiddenSize, 2 * c.intermediateSize),
            ("down_proj", c.intermediateSize, c.hiddenSize),
        ] {
            let path = "\(prefix).\(name)"
            guard let scales = weights["\(path).scales"] else { continue }
            let experts = c.experts
            guard let weight = weights["\(path).weight"], weight.dtype == .uint32,
                scales.dtype == .uint8
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "Invalid Inkling 2 expert weights at \(path)")
            }
            let globalScale = weights["\(path).global_scale"]
            let mode: QuantizationMode
            if weight.shape == [experts, output, input / 8],
                scales.shape == [experts, output, input / 16]
            {
                mode = .nvfp4
            } else if weight.shape == [experts, output, input / 4],
                scales.shape == [experts, output, input / 32]
            {
                mode = .mxfp8
            } else {
                throw ModelFactoryError.invalidConfiguration(
                    "Unsupported Inkling 2 expert layout at \(path)")
            }
            guard mode == .nvfp4 ? globalScale?.shape == [experts] : globalScale == nil else {
                throw ModelFactoryError.invalidConfiguration(
                    "Invalid Inkling 2 global scale at \(path)")
            }
            replacements.append(
                (
                    name,
                    Inkling2ExpertLinear(
                        weight: weight, scales: scales, globalScale: globalScale, mode: mode)
                ))
        }
        try update(modules: ModuleChildren.unflattened(replacements), verify: [.noUnusedKeys])
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var inputs = x.expandedDimensions(axes: [-2, -3])
        var selected = indices
        var inverse: MLXArray?
        if indices.size >= 64 {
            (inputs, selected, inverse) = gatherSort(x: inputs, indices: indices)
        }
        let pair = gateUp(inputs, selected, sorted: inverse != nil)
        let gate = pair[.ellipsis, .stride(by: 2)].asType(.float32)
        let up = pair[.ellipsis, .stride(from: 1, by: 2)].asType(.float32)
        let cappedGate = tanh(gate / c.gateCap) * c.gateCap
        let cappedUp = tanh(up / c.upCap) * c.upCap
        let active = cappedGate * sigmoid(gate) * cappedUp
        var out = down(active.asType(pair.dtype), selected, sorted: inverse != nil)
        if let inverse { out = scatterUnsort(x: out, invOrder: inverse, shape: indices.shape) }
        return out.squeezed(axis: -2)
    }
}

final class Inkling2Router: Module {
    let weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray

    init(_ c: Inkling2Configuration.Text) {
        weight = .zeros([c.experts, c.hiddenSize])
        _correction.wrappedValue = .zeros([c.experts])
    }
}

final class Inkling2MoE: Module {
    @ModuleInfo var gate: Inkling2Router
    @ModuleInfo var experts: Inkling2Experts
    let c: Inkling2Configuration.Text

    init(_ c: Inkling2Configuration.Text) {
        self.c = c
        _gate.wrappedValue = Inkling2Router(c)
        _experts.wrappedValue = Inkling2Experts(c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scores = sigmoid(matmul(x.asType(.float32), gate.weight.T))
        let k = c.expertsPerToken
        let indices = argPartition(-(scores + gate.correction), kth: k - 1, axis: -1)[
            .ellipsis, ..<k]
        let selected = takeAlong(scores, indices, axis: -1)
        let routing = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20) * c.routeScale
        return (experts(x, indices).asType(.float32) * routing.expandedDimensions(axis: -1))
            .sum(axis: -2).asType(x.dtype)
    }
}
