import MLX

/// Per-layer key/value state threaded between Gemma 4 decoder layers when KV sharing
/// is enabled.
///
/// `public` rather than `package` because `Gemma4DecoderLayer.callAsFunction` is exposed
/// at `@_spi(GemmaEncoder)` scope for encoder-style client taps, and it appears in that
/// signature — an SPI method cannot be called from outside the package if a type in its
/// signature is package-scoped. Clients that do not use KV sharing pass `nil` and ignore
/// the returned value.
public enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    /// Cache-owned attention storage shared with later decoder layers.
    case native(cache: KVCache, sequenceLength: Int)

    package static func update(keys: MLXArray, values: MLXArray, cache: KVCache?) -> Self {
        switch AttentionKVState.update(keys: keys, values: values, cache: cache) {
        case .regular(let keys, let values):
            return .regular(keys: keys, values: values)
        case .quantized(let keys, let values, let groupSize, let bits, let mode):
            return .quantized(
                keys: keys, values: values, groupSize: groupSize, bits: bits, mode: mode)
        case .native(let cache, let sequenceLength):
            return .native(cache: cache, sequenceLength: sequenceLength)
        }
    }

    package func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let state: AttentionKVState
        switch self {
        case .regular(let keys, let values):
            state = .regular(keys: keys, values: values)
        case .quantized(let keys, let values, let groupSize, let bits, let mode):
            state = .quantized(
                keys: keys, values: values, groupSize: groupSize, bits: bits, mode: mode)
        case .native(let cache, let sequenceLength):
            guard let cache = cache as? any SharedAttentionKVCache else {
                preconditionFailure("Gemma shared KV requires cache-native attention support")
            }
            state = .native(cache: cache, sequenceLength: sequenceLength)
        }
        return state.attend(queries: queries, scale: scale, mask: mask)
    }

    public var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            keys.0.dim(-2)
        case .native(_, let sequenceLength):
            sequenceLength
        }
    }
}
