//
//  Gemma4MobileQuantization.swift
//  mlx-swift-lm
//
//  Pure-Swift port of the Gemma 4 QAT mobile (wNa8o8) quantization format
//  (`quant_method: "gemma"`) from mlx-vlm's `mlx_vlm/quantization/gemma_mobile.py`.
//
//  Weights stay packed (uint8/int8) in resident memory — the ~2.4 GB footprint
//  is the point of the format — and are unpacked transiently per layer inside
//  the matmul (dequantize-on-forward). Custom Metal qmv/qmm kernels are a
//  follow-on optimization (see GEMMA4_QAT_MOBILE_SWIFT_PORT_PLAN.md §9).
//
//  Format (verified against the official HuggingFace reference and the
//  `gemma-4-E2B-it-qat-mobile-mlx-mm` checkpoint):
//    - int2: 4 values/uint8, LSB-first, range [-2, 1]  (mask & 0x3 - 2)
//    - int4: 2 values/uint8, low-nibble-first, range [-8, 7] (mask & 0xF - 8)
//    - int8: direct int8, range [-128, 127]
//    - per-channel weight_scale [out, 1]: dequant = unpack(w) * weight_scale
//    - SRQ activations: scalar input/output_activation_scale (0 ⇒ uncalibrated)
//    - embeddings: packed table + per-row ([num_emb, 1]) or block-wise
//      ([num_emb, n_blocks]) embedding_scale

import Foundation
import MLX
import MLXNN

// MARK: - JSON insertion-order recovery

extension CodingUserInfoKey {
    /// Raw `config.json` data stashed by the model factory so that
    /// `GemmaMobileQuantizationConfig` can recover JSON key insertion order
    /// (which `JSONDecoder` / `JSONSerialization` do not preserve).
    public static let rawConfigData = CodingUserInfoKey(rawValue: "org.mlx.lm.rawConfigData")!
}

/// Re-escape a decoded key so it can be searched for literally in the raw JSON
/// text (JSON escapes `\` as `\\` and `"` as `\"`).
private func jsonKeyForSearch(_ key: String) -> String {
    var result = "\""
    for c in key {
        switch c {
        case "\\": result += "\\\\"
        case "\"": result += "\\\""
        default: result.append(c)
        }
    }
    return result + "\""
}

/// Extract the keys of the `module_quant_configs` JSON object in insertion
/// order by scanning the raw text.
///
/// Foundation's `JSONSerialization` and `JSONDecoder` return dictionary keys in
/// hash-table order, not JSON insertion order. The Python reference's
/// `resolve_module_bits` is first-match-wins over `module_quant_configs` in
/// insertion order, so we recover the true order by locating each key's
/// position in the raw JSON text (after the `"module_quant_configs"` marker)
/// and sorting by position. Returns `nil` if recovery is not possible, in
/// which case the caller falls back to an alphabetical sort.
private func orderedModuleQuantConfigKeys(from data: Data) -> [String]? {
    guard let text = String(data: data, encoding: .utf8),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }

    // `module_quant_configs` may be at the top level (when decoding a bare
    // `GemmaMobileQuantizationConfig`), inside `quantization_config` (gemma4),
    // or nested in `text_config.quantization_config` (gemma4_text).
    let mqc: [String: Any]?
    if let top = json["module_quant_configs"] as? [String: Any] {
        mqc = top
    } else if let qc = json["quantization_config"] as? [String: Any] {
        mqc = qc["module_quant_configs"] as? [String: Any]
    } else if let tc = json["text_config"] as? [String: Any],
        let qc = tc["quantization_config"] as? [String: Any]
    {
        mqc = qc["module_quant_configs"] as? [String: Any]
    } else {
        mqc = nil
    }
    guard let mqc, !mqc.isEmpty else { return nil }

    guard let marker = text.range(of: "\"module_quant_configs\"") else { return nil }
    let searchRange = marker.upperBound ..< text.endIndex

    var positioned: [(String, String.Index)] = []
    for key in mqc.keys {
        if let range = text.range(of: jsonKeyForSearch(key), range: searchRange) {
            positioned.append((key, range.lowerBound))
        }
    }
    // Every key must be locatable in the text; otherwise fall back.
    guard positioned.count == mqc.count else { return nil }
    positioned.sort { $0.1 < $1.1 }
    return positioned.map { $0.0 }
}

// MARK: - Bit-exact unpacking (matches gemma_quant._unpack_int2 / _unpack_int4)

/// Unpack int2 from uint8 (4 values/byte, LSB-first) → int8 in [-2, 1].
///
/// `packed` is `[..., packed_in]` uint8; result is `[..., in_features]` int8.
public func unpackInt2(_ packed: MLXArray, inFeatures: Int) -> MLXArray {
    precondition(packed.dtype == .uint8, "int2 weights must be uint8, got \(packed.dtype).")
    let v0 = (packed & 0x03).asType(.int8) - 2
    let v1 = ((packed >> 2) & 0x03).asType(.int8) - 2
    let v2 = ((packed >> 4) & 0x03).asType(.int8) - 2
    let v3 = (packed >> 6).asType(.int8) - 2
    let out = MLX.stacked([v0, v1, v2, v3], axis: -1)
    let reshaped = out.reshaped(Array(packed.shape.dropLast()) + [-1])
    return reshaped[.ellipsis, 0 ..< inFeatures]
}

/// Unpack int4 from uint8 (2 values/byte, low-nibble-first) → int8 in [-8, 7].
public func unpackInt4(_ packed: MLXArray, inFeatures: Int) -> MLXArray {
    precondition(packed.dtype == .uint8, "int4 weights must be uint8, got \(packed.dtype).")
    let low = (packed & 0x0F).asType(.int8) - 8
    let high = (packed >> 4).asType(.int8) - 8
    let out = MLX.stacked([low, high], axis: -1)
    let reshaped = out.reshaped(Array(packed.shape.dropLast()) + [-1])
    return reshaped[.ellipsis, 0 ..< inFeatures]
}

/// Dispatch unpacking on the bit width; returns signed int8 values.
public func unpackInt(_ packed: MLXArray, numBits: Int, inFeatures: Int) -> MLXArray {
    switch numBits {
    case 2: return unpackInt2(packed, inFeatures: inFeatures)
    case 4: return unpackInt4(packed, inFeatures: inFeatures)
    case 8:
        precondition(packed.dtype == .int8, "int8 weights must be int8, got \(packed.dtype).")
        return packed
    default:
        fatalError("Unsupported numBits \(numBits); expected 2, 4, or 8.")
    }
}

// MARK: - Static Range Quantization (SRQ) for activations

/// Static Range Quantization rounding/clipping (fake-quant to int8 and back).
///
/// `scale == 0` means the layer is uncalibrated ⇒ no-op. The guard uses
/// `MLX.where` (not a scalar `.item()`) so it stays on-device and compile-friendly.
public func applySRQ(_ x: MLXArray, scale: MLXArray, bits: Int = 8) -> MLXArray {
    let maxValue = (1 << (bits - 1)) - 1  // 127 for bits == 8
    let minValue = -maxValue - 1  // -128
    let scaleF = scale.asType(x.dtype)
    let calibrated = scaleF .!= 0
    let safeScale = MLX.where(calibrated, scaleF, MLXArray.ones(like: scaleF))
    let q = MLX.clip(MLX.round(x / safeScale), min: minValue, max: maxValue) * safeScale
    return MLX.where(calibrated, q, x)
}

/// Compiled SRQ in float32 for calibrated layers (scale != 0). Fuses the
/// divide, round, clip, and multiply into a single Metal kernel via
/// `MLX.compile`, cutting per-layer SRQ kernel launches from ~5 to 1. Used by
/// the fallback path (quantizedMM + SRQ) for large prefill (batch > 16) or
/// unaligned dims. `shapeless: true` lets the same compiled kernel handle both
/// prefill and decode.
///
/// All SRQ math is in float32 (matching `srqF32`, the Python `_srq`, and the
/// native compiled path). The previous bfloat16 SRQ rounded differently from
/// the float32 SRQ, causing significant divergence from the native compiled
/// path (the `quantizedMM` output differed by up to 4.4% relative, compounding
/// across 35 layers to a 1.16 mean abs diff in logits).
private enum CompiledSRQ {
    nonisolated(unsafe) static let apply: (MLXArray, MLXArray) -> MLXArray = MLX.compile(
        shapeless: true
    ) {
        (x, s) in
        let st = s.asType(.float32)
        return MLX.clip(MLX.round(x.asType(.float32) / st), min: -128, max: 127) * st
    }
}

// MARK: - Compile-friendly float32 SRQ (for native quantizedMM compiled segments)

/// Compile-friendly SRQ in float32 (matches Python `_srq` and the qmv kernel).
///
/// All SRQ math is done in float32 (`round` / `clip` / `multiply`), matching the
/// custom qmv kernel's internal precision. When `scale == 0` the SRQ is a no-op
/// (returns `x` unchanged). `scale` may be a scalar or per-row array.
///
/// Use this inside `compile` graphs (not `applySRQ`, which uses the input dtype).
/// The `compile` system fuses the `where` / `round` / `clip` / `multiply` with
/// adjacent norms and the `quantizedMM` into a single Metal graph.
public func srqF32(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
    let s32 = s.asType(.float32)
    let isZero = s32 .== 0
    let safe = MLX.where(isZero, MLXArray.ones(like: s32), s32)
    let q = MLX.clip(MLX.round(x.asType(.float32) / safe), min: -128, max: 127) * safe
    return MLX.where(isZero, x, q)
}

// MARK: - Weight / embedding dequantization

/// Coerce a per-channel scale to a broadcastable `[..., 1]` shape.
private func channelScale(_ weightScale: MLXArray) -> MLXArray {
    if weightScale.ndim >= 1 && weightScale.dim(-1) == 1 {
        return weightScale
    }
    if weightScale.ndim == 0 {
        return weightScale
    }
    return weightScale.reshaped(Array(weightScale.shape) + [1])
}

/// Dequantize a packed per-channel weight → real matrix `unpack(w) * scale`.
///
/// `weight` is `[out, packed_in]` (uint8/int8); result is `[out, in]`. When
/// `dtype` is given the result is cast to it (e.g. the activation dtype so the
/// matmul runs in that precision), matching the Python pure-MLX fallback.
public func dequantizeWeight(
    _ weight: MLXArray,
    weightScale: MLXArray,
    numBits: Int,
    inputDims: Int,
    dtype: DType? = nil
) -> MLXArray {
    let ints = unpackInt(weight, numBits: numBits, inFeatures: inputDims)
    var out = ints.asType(weightScale.dtype) * channelScale(weightScale)
    if let dtype { out = out.asType(dtype) }
    return out
}

/// Dequantize gathered embedding rows using a per-row or block-wise scale.
///
/// `rows` is `[..., packed_dim]` (uint8/int8) and `scales` is `[..., n_blocks]`.
/// Per-row scales (`n_blocks == 1`) broadcast over the embedding dim; block-wise
/// scales reshape the ints into `[..., n_blocks, block_size]` and broadcast the
/// per-block scale, then flatten back to `[..., embedding_dim]`.
public func dequantizeEmbeddingRows(
    _ rows: MLXArray,
    scales: MLXArray,
    numBits: Int,
    embeddingDim: Int,
    numBlocks: Int = 1
) -> MLXArray {
    let ints = unpackInt(rows, numBits: numBits, inFeatures: embeddingDim)
    let outDtype = scales.dtype
    if numBlocks == 1 || scales.dim(-1) == 1 {
        return ints.asType(outDtype) * scales.asType(outDtype)
    }
    let blockSize = embeddingDim / numBlocks
    let leadingShape = Array(ints.shape.dropLast())
    let intsR = ints.reshaped(leadingShape + [numBlocks, blockSize])
    let scaled = intsR.asType(outDtype) * scales[.ellipsis, .newAxis].asType(outDtype)
    return scaled.reshaped(leadingShape + [-1])
}

// MARK: - Mobile → MLX uint32 conversion (fused quantizedMM path)

/// Convert mobile packed weights to MLX's uint32 quantized format (group_size=128).
///
/// Returns `(packed, scales, biases)` for `quantizedMM`. The conversion is
/// bit-exact: `dequantized(packed, scales: scales, biases: biases, groupSize: 128,
/// bits: numBits)` equals `dequantizeWeight(weight, weightScale, numBits, inputDims)`.
///
/// Per-channel (one scale per output row) symmetric quantization maps to
/// group_size=128 with the per-channel scale broadcast across all groups in a
/// row and a constant bias of `-shift * scale` (where `shift` re-centers the
/// signed int range to unsigned).
///
/// Requires `inputDims % 128 == 0` (MLX's smallest supported group_size).
/// For int2/int4 the packed uint8 bytes already contain the shifted unsigned
/// values (LSB-first within each byte), so 4 consecutive bytes map directly to
/// 1 uint32 (little-endian) without unpacking.
public func mobileToMLX(
    weight: MLXArray,
    weightScale: MLXArray,
    numBits: Int,
    inputDims: Int,
    numBlocks: Int = 1
) -> (packed: MLXArray, scales: MLXArray, biases: MLXArray) {
    precondition(
        inputDims % 128 == 0,
        "inputDims must be divisible by 128 for group_size=128, got \(inputDims)")

    let shift: Int
    switch numBits {
    case 2: shift = 2
    case 4: shift = 8
    case 8: shift = 128
    default: fatalError("Unsupported numBits \(numBits); expected 2, 4, or 8.")
    }

    let nGroups = inputDims / 128
    let outDims = weight.shape[0]

    let packed: MLXArray
    if numBits == 2 || numBits == 4 {
        // Mobile uint8 bytes already hold the shifted unsigned values (LSB-first
        // within each byte). MLX packs the same values LSB-first into uint32, so
        // reinterpret the contiguous bytes instead of materializing an identical
        // second packed-weight allocation.
        let nUint32 = weight.shape[1] / 4
        packed = weight.view(dtype: .uint32).reshaped([outDims, nUint32])
    } else {
        // int8: shift signed to unsigned (+128), then pack 4 values per uint32.
        let q = (weight.asType(.int32) + MLXArray(shift, dtype: .int32)).asType(.uint32)
        let nUint32 = q.shape[1] / 4
        let reshaped = q.reshaped([outDims, nUint32, 4])
        let b0 = reshaped[.ellipsis, 0]
        let b1 = reshaped[.ellipsis, 1]
        let b2 = reshaped[.ellipsis, 2]
        let b3 = reshaped[.ellipsis, 3]
        packed = b0 + b1 * 256 + b2 * 65536 + b3 * 16_777_216
    }

    // Per-group scales/biases.
    let ws = weightScale.asType(.float32)
    let scales: MLXArray
    let biases: MLXArray
    if numBlocks > 1 && ws.ndim >= 2 && ws.dim(-1) > 1 {
        // Block-wise scales: [out, n_blocks] → broadcast each block to its groups.
        let groupsPerBlock = inputDims / (numBlocks * 128)
        precondition(
            inputDims % (numBlocks * 128) == 0,
            "inputDims must be divisible by numBlocks * 128 for block-wise scales")
        let wsR = ws.reshaped([outDims, numBlocks, 1])
        scales = broadcast(wsR, to: [outDims, numBlocks, groupsPerBlock])
            .reshaped([outDims, nGroups])
        let biasesR = (-Float(shift) * ws).reshaped([outDims, numBlocks, 1])
        biases = broadcast(biasesR, to: [outDims, numBlocks, groupsPerBlock])
            .reshaped([outDims, nGroups])
    } else {
        // Per-channel / per-row scales: [out, 1] or [out] or scalar.
        let wsC = channelScale(ws)
        scales = broadcast(wsC, to: [outDims, nGroups])
        biases = broadcast(-Float(shift) * wsC, to: [outDims, nGroups])
    }

    return (packed, scales, biases)
}

// MARK: - Configuration

/// `quantization_config` for the Gemma 4 QAT mobile format (`quant_method: "gemma"`).
///
/// `module_quant_configs` is preserved in JSON insertion order so that
/// `resolveModuleBits` can apply first-match-wins like the Python reference
/// (e.g. the layer-0–14 4-bit mlp pattern must be tried before the catch-all
/// 2-bit mlp pattern). Swift `Dictionary` does not preserve order, so this is
/// decoded into an ordered `[(pattern, numBits)]` list.
public struct GemmaMobileQuantizationConfig: Codable, Sendable {
    /// Ordered list of `(regex pattern, num_bits)` from `module_quant_configs`.
    public var moduleQuantConfigs: [(String, Int)] = []
    public var modulesToNotConvert: [String] = []
    public var numBits: Int = 4
    public var quantMethod: String = ""
    public var quantizeEmbeddings: Bool = false

    public var isGemmaMobile: Bool { quantMethod == "gemma" }

    /// A single `module_quant_configs` entry: `{"num_bits": N}` (or a bare int).
    private struct ModuleQuantConfig: Codable {
        var numBits: Int
        init(numBits: Int) { self.numBits = numBits }
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let dict = try? c.decode([String: Int].self), let nb = dict["num_bits"] {
                self.numBits = nb
            } else {
                self.numBits = try c.decode(Int.self)
            }
        }
    }

    /// Dynamic coding key for iterating arbitrary `module_quant_configs` keys.
    private struct AnyCodingKey: CodingKey {
        var stringValue: String
        var intValue: Int?
        init?(stringValue: String) {
            self.stringValue = stringValue
            self.intValue = nil
        }
        init?(intValue: Int) {
            self.stringValue = "\(intValue)"
            self.intValue = intValue
        }
    }

    enum CodingKeys: String, CodingKey {
        case moduleQuantConfigs = "module_quant_configs"
        case modulesToNotConvert = "modules_to_not_convert"
        case numBits = "num_bits"
        case quantMethod = "quant_method"
        case quantizeEmbeddings = "quantize_embeddings"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.quantMethod = try c.decodeIfPresent(String.self, forKey: .quantMethod) ?? ""
        self.numBits = try c.decodeIfPresent(Int.self, forKey: .numBits) ?? 4
        self.quantizeEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .quantizeEmbeddings) ?? false
        self.modulesToNotConvert =
            try c.decodeIfPresent([String].self, forKey: .modulesToNotConvert) ?? []

        // Decode module_quant_configs. JSONDecoder's KeyedDecodingContainer.allKeys
        // does not preserve JSON insertion order, but the Python reference's
        // `resolve_module_bits` is first-match-wins over the patterns in
        // insertion order. When the raw config data is available (stashed by
        // the model factory via `userInfo`), recover the true insertion order
        // by scanning the raw text; otherwise fall back to an alphabetical
        // sort (correct for the E2B/E4B mobile schemas, whose patterns happen
        // to be authored in alphabetical order).
        var ordered: [(String, Int)] = []
        if let nested = try? c.nestedContainer(
            keyedBy: AnyCodingKey.self, forKey: .moduleQuantConfigs)
        {
            for key in nested.allKeys {
                let cfg = try nested.decode(ModuleQuantConfig.self, forKey: key)
                ordered.append((key.stringValue, cfg.numBits))
            }
        }
        if let rawData = decoder.userInfo[.rawConfigData] as? Data,
            let insertionOrder = orderedModuleQuantConfigKeys(from: rawData)
        {
            let byName = Dictionary(uniqueKeysWithValues: ordered)
            ordered = insertionOrder.compactMap { name in
                byName[name].map { (name, $0) }
            }
        } else {
            ordered.sort { $0.0 < $1.0 }
        }
        self.moduleQuantConfigs = ordered
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(quantMethod, forKey: .quantMethod)
        try c.encode(numBits, forKey: .numBits)
        try c.encode(quantizeEmbeddings, forKey: .quantizeEmbeddings)
        try c.encode(modulesToNotConvert, forKey: .modulesToNotConvert)
        var nested = c.nestedContainer(keyedBy: AnyCodingKey.self, forKey: .moduleQuantConfigs)
        for (pattern, bits) in moduleQuantConfigs {
            try nested.encode(
                ModuleQuantConfig(numBits: bits), forKey: AnyCodingKey(stringValue: pattern)!)
        }
    }
}

// MARK: - Per-layer bit resolution (mirrors resolve_module_bits)

/// True if `segment` is a whole path segment of `path` (matches, starts, ends,
/// or is contained as a dotted segment).
private func pathContainsSegment(_ path: String, _ segment: String) -> Bool {
    path == segment
        || path.hasPrefix(segment + ".")
        || path.hasSuffix("." + segment)
        || path.contains("." + segment + ".")
}

/// Resolve the bit width for a leaf module path, or `nil` to skip it.
///
/// `nil` means the module is in `modules_to_not_convert`. Unmatched modules
/// fall back to the config's default `num_bits` (4 for the mobile schema).
///
/// Path normalization maps both the `gemma4` wrapper namespace
/// (`language_model.model.X` / `language_model.lm_head`) and the text-only
/// `gemma4_text` namespace (`model.X` / `lm_head`) onto the HuggingFace
/// `module_quant_configs` regex namespace (`language_model.X` / `lm_head`).
public func resolveModuleBits(path: String, config: GemmaMobileQuantizationConfig) -> Int? {
    var normalized = path
    if normalized.hasPrefix("language_model.model.") {
        normalized = "language_model." + normalized.dropFirst("language_model.model.".count)
    } else if normalized.hasPrefix("model.") {
        normalized = "language_model." + normalized.dropFirst("model.".count)
    }
    if normalized == "language_model.lm_head" {
        normalized = "lm_head"
    }

    for entry in config.modulesToNotConvert {
        let e = entry.hasPrefix("model.") ? String(entry.dropFirst("model.".count)) : entry
        if !e.isEmpty && pathContainsSegment(normalized, e) {
            return nil
        }
    }

    let range = NSRange(location: 0, length: normalized.utf16.count)
    for (pattern, bits) in config.moduleQuantConfigs {
        if let regex = try? NSRegularExpression(pattern: pattern),
            regex.firstMatch(in: normalized, range: range) != nil
        {
            return bits
        }
    }
    return config.numBits
}

// MARK: - Fused qmv Metal kernel (decode / small-batch)

/// Two simdgroups (64 lanes) per threadgroup; each simdgroup computes
/// `OUTPUTS_PER_SIMDGROUP` output rows, sharing the activation reads.
private let gemmaQMVOutputsPerSimdgroup = 4
private let gemmaQMVOutputsPerThreadgroup = 8  // 2 simdgroups × 4

/// Metal source for the fused qmv (quantized matrix-vector) kernel.
///
/// Reads packed uint8/int8 weights + per-channel scale directly (never
/// materializing the full fp weight) and fuses SRQ (input + output) into the
/// kernel — eliminating separate SRQ kernel launches. Ported from the
/// Python `_GEMMA_QMV_SOURCE` in gemma_mobile.py.
///
/// Placeholders (substituted at build time):
/// - `VALUES_PER_BYTE` — 4 (int2), 2 (int4), 1 (int8)
/// - `OUTPUTS_PER_SIMDGROUP` — 4
/// - `OUTPUTS_PER_THREADGROUP` — 8
/// - `UNPACK_BLOCK` — bit-width-specific unpack + MAC snippet
private let gemmaQMVSourceTemplate = """
        uint lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint input_row = threadgroup_position_in_grid.z;
        uint input_dims = x_shape[1];
        uint output_dims = weight_shape[0];
        uint packed_in = weight_shape[1];
        uint output_start = threadgroup_position_in_grid.y * OUTPUTS_PER_THREADGROUP
            + simd_group * OUTPUTS_PER_SIMDGROUP;

        // Input SRQ scale is scalar (shared across all output rows); 0.0 means
        // uncalibrated → no-op. Output SRQ scale is scalar for standalone layers.
        float in_s = static_cast<float>(input_scale[0]);

        float accumulators[OUTPUTS_PER_SIMDGROUP] = {0.0f};
        constexpr uint VALUES_PER_THREAD = 16;
        constexpr uint BYTES_PER_THREAD = VALUES_PER_THREAD / VALUES_PER_BYTE;
        constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;

        for (uint block_start = lane * VALUES_PER_THREAD;
             block_start < input_dims;
             block_start += BLOCK_SIZE) {
            // Read + input-SRQ x values ONCE, shared across OUTPUTS_PER_SIMDGROUP rows.
            float x_thread[VALUES_PER_THREAD];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
                float x_val = static_cast<float>(
                    x[input_row * input_dims + block_start + i]);
                if (in_s != 0.0f) {
                    x_val = clamp(round(x_val / in_s), -128.0f, 127.0f) * in_s;
                }
                x_thread[i] = x_val;
            }
            uint packed_start = block_start / VALUES_PER_BYTE;
            for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; ++row) {
                uint output_row = output_start + row;
                if (output_row >= output_dims) break;
                float row_sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint b = 0; b < BYTES_PER_THREAD; ++b) {
                    UNPACK_BLOCK
                }
                accumulators[row] += row_sum;
            }
        }

        for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; ++row) {
            accumulators[row] = simd_sum(accumulators[row]);
            uint output_row = output_start + row;
            if (lane == 0 && output_row < output_dims) {
                float result = accumulators[row] * static_cast<float>(weight_scale[output_row]);
                float out_s = static_cast<float>(output_scale[0]);
                if (out_s != 0.0f) {
                    result = clamp(round(result / out_s), -128.0f, 127.0f) * out_s;
                }
                out[input_row * output_dims + output_row] = static_cast<T>(result);
            }
        }
    """

/// Bit-width-specific Metal snippet that unpacks BYTES_PER_THREAD packed bytes
/// and MACs into `row_sum`. Ported from Python `_gemma_unpack_block`.
private func gemmaQMVUnpackBlock(numBits: Int) -> String {
    switch numBits {
    case 2:
        return """
                        uint byte = uint(weight[output_row * packed_in + packed_start + b]);
                        row_sum += (float(byte & 0x3) - 2.0f) * x_thread[b * 4 + 0]
                                 + (float((byte >> 2) & 0x3) - 2.0f) * x_thread[b * 4 + 1]
                                 + (float((byte >> 4) & 0x3) - 2.0f) * x_thread[b * 4 + 2]
                                 + (float(byte >> 6) - 2.0f) * x_thread[b * 4 + 3];
            """
    case 4:
        return """
                        uint byte = uint(weight[output_row * packed_in + packed_start + b]);
                        row_sum += (float(byte & 0xF) - 8.0f) * x_thread[b * 2 + 0]
                                 + (float(byte >> 4) - 8.0f) * x_thread[b * 2 + 1];
            """
    case 8:
        return """
                        int v = int(weight[output_row * packed_in + packed_start + b]);
                        row_sum += float(v) * x_thread[b];
            """
    default:
        fatalError("Unsupported numBits \(numBits); expected 2, 4, or 8.")
    }
}

/// Build the qmv Metal kernel source for a given bit width by substituting the
/// placeholders in the template.
private func gemmaQMVSource(numBits: Int) -> String {
    let valuesPerByte: Int
    switch numBits {
    case 2: valuesPerByte = 4
    case 4: valuesPerByte = 2
    case 8: valuesPerByte = 1
    default: fatalError("Unsupported numBits \(numBits); expected 2, 4, or 8.")
    }
    return
        gemmaQMVSourceTemplate
        .replacingOccurrences(of: "VALUES_PER_BYTE", with: "\(valuesPerByte)")
        .replacingOccurrences(of: "OUTPUTS_PER_SIMDGROUP", with: "\(gemmaQMVOutputsPerSimdgroup)")
        .replacingOccurrences(
            of: "OUTPUTS_PER_THREADGROUP", with: "\(gemmaQMVOutputsPerThreadgroup)"
        )
        .replacingOccurrences(of: "UNPACK_BLOCK", with: gemmaQMVUnpackBlock(numBits: numBits))
}

/// Cache of compiled qmv kernels keyed by bit width. Metal kernel compilation
/// is expensive, so each specialization is compiled once and reused.
private enum GemmaQMVKernelCache {
    nonisolated(unsafe) static var cache: [Int: MLXFast.MLXFastKernel] = [:]

    static func kernel(numBits: Int) -> MLXFast.MLXFastKernel {
        if let cached = cache[numBits] { return cached }
        let kernel = MLXFast.metalKernel(
            name: "gemma_mobile_qmv_b\(numBits)",
            inputNames: ["x", "weight", "weight_scale", "input_scale", "output_scale"],
            outputNames: ["out"],
            source: gemmaQMVSource(numBits: numBits),
            header: "#include <metal_simdgroup>\nusing namespace metal;")
        cache[numBits] = kernel
        return kernel
    }
}

// MARK: - Fused q/k/v projection (plan §9.3)

/// True when the three projections' input SRQ scales are identical (so they
/// can share a single fused matmul). Mirrors Python `_qkv_input_scales_match`.
public func gemmaQKVInputScalesMatch(
    _ q: GemmaQuantizedLinear, _ k: GemmaQuantizedLinear, _ v: GemmaQuantizedLinear
) -> Bool {
    // Input SRQ scales are scalars; compare as float32 (bfloat16 → float32 is
    // lossless). Mirrors Python `array_equal` (exact, not approximate).
    let qS = q.inputActivationScale.asType(.float32).item(Float.self)
    let kS = k.inputActivationScale.asType(.float32).item(Float.self)
    let vS = v.inputActivationScale.asType(.float32).item(Float.self)
    return qS == kS && qS == vS
}

/// Build the per-row [q_out + k_out + v_out] output SRQ scale used by the
/// native compiled path, whose quantizedMM arguments are concatenated.
public func gemmaBuildPerRowOutputScale(
    _ q: GemmaQuantizedLinear, _ k: GemmaQuantizedLinear, _ v: GemmaQuantizedLinear,
    dtype: DType
) -> MLXArray {
    let parts = [q, k, v].map { projection in
        broadcast(
            projection.outputActivationScale.asType(dtype),
            to: [projection.weight.shape[0]])
    }
    return concatenated(parts, axis: 0)
}

/// Metal source for fused q/k/v projection without concatenating the three
/// packed weight tensors. The projection choice is uniform for an output row,
/// so each simdgroup reads directly from one of the original buffers while
/// retaining a single kernel launch.
private let gemmaFusedQKVSourceTemplate = """
        uint lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint input_row = threadgroup_position_in_grid.z;
        uint input_dims = x_shape[1];
        uint q_output_dims = q_weight_shape[0];
        uint k_output_dims = k_weight_shape[0];
        uint v_output_dims = v_weight_shape[0];
        uint output_dims = q_output_dims + k_output_dims + v_output_dims;
        uint packed_in = q_weight_shape[1];
        uint output_start = threadgroup_position_in_grid.y * OUTPUTS_PER_THREADGROUP
            + simd_group * OUTPUTS_PER_SIMDGROUP;

        float in_s = static_cast<float>(input_scale[0]);
        float accumulators[OUTPUTS_PER_SIMDGROUP] = {0.0f};
        constexpr uint VALUES_PER_THREAD = 16;
        constexpr uint BYTES_PER_THREAD = VALUES_PER_THREAD / VALUES_PER_BYTE;
        constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;

        for (uint block_start = lane * VALUES_PER_THREAD;
             block_start < input_dims;
             block_start += BLOCK_SIZE) {
            float x_thread[VALUES_PER_THREAD];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
                float x_val = static_cast<float>(
                    x[input_row * input_dims + block_start + i]);
                if (in_s != 0.0f) {
                    x_val = clamp(round(x_val / in_s), -128.0f, 127.0f) * in_s;
                }
                x_thread[i] = x_val;
            }
            uint packed_start = block_start / VALUES_PER_BYTE;
            for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; ++row) {
                uint output_row = output_start + row;
                if (output_row >= output_dims) break;

                uint projection_row = output_row;
                auto selected_weight = q_weight;
                if (output_row >= q_output_dims + k_output_dims) {
                    projection_row -= q_output_dims + k_output_dims;
                    selected_weight = v_weight;
                } else if (output_row >= q_output_dims) {
                    projection_row -= q_output_dims;
                    selected_weight = k_weight;
                }

                float row_sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint b = 0; b < BYTES_PER_THREAD; ++b) {
                    UNPACK_BLOCK
                }
                accumulators[row] += row_sum;
            }
        }

        for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; ++row) {
            accumulators[row] = simd_sum(accumulators[row]);
            uint output_row = output_start + row;
            if (lane == 0 && output_row < output_dims) {
                uint projection_row = output_row;
                float weight_s;
                float out_s;
                if (output_row >= q_output_dims + k_output_dims) {
                    projection_row -= q_output_dims + k_output_dims;
                    weight_s = static_cast<float>(v_weight_scale[projection_row]);
                    out_s = static_cast<float>(v_output_scale[0]);
                } else if (output_row >= q_output_dims) {
                    projection_row -= q_output_dims;
                    weight_s = static_cast<float>(k_weight_scale[projection_row]);
                    out_s = static_cast<float>(k_output_scale[0]);
                } else {
                    weight_s = static_cast<float>(q_weight_scale[projection_row]);
                    out_s = static_cast<float>(q_output_scale[0]);
                }

                float result = accumulators[row] * weight_s;
                if (out_s != 0.0f) {
                    result = clamp(round(result / out_s), -128.0f, 127.0f) * out_s;
                }
                out[input_row * output_dims + output_row] = static_cast<T>(result);
            }
        }
    """

private func gemmaFusedQKVSource(numBits: Int) -> String {
    let valuesPerByte: Int
    switch numBits {
    case 2: valuesPerByte = 4
    case 4: valuesPerByte = 2
    case 8: valuesPerByte = 1
    default: fatalError("Unsupported numBits \(numBits); expected 2, 4, or 8.")
    }
    let unpack = gemmaQMVUnpackBlock(numBits: numBits)
        .replacingOccurrences(of: "weight[", with: "selected_weight[")
        .replacingOccurrences(of: "output_row", with: "projection_row")
    return
        gemmaFusedQKVSourceTemplate
        .replacingOccurrences(of: "VALUES_PER_BYTE", with: "\(valuesPerByte)")
        .replacingOccurrences(of: "OUTPUTS_PER_SIMDGROUP", with: "\(gemmaQMVOutputsPerSimdgroup)")
        .replacingOccurrences(
            of: "OUTPUTS_PER_THREADGROUP", with: "\(gemmaQMVOutputsPerThreadgroup)"
        )
        .replacingOccurrences(of: "UNPACK_BLOCK", with: unpack)
}

private enum GemmaFusedQKVKernelCache {
    nonisolated(unsafe) static var cache: [Int: MLXFast.MLXFastKernel] = [:]

    static func kernel(numBits: Int) -> MLXFast.MLXFastKernel {
        if let cached = cache[numBits] { return cached }
        let kernel = MLXFast.metalKernel(
            name: "gemma_mobile_qkv_split_b\(numBits)",
            inputNames: [
                "x", "q_weight", "k_weight", "v_weight",
                "q_weight_scale", "k_weight_scale", "v_weight_scale",
                "input_scale", "q_output_scale", "k_output_scale", "v_output_scale",
            ],
            outputNames: ["out"],
            source: gemmaFusedQKVSource(numBits: numBits),
            header: "#include <metal_simdgroup>\nusing namespace metal;")
        cache[numBits] = kernel
        return kernel
    }
}

/// Fused q/k/v quantized matrix-vector multiplication over the original packed
/// projection tensors. This keeps the single-launch decode optimization without
/// retaining concatenated copies of q/k/v weights or scales.
public func gemmaFusedQKVMatmul(
    x: MLXArray,
    q: GemmaQuantizedLinear,
    k: GemmaQuantizedLinear,
    v: GemmaQuantizedLinear
) -> MLXArray {
    precondition(q.numBits == k.numBits && q.numBits == v.numBits)
    precondition(q.inputDims == k.inputDims && q.inputDims == v.inputDims)

    let kernel = GemmaFusedQKVKernelCache.kernel(numBits: q.numBits)
    let totalOutDims = q.weight.shape[0] + k.weight.shape[0] + v.weight.shape[0]
    let outputShape = Array(x.shape.dropLast()) + [totalOutDims]
    let x2d = x.reshaped([-1, q.inputDims])
    let batch = x2d.shape[0]

    let out = kernel(
        [
            x2d, q.weight, k.weight, v.weight,
            q.weightScale, k.weightScale, v.weightScale,
            q.inputActivationScale.reshaped([1]).asType(x.dtype),
            q.outputActivationScale.reshaped([1]).asType(x.dtype),
            k.outputActivationScale.reshaped([1]).asType(x.dtype),
            v.outputActivationScale.reshaped([1]).asType(x.dtype),
        ],
        template: [("T", x.dtype)],
        grid: (
            64,
            (totalOutDims + gemmaQMVOutputsPerThreadgroup - 1)
                / gemmaQMVOutputsPerThreadgroup,
            batch
        ),
        threadGroup: (64, 1, 1),
        outputShapes: [[batch * totalOutDims]],
        outputDTypes: [x.dtype]
    )[0]

    return out.reshaped(outputShape)
}

// MARK: - Quantized layers

/// Linear with packed int2/4/8 per-channel weights and SRQ activations.
///
/// Subclasses `Linear` so it drops into existing `@ModuleInfo var …: Linear`
/// properties. The packed `weight` (uint8/int8) is inherited from `Linear` and
/// discovered by reflection under the key `"weight"`; the scales use
/// `@ParameterInfo` with explicit snake_case keys.
public final class GemmaQuantizedLinear: Linear {
    public let numBits: Int
    public let inputDims: Int

    @ParameterInfo(key: "weight_scale") public var weightScale: MLXArray
    @ParameterInfo(key: "input_activation_scale") public var inputActivationScale: MLXArray
    @ParameterInfo(key: "output_activation_scale") public var outputActivationScale: MLXArray

    // Converted MLX uint32 format (lazily computed on first forward pass so the
    // fused quantizedMM Metal kernel can be used instead of dequant-on-forward).
    // Prefixed with `_` so MLX's Module reflection does not discover them as
    // parameters (parameterIsValid filters keys starting with `_`).
    private var _mlxWeight: MLXArray?
    private var _mlxScales: MLXArray?
    private var _mlxBiases: MLXArray?
    private var _conversionDone = false
    private var _needsInputSRQ = true
    private var _needsOutputSRQ = true

    public init(inputDims: Int, outputDims: Int, numBits: Int, bias: Bool = false) {
        let packedIn: Int
        let wDtype: DType
        switch numBits {
        case 2:
            packedIn = (inputDims + 3) / 4
            wDtype = .uint8
        case 4:
            packedIn = (inputDims + 1) / 2
            wDtype = .uint8
        case 8:
            packedIn = inputDims
            wDtype = .int8
        default: fatalError("Unsupported numBits \(numBits); expected 2, 4, or 8.")
        }
        self.numBits = numBits
        self.inputDims = inputDims
        super.init(
            weight: MLXArray.zeros([outputDims, packedIn], dtype: wDtype),
            bias: bias ? MLXArray.zeros([outputDims]) : nil)
        self._weightScale.wrappedValue = MLXArray.ones([outputDims, 1])
        self._inputActivationScale.wrappedValue = MLXArray.zeros([])
        self._outputActivationScale.wrappedValue = MLXArray.zeros([])
        self.freeze()
    }

    /// Lazily convert packed mobile weights to MLX uint32 format on the first
    /// forward pass, and cache whether SRQ scales are non-zero (skip the no-op
    /// `applySRQ` graph construction entirely for uncalibrated layers).
    private func convertToMLXFormat() {
        // SRQ scales are scalar (shape []); read the value to decide whether to
        // skip the SRQ pass entirely. Only lm_head has zero scales (uncalibrated).
        let inScale = inputActivationScale.asType(.float32).item(Float.self)
        let outScale = outputActivationScale.asType(.float32).item(Float.self)
        _needsInputSRQ = inScale != 0
        _needsOutputSRQ = outScale != 0

        // Convert to uint32 format for the fused quantizedMM kernel. Falls back
        // to dequant-on-forward if inputDims is not aligned to group_size=128.
        if inputDims % 128 == 0 {
            let (packed, scales, biases) = mobileToMLX(
                weight: weight, weightScale: weightScale,
                numBits: numBits, inputDims: inputDims)
            eval([packed, scales, biases])
            _mlxWeight = packed
            _mlxScales = scales
            _mlxBiases = biases
        }
    }

    /// Returns `(packed, scales, biases, inScale, outScale)` for `quantizedMM`,
    /// converting lazily on first call. Returns `nil` when `inputDims` is not
    /// divisible by 128 (the MLX group_size constraint) — the caller should fall
    /// back to the eager / qmv path. Mirrors Python `_qlinear_native_args`.
    ///
    /// The SRQ scales (`inputActivationScale`, `outputActivationScale`) are always
    /// returned (even when zero / uncalibrated) — the compiled segment's `srqF32`
    /// handles the zero-scale no-op.
    public func nativeArgs()
        -> (
            packed: MLXArray, scales: MLXArray, biases: MLXArray, inScale: MLXArray,
            outScale: MLXArray
        )?
    {
        if !_conversionDone {
            convertToMLXFormat()
            _conversionDone = true
        }
        guard let packed = _mlxWeight, let scales = _mlxScales, let biases = _mlxBiases
        else {
            return nil  // inputDims not divisible by 128
        }
        return (packed, scales, biases, inputActivationScale, outputActivationScale)
    }

    /// Free the mobile-format weights after native conversion.
    ///
    /// Replaces the packed mobile `weight` and per-channel `weightScale` with
    /// tiny dummy arrays to recover memory (the native compiled path uses the
    /// converted `_mlxWeight`/`_mlxScales`/`_mlxBiases`, not the mobile weights).
    /// The SRQ scales (`inputActivationScale`, `outputActivationScale`) are
    /// preserved because the native path still reads them.
    ///
    /// No-op if native conversion has not succeeded (`nativeArgs()` returned
    /// `nil`), so the eager / qmv fallback path keeps its mobile weights. Mirrors
    /// Python `_free_mobile_weights` (the per-module part).
    public func freeMobileWeights() {
        guard _conversionDone, _mlxWeight != nil else { return }
        let wDtype = weight.dtype
        let sDtype = weightScale.dtype
        weight._updateInternal(MLXArray.zeros([1], dtype: wDtype))
        weightScale._updateInternal(MLXArray.zeros([1], dtype: sDtype))
    }

    /// Whether the fused qmv kernel can be used for this layer and batch size.
    /// Requires batch ≤ 16 and aligned input dims (÷512 for int2/int4, ÷16 for int8).
    public func canUseQMV(batchSize: Int) -> Bool {
        guard batchSize <= 16 else { return false }
        if numBits == 8 { return inputDims % 16 == 0 }
        return inputDims % 512 == 0  // int2, int4
    }

    /// Fused qmv matmul: reads packed weights + per-channel scale directly,
    /// fusing SRQ into the kernel. No dequant, no separate SRQ kernel launches.
    private func qmvCall(_ x: MLXArray) -> MLXArray {
        let kernel = GemmaQMVKernelCache.kernel(numBits: numBits)
        let outputDims = weight.shape[0]
        let outputShape = Array(x.shape.dropLast()) + [outputDims]
        let x2d = x.reshaped([-1, inputDims])
        let batch = x2d.shape[0]

        // SRQ scales are scalar (shape []); reshape to [1] for the kernel.
        let inS = inputActivationScale.reshaped([1]).asType(x.dtype)
        let outS = outputActivationScale.reshaped([1]).asType(x.dtype)

        let out = kernel(
            [x2d, weight, weightScale, inS, outS],
            template: [("T", x.dtype)],
            grid: (
                64,
                (outputDims + gemmaQMVOutputsPerThreadgroup - 1) / gemmaQMVOutputsPerThreadgroup,
                batch
            ),
            threadGroup: (64, 1, 1),
            outputShapes: [[batch * outputDims]],
            outputDTypes: [x.dtype]
        )[0]

        return out.reshaped(outputShape)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.shape.dropLast().reduce(1, *)

        // Fast path: fused qmv kernel for decode/small-batch with aligned dims.
        // Reads packed uint8/int8 weights + per-channel scale directly, fusing
        // SRQ into the kernel — no dequant, no separate SRQ kernel launches.
        let matmulOut: MLXArray
        let usedQMV: Bool
        if canUseQMV(batchSize: batchSize) {
            matmulOut = qmvCall(x)
            usedQMV = true
        } else {
            // Fallback: quantizedMM + compiled SRQ for prefill (batch > 16) or
            // unaligned dims. SRQ + matmul are done in float32 to match the native
            // compiled path (srqF32 + quantizedMM) and Python `_srq`. The bfloat16
            // SRQ used previously rounded differently from the float32 SRQ, causing
            // significant divergence from the native path (up to 4.4% relative
            // error per matmul, compounding across 35 layers).
            if !_conversionDone {
                convertToMLXFormat()
                _conversionDone = true
            }

            // Input SRQ in float32 (skip entirely if uncalibrated). Uses the
            // compiled (fused) SRQ kernel for calibrated layers.
            var xi = x.asType(.float32)
            if _needsInputSRQ {
                xi = CompiledSRQ.apply(xi, inputActivationScale)
            }

            // Fused quantized matmul (Metal kernel) when conversion succeeded;
            // otherwise fall back to dequant-on-forward.
            if let mw = _mlxWeight {
                matmulOut = quantizedMM(
                    xi, mw, scales: _mlxScales!, biases: _mlxBiases!,
                    transpose: true, groupSize: 128, bits: numBits, mode: .affine)
            } else {
                let w = dequantizeWeight(
                    weight, weightScale: weightScale, numBits: numBits,
                    inputDims: inputDims, dtype: .float32)
                matmulOut = matmul(xi, w.T)
            }
            usedQMV = false
        }

        // Output SRQ in float32 (matches native compiled path). For the qmv
        // path, output SRQ is already fused into the kernel, so skip.
        var out = matmulOut
        if !usedQMV && _needsOutputSRQ {
            out = CompiledSRQ.apply(out, outputActivationScale)
        }
        // Cast back to the original input dtype to preserve the model's dtype
        // flow (the native path does .asType(x.dtype) after the output SRQ).
        if !usedQMV {
            out = out.asType(x.dtype)
        }

        if let bias { out = out + bias }

        return out
    }

    /// Tolerate missing SRQ scales (uncalibrated layers) under `verify: .all`:
    /// the zero placeholder already makes `applySRQ` a no-op.
    public override func updateMissing(
        parameter: String, verify: Module.VerifyUpdate, path: [String], modulePath: [String]
    ) throws {
        if parameter == "input_activation_scale" || parameter == "output_activation_scale" {
            return
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }
}

/// Embedding with a packed int2/4/8 table and per-row (or block-wise) scale.
///
/// The architectural `embed_scale` is applied by the surrounding model (as in
/// the Gemma 4 text model), so this layer returns the *unscaled* dequantized
/// rows. Subclasses `Embedding` so it drops into `@ModuleInfo var …: Embedding`.
public final class GemmaQuantizedEmbedding: Embedding {
    public let numBits: Int
    public let embeddingDim: Int
    public let numBlocks: Int

    @ParameterInfo(key: "embedding_scale") public var embeddingScale: MLXArray

    public init(numEmbeddings: Int, embeddingDim: Int, numBits: Int, numBlocks: Int = 1) {
        let packedDim: Int
        let wDtype: DType
        switch numBits {
        case 2:
            packedDim = (embeddingDim + 3) / 4
            wDtype = .uint8
        case 4:
            packedDim = (embeddingDim + 1) / 2
            wDtype = .uint8
        case 8:
            packedDim = embeddingDim
            wDtype = .int8
        default: fatalError("Unsupported numBits \(numBits); expected 2, 4, or 8.")
        }
        self.numBits = numBits
        self.embeddingDim = embeddingDim
        self.numBlocks = numBlocks
        super.init(weight: MLXArray.zeros([numEmbeddings, packedDim], dtype: wDtype))
        self._embeddingScale.wrappedValue = MLXArray.ones([numEmbeddings, numBlocks])
        self.freeze()
    }

    // Embeddings use dequantize-on-forward (gather + unpack + scale) rather than
    // mobileToMLX conversion. The conversion would create a 4× uint32 copy of the
    // full table (~4.5 GB for embed_tokens_per_layer at 262144×8960 int4), causing
    // a large memory spike on the first prompt. Dequant-on-forward only gathers
    // and unpacks the requested rows (batch ≤ seq_len), which is tiny by
    // comparison and fast enough since embeddings are looked up per-token.

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rows = weight[x]
        let scales = embeddingScale[x]
        return dequantizeEmbeddingRows(
            rows, scales: scales, numBits: numBits, embeddingDim: embeddingDim,
            numBlocks: numBlocks)
    }

    public override func asLinear(_ x: MLXArray) -> MLXArray {
        let w = dequantizeEmbeddingRows(
            weight, scales: embeddingScale, numBits: numBits, embeddingDim: embeddingDim,
            numBlocks: numBlocks)
        return matmul(x, w.T)
    }

    /// Tolerate a missing embedding_scale under `verify: .all` (the ones
    /// placeholder would otherwise leave the table unquantized-looking).
    public override func updateMissing(
        parameter: String, verify: Module.VerifyUpdate, path: [String], modulePath: [String]
    ) throws {
        if parameter == "embedding_scale" {
            return
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }
}

// MARK: - Module replacement (mirrors replace_with_gemma_quant_layers)

/// Replace `Linear`/`Embedding` leaves with their Gemma quantized counterparts,
/// driven by `quantization_config.module_quant_configs`.
///
/// A layer is replaced only when (a) it is not in `modules_to_not_convert`, and
/// (b) the checkpoint actually carries a packed scale for it
/// (`<path>.weight_scale` for linears, `<path>.embedding_scale` for embeddings),
/// so unquantized / skipped modules (e.g. `per_layer_model_projection`) stay as
/// their original fp layer. `model` is the module whose `leafModules()` paths
/// match the (post-sanitize) weight keys — the top-level model for the loaded
/// checkpoint.
public func replaceWithGemmaQuantLayers(
    model: Module,
    quantizationConfig: GemmaMobileQuantizationConfig,
    weights: [String: MLXArray]
) {
    let updates = model.leafModules().flattened().compactMap { (path, m) -> (String, Module)? in
        // Skip already-quantized / already-replaced modules.
        if m is Quantized || m is GemmaQuantizedLinear || m is GemmaQuantizedEmbedding {
            return nil
        }
        guard let bits = resolveModuleBits(path: path, config: quantizationConfig) else {
            return nil  // in modules_to_not_convert
        }

        if let linear = m as? Linear {
            guard weights["\(path).weight_scale"] != nil else { return nil }
            let outDims = linear.shape.0
            let inDims = linear.shape.1
            let hasBias = linear.bias != nil
            return (
                path,
                GemmaQuantizedLinear(
                    inputDims: inDims, outputDims: outDims, numBits: bits, bias: hasBias)
            )
        }

        if let embedding = m as? Embedding {
            guard quantizationConfig.quantizeEmbeddings else { return nil }
            guard let scaleArr = weights["\(path).embedding_scale"] else { return nil }
            let (numEmb, dim) = embedding.shape
            let numBlocks = scaleArr.dim(-1)
            return (
                path,
                GemmaQuantizedEmbedding(
                    numEmbeddings: numEmb, embeddingDim: dim, numBits: bits, numBlocks: numBlocks)
            )
        }

        return nil
    }

    model.update(modules: ModuleChildren.unflattened(updates))
}

// MARK: - Sanitize hook

/// Remap `*.embedding_quantized` → `*.weight` (so the packed table loads into the
/// inherited `Embedding.weight`) and replace quantizable `Linear`/`Embedding`
/// leaves with their Gemma quantized counterparts.
///
/// `model` is the module whose `leafModules()` paths match the (post-sanitize)
/// weight keys — the top-level model for the loaded checkpoint (`Gemma4Model`
/// for `gemma4`, `Gemma4TextModel` for `gemma4_text`). Returns the remapped
/// weights for `loadWeights` to apply via `update(parameters:)`.
public func applyGemmaMobileQuantization(
    model: Module,
    weights: [String: MLXArray],
    config: GemmaMobileQuantizationConfig
) -> [String: MLXArray] {
    let suffix = ".embedding_quantized"
    var sanitized = [String: MLXArray]()
    for (key, value) in weights {
        if key.hasSuffix(suffix) {
            sanitized[String(key.dropLast(suffix.count)) + ".weight"] = value
        } else {
            sanitized[key] = value
        }
    }
    replaceWithGemmaQuantLayers(model: model, quantizationConfig: config, weights: sanitized)
    return sanitized
}
