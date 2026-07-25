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
    let searchRange = marker.upperBound..<text.endIndex

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
        init?(stringValue: String) { self.stringValue = stringValue; self.intValue = nil }
        init?(intValue: Int) { self.stringValue = "\(intValue)"; self.intValue = intValue }
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
            try nested.encode(ModuleQuantConfig(numBits: bits), forKey: AnyCodingKey(stringValue: pattern)!)
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

    public init(inputDims: Int, outputDims: Int, numBits: Int, bias: Bool = false) {
        let packedIn: Int
        let wDtype: DType
        switch numBits {
        case 2: packedIn = (inputDims + 3) / 4; wDtype = .uint8
        case 4: packedIn = (inputDims + 1) / 2; wDtype = .uint8
        case 8: packedIn = inputDims; wDtype = .int8
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

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xi = applySRQ(x, scale: inputActivationScale)
        let w = dequantizeWeight(
            weight, weightScale: weightScale, numBits: numBits, inputDims: inputDims,
            dtype: xi.dtype)
        var out = matmul(xi, w.T)
        out = applySRQ(out, scale: outputActivationScale)
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
        try super.updateMissing(parameter: parameter, verify: verify, path: path, modulePath: modulePath)
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
        case 2: packedDim = (embeddingDim + 3) / 4; wDtype = .uint8
        case 4: packedDim = (embeddingDim + 1) / 2; wDtype = .uint8
        case 8: packedDim = embeddingDim; wDtype = .int8
        default: fatalError("Unsupported numBits \(numBits); expected 2, 4, or 8.")
        }
        self.numBits = numBits
        self.embeddingDim = embeddingDim
        self.numBlocks = numBlocks
        super.init(weight: MLXArray.zeros([numEmbeddings, packedDim], dtype: wDtype))
        self._embeddingScale.wrappedValue = MLXArray.ones([numEmbeddings, numBlocks])
        self.freeze()
    }

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
        try super.updateMissing(parameter: parameter, verify: verify, path: path, modulePath: modulePath)
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
