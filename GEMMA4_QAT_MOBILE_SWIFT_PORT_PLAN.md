# Plan: Port Gemma 4 QAT Mobile (wNa8o8) to Swift

## 1. Overview

Port the Gemma 4 QAT mobile quantization format (`quant_method: "gemma"`,
wNa8o8) to Swift in `mlx-swift-lm`. The Python implementation is complete and
verified in `mlx-vlm` (122 tests pass, bit-exact vs HuggingFace reference).

**Goal**: Load and run inference on `gemma-4-E2B-it-qat-mobile-mlx-mm` (and
text-only variants) through the existing `Gemma4Model` / `Gemma4TextModel` Swift
model, with the custom quantized layers replacing standard `Linear`/`Embedding`
at load time.

**Approach**: Pure-Swift dequant-on-forward (no custom Metal kernels initially).
Weights stay packed in resident memory (the ~2.4 GB footprint is the point of
the format) and are unpacked transiently per layer inside the matmul. Custom
Metal kernels (qmv/qmm) are a follow-on optimization.

---

## 2. What exists today

### Swift repo (`mlx-swift-lm`)

| File | Role |
|---|---|
| `Libraries/MLXLLM/Models/Gemma4Text.swift` (947 lines) | Text-only Gemma 4: `Gemma4TextConfiguration`, `Gemma4Attention`, `Gemma4MLP`, `Gemma4DecoderLayer`, `Gemma4TextModelInner`, `Gemma4TextModel`. Uses standard `Linear`/`Embedding`. Already handles KV-shared layers, PLE gating, MoE, fused RMSNorm. |
| `Libraries/MLXLLM/Models/Gemma4.swift` (110 lines) | `Gemma4Configuration` (wraps `text_config`), `Gemma4Model` (wraps `Gemma4TextModel` as `languageModel`). `sanitize` strips `model.` prefix, skips vision/audio, remaps `language_model.` → `language_model.model.`. Registered as `"gemma4"`, `"gemma4_unified"`, `"gemma4_text"`. |
| `LLMModelFactory.swift` | `_load`: decodes `BaseConfiguration` (parses `"quantization"` key, **not** `"quantization_config"`), creates model, calls `loadWeights`. |
| `MLXLMCommon/Load.swift` | `loadWeights`: loads safetensors → `model.sanitize` → `quantize()` (only if `quantization`/`perLayerQuantization` non-nil) → `model.update(parameters:)`. |
| `MLXNN/Quantized.swift` | `QuantizedLinear` (subclasses `Linear`), `QuantizedEmbedding` (subclasses `Embedding`). Uses MLX group quantization (uint32, group_size, scales, biases). **Not compatible** with the Gemma mobile format. |
| `MLXNN/Linear.swift` | `Linear`: `public let weight: MLXArray`, `public let bias: MLXArray?`, `init(weight:bias:)` for subclasses. `callAsFunction` does `matmul(x, weight.T)`. |
| `MLXNN/Embedding.swift` | `Embedding`: `public let weight: MLXArray`, `init(weight:)` for subclasses. `callAsFunction` does `weight[x]`, `asLinear` does `matmul(x, weight.T)`. |

### Key insight: the loading flow

The Gemma mobile config uses `"quantization_config"` (not `"quantization"`), so
`BaseConfiguration.quantizationContainer` is **nil** → `perLayerQuantization`
is **nil** → the standard `quantize()` call in `Load.swift` is **skipped**. This
means we must do module replacement ourselves, inside `sanitize`.

The `sanitize` function is called on the model and has access to both the model
(`self`) and the weights dict. This is the same pattern used by `Gemma3.swift`
(VLM), which calls `quantize(model: self)` inside `sanitize`.

### Python reference (`mlx-vlm`)

| File | Role |
|---|---|
| `mlx_vlm/quantization/gemma_mobile.py` (1173 lines) | **THE KEY FILE**: `unpack_int2/int4/int8`, `apply_srq`, `dequantize_weight`, Metal kernels (qmv/qmm), `GemmaQuantizedLinear`, `GemmaQuantizedEmbedding`, `resolve_module_bits`, `replace_with_gemma_quant_layers`. |
| `mlx_vlm/quantization/gemma_mobile_cache.py` (226 lines) | Hybrid KV cache (4-bit global, 8-bit local). Scales extracted from HF weights before sanitize. **MLX checkpoint drops these scales** → falls back to standard caches. |
| `mlx_vlm/quantization/gemma_mobile_quantize.py` (~357 lines) | Conversion: `pack_int2/int4_row`, transcode, PTQ, `build_e2b_mobile_quantization_config`. |
| `mlx_vlm/models/gemma4/language.py` | `make_cache`: uses hybrid cache if KV scales present, else standard `KVCache`/`RotatingKVCache`. |

---

## 3. Format specification (verified against official sources)

### Weight packing (per-channel symmetric, packed in uint8/int8)

| Bits | Packing | Range | Shift |
|---|---|---|---|
| int2 | 4 values/byte in uint8, LSB-first: bits [1:0][3:2][5:4][7:6] | [-2, 1] | `& 0x3 - 2` |
| int4 | 2 values/byte in uint8, low-nibble-first | [-8, 7] | `& 0xF - 8` |
| int8 | direct int8 | [-128, 127] | none |

**Unpack** (Python `unpack_int2`):
```python
v0 = (packed & 0x03).astype(int8) - 2
v1 = ((packed >> 2) & 0x03).astype(int8) - 2
v2 = ((packed >> 4) & 0x03).astype(int8) - 2
v3 = (packed >> 6).astype(int8) - 2
out = stack([v0, v1, v2, v3], axis=-1).reshape(..., -1)[..., :in_features]
```

### Weight scale
- Per-channel: `[out_features, 1]` float32
- Dequant: `unpack(weight) * weight_scale`

### Static Range Quantization (SRQ) for activations
- Scalar `input_activation_scale` / `output_activation_scale` per layer
- `scale == 0` → no-op (uncalibrated)
- `apply_srq(x, scale) = clip(round(x / safe_scale), -128, 127) * safe_scale`
  where `safe_scale = where(scale != 0, scale, 1)`

### Embedding
- Same packing as weights
- Per-row scale: `[num_emb, 1]` or block-wise: `[num_emb, n_blocks]`
- `scalar_embed_scale` applied by surrounding model (not the layer)
- Checkpoint keys: `embedding_quantized` (packed), `embedding_scale` (float)

### Per-layer bit resolution (E2B, 35 layers)

| Pattern (regex) | Bits |
|---|---|
| `^lm_head$` | 2 |
| `language_model\.embed_tokens$` | 2 |
| `language_model\.embed_tokens_per_layer$` | 4 |
| `language_model\.layers\.(\d\|1[0-4])\.mlp\.` | 4 (layers 0–14) |
| `language_model\.layers\.\d+\.mlp\.` | 2 (layers 15–34) |
| `language_model\.layers\.\d+\.per_layer_input_gate$` | 8 |
| `language_model\.layers\.\d+\.per_layer_projection$` | 8 |
| `language_model\.layers\.\d+\.self_attn\.` | 4 |
| `vision_tower` | 8 (skipped for text-only) |
| `audio_tower(?!.*lconv1d\.linear_start)` | 2 (skipped for text-only) |
| `audio_tower\.layers\.\d+\.lconv1d\.linear_start\.` | 4 (skipped for text-only) |
| Default (`num_bits`) | 4 |

**`modules_to_not_convert`** (not quantized, stay fp):
`per_layer_model_projection`, `relative_k_proj`, `model.vision_tower.patch_embedder`,
`model.audio_tower.subsample_conv_projection`, `model.audio_tower.output_proj`,
`model.embed_audio`, `model.embed_vision`.

### KV cache
- **Hybrid**: 4-bit for global (full-attention) layers, 8-bit for local (sliding)
- Static per-tensor symmetric with precomputed `k_cache_scale`/`v_cache_scale`
- **BUT**: the MLX conversion drops these scales (`_DROP_SUFFIXES` in
  `gemma_mobile_quantize.py`). The MLX checkpoint has **no** `cache_scale` keys.
  → The Python implementation falls back to standard `KVCache`/`RotatingKVCache`.
  → **The Swift port will use standard caches for now** (the existing
  `Gemma4TextModel.newCache` already creates the right standard caches).

---

## 4. Weight key mapping

### Checkpoint keys (MLX format, `gemma-4-E2B-it-qat-mobile-mlx-mm`)

```
language_model.lm_head.{weight, weight_scale, input_activation_scale, output_activation_scale}
language_model.model.embed_tokens.{embedding_quantized, embedding_scale}
language_model.model.embed_tokens_per_layer.{embedding_quantized, embedding_scale}
language_model.model.layers.{N}.mlp.{gate_proj,up_proj,down_proj}.{weight, weight_scale, input_activation_scale, output_activation_scale}
language_model.model.layers.{N}.self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight, weight_scale, input_activation_scale, output_activation_scale}
language_model.model.layers.{N}.per_layer_input_gate.{weight, weight_scale, input_activation_scale, output_activation_scale}
language_model.model.layers.{N}.per_layer_projection.{weight, weight_scale, input_activation_scale, output_activation_scale}
language_model.model.per_layer_model_projection.weight          ← NOT quantized (no scales)
language_model.model.per_layer_projection_norm.weight
language_model.model.norm.weight
language_model.model.layers.{N}.{input_layernorm,post_attention_layernorm,...}.weight
language_model.model.layers.{N}.layer_scalar
```

### Swift model structure → weight paths

The existing `Gemma4Model.sanitize` already handles the key remapping:
- Strips `model.` prefix (for HF-format keys)
- Skips `vision_tower`, `audio_tower`, `embed_audio`, `embed_vision`, etc.
- Remaps `language_model.` → `language_model.model.` (for HF keys that started with `model.`)

The MLX checkpoint keys already have the `language_model.model.` prefix, so they
pass through sanitize unchanged. The model structure maps:
- `Gemma4Model.languageModel` (key: `"language_model"`) → `Gemma4TextModel`
  - `.lmHead` (key: `"lm_head"`) → `Linear` / `GemmaQuantizedLinear`
  - `.model` → `Gemma4TextModelInner`
    - `.embedTokens` (key: `"embed_tokens"`) → `Embedding` / `GemmaQuantizedEmbedding`
    - `.layers[N]` → `Gemma4DecoderLayer`
      - `.mlp.gateProj` (key: `"gate_proj"`) → `Linear` / `GemmaQuantizedLinear`
      - `.selfAttn.qProj` (key: `"q_proj"`) → `Linear` / `GemmaQuantizedLinear`
      - etc.

### Key remapping needed in `sanitize`

| Checkpoint key | Remapped to | Why |
|---|---|---|
| `*.embedding_quantized` | `*.weight` | `Embedding.weight` is discovered by property name `"weight"`, not `"embedding_quantized"` |
| `*.embedding_scale` | (no change) | `GemmaQuantizedEmbedding.embeddingScale` uses `@ParameterInfo(key: "embedding_scale")` |
| `*.weight` | (no change) | `Linear.weight` is discovered by property name `"weight"` |
| `*.weight_scale` | (no change) | `GemmaQuantizedLinear.weightScale` uses `@ParameterInfo(key: "weight_scale")` |
| `*.input_activation_scale` | (no change) | `@ParameterInfo(key: "input_activation_scale")` |
| `*.output_activation_scale` | (no change) | `@ParameterInfo(key: "output_activation_scale")` |

---

## 5. Implementation plan

### Phase 1: Custom quantized layers (`Gemma4MobileQuantization.swift`)

Create `Libraries/MLXLLM/Models/Gemma4MobileQuantization.swift` containing:

#### 5.1 `GemmaQuantizedLinear` (subclasses `Linear`)

```swift
class GemmaQuantizedLinear: Linear {
    let numBits: Int       // 2, 4, or 8
    let inputDims: Int     // unpacked input dimension

    @ParameterInfo(key: "weight_scale") var weightScale: MLXArray
    @ParameterInfo(key: "input_activation_scale") var inputActivationScale: MLXArray?
    @ParameterInfo(key: "output_activation_scale") var outputActivationScale: MLXArray?

    init(inputDims: Int, outputDims: Int, numBits: Int, bias: Bool = false) {
        // Create packed weight placeholder
        let packedIn = numBits == 2 ? (inputDims + 3) / 4
                     : numBits == 4 ? (inputDims + 1) / 2
                     : inputDims
        let wDtype: DType = numBits == 8 ? .int8 : .uint8
        let weight = MLXArray.zeros([outputDims, packedIn], dtype: wDtype)
        super.init(weight: weight, bias: bias ? .zeros([outputDims]) : nil)
        self.numBits = numBits
        self.inputDims = inputDims
        // Initialize scale placeholders
        self._weightScale.wrappedValue = MLXArray.ones([outputDims, 1])
        self._inputActivationScale.wrappedValue = nil
        self._outputActivationScale.wrappedValue = nil
    }

    // Designated init for subclass (accepts pre-quantized arrays)
    init(weight: MLXArray, bias: MLXArray?, weightScale: MLXArray,
         inputActivationScale: MLXArray?, outputActivationScale: MLXArray?,
         numBits: Int, inputDims: Int) {
        self.numBits = numBits
        self.inputDims = inputDims
        self._weightScale.wrappedValue = weightScale
        self._inputActivationScale.wrappedValue = inputActivationScale
        self._outputActivationScale.wrappedValue = outputActivationScale
        super.init(weight: weight, bias: bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // 1. Input SRQ (if calibrated)
        var x = x
        if let inScale = inputActivationScale {
            x = applySRQ(x, scale: inScale)
        }
        // 2. Dequantize weight: unpack(w) * weight_scale
        let w = dequantizeWeight(weight, weightScale: weightScale,
                                  numBits: numBits, inputDims: inputDims)
        // 3. Matmul
        var out = matmul(x, w.T)
        // 4. Output SRQ (if calibrated)
        if let outScale = outputActivationScale {
            out = applySRQ(out, scale: outScale)
        }
        // 5. Bias
        if let bias { out = out + bias }
        return out
    }
}
```

**Key details**:
- `weight` is inherited from `Linear` (`public let weight: MLXArray`), stores
  packed uint8/int8. Discovered by reflection with key `"weight"`.
- `weightScale`, `inputActivationScale`, `outputActivationScale` are
  `@ParameterInfo` with explicit snake_case keys. Optional scales use `MLXArray?`
  so missing keys don't fail `verify: .all`.
- `numBits` and `inputDims` are regular stored properties (not parameters).

#### 5.2 `GemmaQuantizedEmbedding` (subclasses `Embedding`)

```swift
class GemmaQuantizedEmbedding: Embedding {
    let numBits: Int
    let embeddingDim: Int
    let scalarEmbedScale: Float

    @ParameterInfo(key: "embedding_scale") var embeddingScale: MLXArray

    init(numEmbeddings: Int, embeddingDim: Int, numBits: Int,
         scalarEmbedScale: Float = 1.0, numBlocks: Int = 1) {
        let packedDim = numBits == 2 ? (embeddingDim + 3) / 4
                      : numBits == 4 ? (embeddingDim + 1) / 2
                      : embeddingDim
        let wDtype: DType = numBits == 8 ? .int8 : .uint8
        let weight = MLXArray.zeros([numEmbeddings, packedDim], dtype: wDtype)
        super.init(weight: weight)
        self.numBits = numBits
        self.embeddingDim = embeddingDim
        self.scalarEmbedScale = scalarEmbedScale
        self._embeddingScale.wrappedValue = MLXArray.ones([numEmbeddings, numBlocks])
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rows = weight[x]          // [*, packedDim] uint8/int8
        let scales = embeddingScale[x] // [*, n_blocks] float
        return dequantizeEmbeddingRows(rows, scales: scales,
                                       numBits: numBits, embeddingDim: embeddingDim)
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        // Dequantize full table, then matmul
        let w = dequantizeEmbeddingRows(weight, scales: embeddingScale,
                                        numBits: numBits, embeddingDim: embeddingDim)
        return matmul(x, w.T)
    }
}
```

**Key details**:
- `weight` inherited from `Embedding`, stores packed table. Key `"weight"`.
- `embeddingScale` is `@ParameterInfo(key: "embedding_scale")`.
- `sanitize` remaps `embedding_quantized` → `weight` for these layers.
- Block-wise scale: if `embeddingScale.shape[-1] != 1`, reshape ints to
  `[..., n_blocks, block_size]` and broadcast-multiply.

#### 5.3 Unpacking + SRQ helpers

Port from Python `gemma_mobile.py`:

```swift
func unpackInt2(_ packed: MLXArray, inFeatures: Int) -> MLXArray
func unpackInt4(_ packed: MLXArray, inFeatures: Int) -> MLXArray
func unpackInt(_ packed: MLXArray, numBits: Int, inFeatures: Int) -> MLXArray
func applySRQ(_ x: MLXArray, scale: MLXArray, bits: Int = 8) -> MLXArray
func dequantizeWeight(_ weight: MLXArray, weightScale: MLXArray,
                      numBits: Int, inputDims: Int, dtype: DType? = nil) -> MLXArray
func dequantizeEmbeddingRows(_ rows: MLXArray, scales: MLXArray,
                             numBits: Int, embeddingDim: Int) -> MLXArray
```

**Swift MLX equivalents** (from `porting.md` converting-python reference):
- `mx.array(0x03, uint8)` → `MLXArray(UInt8(0x03), dtype: .uint8)`
- `& _U2_MASK` → `& u2Mask`
- `.astype(mx.int8)` → `.asType(.int8)`
- `mx.stack([v0, v1, v2, v3], axis=-1)` → `MLX.stacked([v0, v1, v2, v3], axis: -1)`
- `mx.where(cond, a, b)` → `MLX.where(cond, a, b)`
- `mx.clip(x, lo, hi)` → `MLX.clip(x, min: lo, max: hi)` or `.clip(min:max:)`
- `mx.round(x)` → `MLX.round(x)`
- `packed >> 2` → `packed >> 2` (MLXArray supports bitwise shift)

#### 5.4 Configuration parsing (`GemmaMobileQuantizationConfig`)

```swift
struct GemmaMobileQuantizationConfig: Codable, Sendable {
    var moduleQuantConfigs: [String: ModuleQuantConfig]
    var modulesToNotConvert: [String]
    var numBits: Int = 4
    var quantMethod: String
    var quantizeEmbeddings: Bool = false

    struct ModuleQuantConfig: Codable {
        var numBits: Int
    }

    enum CodingKeys: String, CodingKey {
        case moduleQuantConfigs = "module_quant_configs"
        case modulesToNotConvert = "modules_to_not_convert"
        case numBits = "num_bits"
        case quantMethod = "quant_method"
        case quantizeEmbeddings = "quantize_embeddings"
    }

    var isGemmaMobile: Bool { quantMethod == "gemma" }
}
```

#### 5.5 Per-layer bit resolution

Port `resolve_module_bits` from Python. The key difference: Swift module paths
use `language_model.model.` prefix (matching the model structure), while the
regex patterns use `language_model.` (HF namespace). Apply the same
normalization as Python's `_normalize_module_path`:

```swift
func resolveModuleBits(path: String, config: GemmaMobileQuantizationConfig) -> Int? {
    // 1. Normalize path: language_model.model.X → language_model.X
    //    language_model.lm_head → lm_head
    var normalized = path
    if normalized.hasPrefix("language_model.model.") {
        normalized = "language_model." + normalized.dropFirst("language_model.model.".count)
    }
    if normalized == "language_model.lm_head" { normalized = "lm_head" }

    // 2. Check modules_to_not_convert (strip "model." prefix, match as path segment)
    for entry in config.modulesToNotConvert {
        let e = entry.hasPrefix("model.") ? String(entry.dropFirst(6)) : entry
        if !e.isEmpty && pathContainsSegment(normalized, e) { return nil }
    }

    // 3. Match regex patterns (first match wins)
    for (pattern, opts) in config.moduleQuantConfigs {
        if let regex = try? NSRegularExpression(pattern: pattern),
           regex.firstMatch(in: normalized, range: NSRange(location: 0, length: normalized.count)) != nil {
            return opts.numBits
        }
    }

    // 4. Default
    return config.numBits
}
```

**Note**: Python uses `re.search` (find anywhere in string), so we use
`NSRegularExpression` with `firstMatch` (not `^...$` anchored unless the pattern
has `^`/`$`).

#### 5.6 Module replacement function

```swift
func replaceWithGemmaQuantLayers(
    model: Module,
    quantizationConfig: GemmaMobileQuantizationConfig,
    weights: [String: MLXArray]
) {
    let updates = model.leafModules().flattened().compactMap { (path, m) -> (String, Module)? in
        // Skip already-quantized modules
        if m is Quantized { return nil }

        let bits = resolveModuleBits(path: path, config: quantizationConfig)
        guard let bits else { return nil }  // in modules_to_not_convert

        if let linear = m as? Linear {
            // Only replace if checkpoint has weight_scale for this layer
            guard weights["\(path).weight_scale"] != nil else { return nil }
            let outDims = linear.shape.0
            let inDims = linear.shape.1
            let hasBias = linear.bias != nil
            return (path, GemmaQuantizedLinear(inputDims: inDims, outputDims: outDims,
                                               numBits: bits, bias: hasBias))
        }

        if let embedding = m as? Embedding {
            guard quantizationConfig.quantizeEmbeddings else { return nil }
            guard weights["\(path).embedding_scale"] != nil else { return nil }
            let (numEmb, dim) = embedding.shape
            // Determine block-wise vs per-row from checkpoint scale shape
            let numBlocks = weights["\(path).embedding_scale"]?.dim(-1) ?? 1
            return (path, GemmaQuantizedEmbedding(numEmbeddings: numEmb,
                                                  embeddingDim: dim, numBits: bits,
                                                  numBlocks: numBlocks))
        }

        return nil
    }

    model.update(modules: ModuleChildren.unflattened(updates))
}
```

**Key details**:
- `leafModules().flattened()` returns `[(path: String, module: Module)]` where
  path is dot-separated (e.g., `"language_model.model.layers.0.mlp.gate_proj"`).
- We check `weights["\(path).weight_scale"]` to only replace layers that are
  actually quantized in the checkpoint (same as Python).
- `model.update(modules:)` replaces the modules in-place. Since
  `GemmaQuantizedLinear` IS a `Linear` and `GemmaQuantizedEmbedding` IS an
  `Embedding`, the `@ModuleInfo` typed properties accept them.

### Phase 2: Wire into the model

#### 5.7 Add `quantizationConfig` to configurations

**`Gemma4TextConfiguration`** (in `Gemma4Text.swift`):
```swift
var quantizationConfig: GemmaMobileQuantizationConfig?

// In init(from:):
self.quantizationConfig = try container.decodeIfPresent(
    GemmaMobileQuantizationConfig.self, forKey: .quantizationConfig)

// In CodingKeys:
case quantizationConfig = "quantization_config"
```

**`Gemma4Configuration`** (in `Gemma4.swift`):
```swift
var quantizationConfig: GemmaMobileQuantizationConfig?

// In init(from:):
self.quantizationConfig = try container.decodeIfPresent(
    GemmaMobileQuantizationConfig.self, forKey: .quantizationConfig)
// Propagate to textConfig if text_config didn't have its own
if self.quantizationConfig != nil && self.textConfig.quantizationConfig == nil {
    self.textConfig.quantizationConfig = self.quantizationConfig
}
```

#### 5.8 Override `sanitize` in `Gemma4TextModel`

Add to `Gemma4TextModel`:

```swift
public func sanitize(weights: [String: MLXArray], metadata: [String: String])
    -> [String: MLXArray]
{
    var sanitized = sanitizeWeights(weights)  // existing MoE/KV-shared logic

    // Gemma mobile quantization: replace Linear/Embedding with quantized layers
    if let qc = config.quantizationConfig, qc.isGemmaMobile {
        // 1. Remap embedding keys: embedding_quantized → weight
        sanitized = sanitized.mapKeys { key in
            if key.hasSuffix(".embedding_quantized") {
                return String(key.dropLast(".embedding_quantized".count)) + ".weight"
            }
            return key
        }
        // 2. Replace modules in-place
        replaceWithGemmaQuantLayers(model: self, quantizationConfig: qc,
                                    weights: sanitized)
    }

    return sanitized
}
```

**Note**: The existing `Gemma4TextModel.sanitize(weights:)` is the
`BaseLanguageModel` protocol method. We add a `sanitize(weights:metadata:)`
override (the version called by `loadWeights`) that wraps the existing logic
and adds the Gemma mobile path. The existing `sanitize(weights:)` (MoE remap,
KV-shared pruning) is called as a helper.

**`Gemma4Model.sanitize`** (in `Gemma4.swift`):
- The existing sanitize strips `model.`, skips vision/audio, remaps keys, then
  calls `languageModel.sanitize(weights:)`.
- We need to ensure `languageModel.sanitize(weights:metadata:)` is called
  instead (or the Gemma mobile path is triggered). The simplest approach:
  `Gemma4Model.sanitize` delegates to `languageModel.sanitize(weights:metadata:)`
  after its key remapping.

#### 5.9 KV cache (no change needed)

The MLX checkpoint has no `k_cache_scale`/`v_cache_scale` (dropped during
conversion). The existing `Gemma4TextModel.newCache` already creates:
- `StandardKVCache` for full-attention layers
- `RotatingKVCache` for sliding layers
- Only for the first `numHiddenLayers - numKvSharedLayers` layers (KV-shared
  layers get nil caches)

This is exactly what the Python implementation does when KV scales are absent.
**No change needed for Phase 1.**

### Phase 3: Registration (no change needed)

The model is already registered:
```swift
"gemma4": create(Gemma4Configuration.self, Gemma4Model.init),
"gemma4_text": create(Gemma4TextConfiguration.self, Gemma4TextModel.init),
```

The MLX checkpoint has `model_type: "gemma4"`, so it loads through
`Gemma4Model`. No new registration needed.

### Phase 4: Testing

#### 5.10 Unit tests for the quantization primitives

Create `Tests/MLXLMTests/Gemma4MobileQuantizationTests.swift`:
- `testUnpackInt2` — verify unpacking matches Python (known byte → known values)
- `testUnpackInt4` — same for int4
- `testApplySRQ` — verify `scale == 0` is no-op, `scale != 0` quantizes
- `testDequantizeWeight` — round-trip: pack → unpack → dequant
- `testResolveModuleBits` — verify regex matching for all E2B patterns
- `testResolveModuleBitsNotConvert` — verify `per_layer_model_projection` → nil

#### 5.11 Integration test

Create `Tests/MLXLMTests/Gemma4MobileIntegrationTests.swift`:
- Load the model from `gemma-4-E2B-it-qat-mobile-mlx-mm` (or a small fixture)
- Verify module replacement: check that `lm_head` is `GemmaQuantizedLinear` with
  `numBits == 2`, `embed_tokens` is `GemmaQuantizedEmbedding`, etc.
- Run a short prompt and verify output is coherent (compare with Python output)
- Verify memory footprint is ~2.4 GB (not 4.4 GB like the 4-bit model)

#### 5.12 Cross-validation with Python

Use the `trace` technique from `porting.md`:
- Add temporary `trace(name, x)` functions in both Python and Swift
- Compare shapes and `sum()` values at each layer boundary
- Start with: token IDs → embedding output → first layer output → ... → logits

---

## 6. File summary

| File | Action | Description |
|---|---|---|
| `Libraries/MLXLLM/Models/Gemma4MobileQuantization.swift` | **Create** | `GemmaQuantizedLinear`, `GemmaQuantizedEmbedding`, unpack/SRQ helpers, `GemmaMobileQuantizationConfig`, `resolveModuleBits`, `replaceWithGemmaQuantLayers` |
| `Libraries/MLXLLM/Models/Gemma4Text.swift` | **Modify** | Add `quantizationConfig` to `Gemma4TextConfiguration`; add `sanitize(weights:metadata:)` override to `Gemma4TextModel` |
| `Libraries/MLXLLM/Models/Gemma4.swift` | **Modify** | Add `quantizationConfig` to `Gemma4Configuration`; ensure `sanitize` delegates to `languageModel.sanitize(weights:metadata:)` |
| `Tests/MLXLMTests/Gemma4MobileQuantizationTests.swift` | **Create** | Unit tests for primitives |
| `Tests/MLXLMTests/Gemma4MobileIntegrationTests.swift` | **Create** | Integration test |

---

## 7. Implementation order

1. **`Gemma4MobileQuantization.swift`** — unpack helpers, SRQ, dequantize,
   `GemmaQuantizedLinear`, `GemmaQuantizedEmbedding`, config struct,
   `resolveModuleBits`, `replaceWithGemmaQuantLayers`
2. **Unit tests** for primitives (verify bit-exactness vs Python)
3. **`Gemma4Text.swift`** — add `quantizationConfig` field + `sanitize` override
4. **`Gemma4.swift`** — add `quantizationConfig` field + sanitize delegation
5. **Build** — `swift build` to verify compilation
6. **Integration test** — load model, verify module replacement, run prompt
7. **Cross-validate** — compare output with Python using trace technique
8. **Clean up** — remove temporary trace functions

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `@ParameterInfo` with optional `MLXArray?` may not work with `verify: .all` | The porting guide explicitly shows `@ParameterInfo var bias: MLXArray?` for optional params. `QuantizedLinear.biases` is also optional. Should work. |
| `leafModules()` path format may differ from expected | Test with `print(model.leafModules().flattened())` to verify paths match `language_model.model.layers.N.mlp.gate_proj` format |
| `Embedding.weight` is `let` (not `@ParameterInfo`) — key is property name `"weight"` | Remap `embedding_quantized` → `weight` in sanitize. Verified: `QuantizedEmbedding` uses the same inherited `weight`. |
| Regex matching: Python `re.search` vs Swift `NSRegularExpression` | Use `firstMatch(in:range:)` (not anchored unless pattern has `^`/`$`). Test all E2B patterns. |
| `model.update(modules:)` may not accept subclass types | `QuantizedLinear` (subclass of `Linear`) is already used this way by `quantize()`. Same pattern. |
| Performance: pure-Swift dequant-on-forward is slower than custom Metal kernels | Acceptable for initial port. Python's pure-MLX fallback works. Optimize with Metal kernels in Phase 2. |
| `per_layer_model_projection` must stay unquantized | `resolveModuleBits` returns `nil` for `modules_to_not_convert`. Replacement checks `weights["\(path).weight_scale"]` — absent for this layer. |

---

## 9. Future optimizations (not in scope for Phase 1)

### 9.1 Custom Metal kernels (qmv/qmm)
Port the Python `_gemma_qmv_kernel` (decode, batch ≤ 16) and `_gemma_qmm_kernel`
(prefill, batch > 1) Metal kernels. These fuse SRQ and avoid materializing the
full fp weight. Requires `mx.fast.metalKernel` equivalent in Swift
(`MLXFast.metalKernel` or raw Metal kernel compilation).

### 9.2 Hybrid KV cache (4-bit global, 8-bit local)
Re-convert the checkpoint to include `k_cache_scale`/`v_cache_scale` (currently
dropped by `gemma_mobile_quantize.py`), or compute them at runtime. Then
implement `GemmaStaticQuantizedKVCache` / `GemmaStaticQuantizedRotatingKVCache`
(subclassing `KVCacheSimple` / `RotatingKVCache`, overriding `update` to
quantize/dequantize with the static scale).

### 9.3 Fused q/k/v projection
Port `gemma_fused_qkv_matmul` — concatenate packed q/k/v weights and run a
single matmul with per-row output SRQ, replacing three kernel launches with one.

### 9.4 Kernel precompilation
Port `precompile_gemma_mobile_kernels` — JIT-compile all Metal kernel variants
at load time so the first prompt doesn't pay the ~0.5 s compilation cost.

---

## 10. References

- [HuggingFace collection](https://huggingface.co/collections/google/gemma-4-qat-mobile)
- [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
- [ai.google.dev docs](https://ai.google.dev/gemma/docs/core#qat)
- [arXiv paper](https://arxiv.org/pdf/2607.02770)
- Python reference: `mlx-vlm/mlx_vlm/quantization/gemma_mobile.py`
- Swift porting guide: `mlx-swift-lm/Libraries/MLXLMCommon/Documentation.docc/porting.md`
- MLX checkpoint: `mlx-vlm/gemma-4-E2B-it-qat-mobile-mlx-mm/`
