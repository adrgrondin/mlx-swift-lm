# Plan: Port Native Quantized MatMul + Compiled Fusion to Swift

## 1. Goal

Port the Python Phase 6/7 optimization (native `quantized_matmul` + `mx.compile`
fusion, verified in `mlx-vlm`) to the Swift port in `mlx-swift-lm`. This is the
single most impactful optimization identified in the Python work: it improved
decode from ~140 → ~155 tok/s (+11% CLI) and prefill from ~189 → ~441 tok/s
(+133%), roughly matching LiteRT-LM's decode speed (160 tok/s).

**The core idea:** MLX's native `quantizedMM` is *compile-friendly* (unlike the
custom `metalKernel` qmv/qmm kernels, which are opaque to `compile`). This lets
`compile` fuse the **entire** element-wise chain — RMSNorm → SRQ → quantizedMM →
SRQ → RMSNorm → residual → gelu → … — into a single compiled Metal graph per
decoder-layer segment. This is true LiteRT-style fusion: ~100 separate kernel
launches per step collapse into ~3 (one compiled pre-attention graph, one eager
SDPA, one compiled post-attention graph).

---

## 1a. Current Status

| Phase | Status | Notes |
|---|---|---|
| Phase 0 — Spike | ✅ DONE | 5 tests pass: `quantizedMM` + `MLXFast.RoPE` (incl. `inf` freqs) inside `compile`. |
| Phase 1 — Weight extraction + SRQ | ✅ DONE | `srqF32`, `nativeArgs()`, `buildFusedQKVNative`, `compiledRopeFreqs`. |
| Phase 2 — Compiled pre-attention | ✅ DONE | `CompiledPreAttnSource`, `CompiledPreAttnKvshared` factory + cache. |
| Phase 3 — Compiled post-attention | ✅ DONE | `CompiledPostAttn` factory + cache (38-element input). |
| Phase 4 — Wire into decoder layer | ✅ DONE | `getNativeArgs()`, native compiled path, `eagerCall` fallback, `useNativeCompiledPath` A/B flag. **A/B equivalence test PASSES** (mean abs diff 0.30, 2/32 argmax mismatches). |
| Phase 5 — Load-time precompilation | ✅ DONE | `NativePrecompilable` protocol + `loadWeights` hook; `precompileNativeFunctions` (layer-by-layer convert + free + hybrid compile: full pass ≤ 32, direct call > 32); all source/shared + sliding/full signatures and 512-token chunks are covered; `freeMobileWeights`; `precompileAtLoad` A/B flag. **3 tests pass**: E2B signature selection, weight freeing + prefill stability (real model), and no-op for unaligned dims (tiny model). |
| Phase 6 — Benchmark | ⬜ Not started | Decode/prefill tok/s, peak memory. |

### Root cause of the A/B test failure (resolved)

The native compiled path and the eager path produced different logits (mean abs
diff 1.16, 9/32 argmax mismatches). The root cause was **SRQ precision**: the
eager path's `CompiledSRQ` did SRQ in **bfloat16** (`s.asType(x.dtype)`), while
the native path's `srqF32` did SRQ in **float32**. The bfloat16 SRQ rounded
differently from the float32 SRQ (up to 0.5 per element), and `quantizedMM`
amplified this to a 4.4% relative output difference, compounding across 35
layers.

**Fix:** Updated `CompiledSRQ` to do all SRQ math in float32 (`x.asType(.float32)`,
`s.asType(.float32)`) and updated `GemmaQuantizedLinear.callAsFunction` (batch > 16
path) to cast the input to float32 before SRQ, pass float32 to `quantizedMM`, and
cast the output back to the original dtype after the output SRQ — matching the
native path's dtype flow exactly. This reduced the mean abs diff from 1.16 → 0.30
and argmax mismatches from 9 → 2.

The remaining 0.30 mean abs diff (2/32 argmax mismatches) is numerical noise from
(1) the fused q/k/v matmul (native) vs three separate matmuls (eager) — different
Metal tiling/accumulation, and (2) compile fusion (native) vs separate ops
(eager). This is expected for two different code paths and is within the test
tolerance (mean < 0.5, argmax ≤ 4).

**Key investigation findings:**
- RoPE `inf`-freqs approach (native) is **exactly equivalent** to the eager
  `ProportionalRoPE` half-split approach (max diff = 0.0, both inside and
  outside `compile`). Not the bug.
- `quantizedMM` always returns **float32** output regardless of input dtype.
- The `quantizedMM` values are identical for bfloat16 vs float32 input **when the
  SRQ rounds to the same integers** — the difference is entirely from SRQ rounding.
- Weight mapping, arg ordering, `quantizedMM` mode (`.affine`), and `vNormW =
  ones` vs `mlxNone` were all verified correct.

---

## 2. What exists today (Swift port)

The Swift port **already has** the building blocks for the native path. This
plan is about *assembling* them into compiled-fusion segments, not building
from scratch.

### `Libraries/MLXLMCommon/GemmaMobileQuantization.swift` (~990 lines)

| Component | Status | Notes |
|---|---|---|
| `unpackInt2/int4/int` | ✅ done | Bit-exact vs Python. 17 unit tests pass. |
| `applySRQ` | ✅ done | Plain (compile-friendly) SRQ: `where(scale≠0, clip(round(x/s),-128,127)*s, x)`. |
| `CompiledSRQ` | ✅ done | Separately compiled (`compile(shapeless: true)`) SRQ. Used by the *current* per-layer fallback path. Will be **superseded** by the fused segments. |
| `dequantizeWeight` / `dequantizeEmbeddingRows` | ✅ done | Dequant-on-forward. |
| `mobileToMLX` | ✅ done | Mobile → MLX uint32 (group_size=128) conversion. Bit-exact. **This is the key ingredient** for `quantizedMM`. |
| `GemmaMobileQuantizationConfig` + `resolveModuleBits` | ✅ done | Config parsing, per-layer bit resolution. |
| Metal qmv kernel (`gemmaQMVSource`, `GemmaQMVKernelCache`) | ✅ done | Fused SRQ, 2 simdgroups. Used for decode (batch ≤ 16) in the *current* path. Will become a **fallback** (lm_head, non-compiled layers). |
| `gemmaFusedQKVMatmul` | ✅ done | Fused q/k/v via qmv kernel. Will be replaced by fused native `quantizedMM` in the compiled pre-attention segment. |
| `GemmaQuantizedLinear` | ✅ done | Current dispatch: **qmv kernel** (batch ≤ 16) or **quantizedMM + CompiledSRQ** (batch > 16). Has lazy `convertToMLXFormat()` storing `_mlxWeight/_mlxScales/_mlxBiases`. |
| `GemmaQuantizedEmbedding` | ✅ done | Dequant-on-forward (gather + unpack). Stays as-is. |
| `replaceWithGemmaQuantLayers` + `applyGemmaMobileQuantization` | ✅ done | Module replacement at load time. |

### `Libraries/MLXLLM/Models/Gemma4Text.swift` (~1040 lines)

| Component | Status | Notes |
|---|---|---|
| `compile(shapeless: true)` for `_addRMSNorm` and `_geluMul` | ✅ done | Already fuses residual + RMSNorm and gelu*other. Proves `compile` works for this model. |
| `Gemma4Attention` | ✅ done | Fused qkv (qmv), qNorm/kNorm/vNorm, RoPE, SDPA, KV-shared layers. |
| `Gemma4MLP` | ✅ done | gate/up/down with `geluApproximate`. |
| `Gemma4DecoderLayer` | ✅ done | input_norm → attn → `_addRMSNorm` → pre_ff_norm → mlp → `_addRMSNorm` → PLE → `_geluMul` → `_addRMSNorm` → `layerScalar`. **This is the eager path that will get a compiled alternative.** |
| `Gemma4TextModelInner` | ✅ done | Embeddings, PLE, layer loop with KV sharing. |
| `Gemma4TextModel` | ✅ done | lm_head, logit softcap, sanitize (mobile quant hook). |
| `kRMSEps = 1e-6` | ✅ done | Hardcoded eps (all layers assert match). Lets one compiled graph serve every layer. |

### MLX-Swift APIs (all available, verified)

| API | Swift equivalent of Python |
|---|---|
| `mx.compile` | `compile(shapeless: Bool, _ f: ([MLXArray]) -> [MLXArray])` (primary, any arg count) + convenience overloads (1–12 args) |
| `mx.quantized_matmul` | `quantizedMM(x, w, scales:, biases:, transpose:, groupSize:, bits:, mode:)` |
| `mx.fast.rms_norm` | `MLXFast.rmsNorm(x, weight:, eps:)` |
| `mx.fast.rope` | `MLXFast.RoPE(x, dims:, traditional:, base:, scale:, offset:, freqs:)` |
| `mx.fast.scaled_dot_product_attention` | `MLXFast.scaledDotProductAttention(queries:, keys:, values:, scale:, mask:)` |
| `mx.eval` | `eval([array])` |

### Build & test baseline

- `swift build --target MLXLLM` — ✅ builds clean.
- `swift test --filter Gemma4MobileQuantizationTests` — ✅ 17/17 pass.

---

## 3. The gap: what Phase 6/7 does that Swift doesn't

### Current Swift decode path (per layer, batch = 1)

```
inputLayernorm(x)                    → 1 RMSNorm kernel
qProj(h) → qmvCall                    → 1 qmv Metal kernel (fuses SRQ)
qNorm / kNorm / vNorm                 → 3 RMSNorm kernels
RoPE                                  → 1 RoPE kernel
SDPA                                  → 1 SDPA kernel
oProj(attnOut) → qmvCall              → 1 qmv Metal kernel
_addRMSNorm(residual, h, postAttnNorm)→ 1 compiled kernel (norm + residual)
preFeedforwardLayernorm               → 1 RMSNorm kernel
gateProj/upProj → qmvCall × 2         → 2 qmv Metal kernels
geluApproximate(gate) * up            → 1 compiled kernel (_geluMul)
downProj → qmvCall                    → 1 qmv Metal kernel
_addRMSNorm(residual, h, postFfNorm) → 1 compiled kernel
PLE gate → qmvCall                    → 1 qmv Metal kernel
_geluMul(gate, perLayerInput)         → 1 compiled kernel
PLE proj → qmvCall                    → 1 qmv Metal kernel
_addRMSNorm(residual, g, pleNorm)     → 1 compiled kernel
layerScalar multiply                  → 1 element-wise kernel
```
**~20 kernel launches per layer × 35 layers = ~700 launches/step.**

### Phase 6/7 compiled path (per layer, batch = 1)

```
preAttnFn(x, *args, offset)     → 1 compiled graph (norm + SRQ + quantMM + SRQ + 3 norms + 2 RoPEs)
SDPA                            → 1 eager SDPA kernel
postAttnFn(residual, attn, pli) → 1 compiled graph (SRQ + quantMM + SRQ + norm + residual + norm + SRQ + 2×quantMM + SRQ + gelu + SRQ + quantMM + SRQ + norm + residual + PLE gate/gelu/mul/proj/norm/residual + layerScalar)
```
**3 kernel launches per layer × 35 layers = ~105 launches/step.**

The ~6× reduction in kernel launches is the primary speedup. The secondary
speedup is that `quantizedMM` is 1.03–1.61× faster than the custom qmv kernel
for the raw matmul (better compute efficiency).

### Why the current Swift path can't just "use quantizedMM more"

The Swift `GemmaQuantizedLinear.callAsFunction` already calls `quantizedMM` for
prefill (batch > 16), but with **separate** `CompiledSRQ` calls before/after.
The SRQ is compiled in isolation, not fused with the matmul or the surrounding
norms. The Python Phase 4 experiment showed this exact pattern is *net-negative*
(unfused SRQ has 5+ separate ops that add overhead). The Phase 6/7 fix is to
put the SRQ **inside** the `compile` graph so it fuses with the matmul and
norms for free.

**Note:** The `CompiledSRQ` now uses float32 (matching `srqF32` and the Python
`_srq`). The previous bfloat16 SRQ caused divergence from the native compiled
path (see §1a for details).

---

## 4. Architecture

### 4.1 Three compiled segments per decoder layer

Mirror the Python `_get_compiled_pre_attn_source`, `_get_compiled_pre_attn_kvshared`,
`_get_compiled_post_attn_mlp_ple`. The KV cache update + SDPA stay **eager**
between the two compiled segments (avoids the O(n²) growing-concatenate
regression — confirmed in Python).

#### Pre-attention (source layers 0–14, own K/V)

```
input_layernorm → [cast f32] → SRQ → quantizedMM(fused_qkv) → SRQ → [cast dtype]
→ q_norm → k_norm → v_norm
→ transpose → RoPE(keys) → transpose → RoPE(queries)
→ returns (queries, keys, values)
```

Inputs (12 MLXArrays — fits the convenience overload):
`x, inputNormW, qkvWq, qkvScales, qkvBiases, qkvInS, qkvOutS, qNormW, kNormW, vNormW, ropeFreqs, offset`

Captured constants: `nHeads`, `headDim`, `nKvHeads`, `kRMSEps`, `bits=4` (qkv
is always 4-bit in the E2B mobile schema).

#### Pre-attention (KV-shared layers 15–34, reuse earlier K/V)

```
input_layernorm → [cast f32] → SRQ → quantizedMM(q_proj) → SRQ → [cast dtype]
→ q_norm → transpose → RoPE(queries)
→ returns queries
```

Inputs (10 MLXArrays):
`x, inputNormW, qWq, qScales, qBiases, qInS, qOutS, qNormW, ropeFreqs, offset`

Captured constants: `nHeads`, `headDim`, `kRMSEps`, `bits=4`.

#### Post-attention (all layers)

```
SRQ → quantizedMM(o_proj) → SRQ → postAttnNorm → residual
→ preFfNorm → [cast f32] → SRQ → quantizedMM(gate) → SRQ → SRQ → quantizedMM(up) → SRQ
→ gelu(gate) * up → [cast f32] → SRQ → quantizedMM(down) → SRQ → postFfNorm → residual
→ PLE: [cast f32] → SRQ → quantizedMM(ple_gate) → SRQ → gelu → * perLayerInput
  → [cast f32] → SRQ → quantizedMM(ple_proj) → SRQ → pleNorm → residual
→ * layerScalar
→ returns h
```

Inputs (~37 MLXArrays — **must use the `[MLXArray]` array form** of `compile`):
`[residual, attnOutput, perLayerInput, postAttnW, preFfW, postFfW, pleNormW,
layerScalar, oWq, oScales, oBiases, oInS, oOutS, gateWq, gateScales, gateBiases,
gateInS, gateOutS, upWq, upScales, upBiases, upInS, upOutS, downWq, downScales,
downBiases, downInS, downOutS, pleGwq, pleGscales, pleGbiases, pleGinS, pleGoutS,
plePwq, plePscales, plePbiases, plePinS, plePoutS]`

Captured constants: `kRMSEps`, `mlpBits` (4 for layers 0–14, 2 for 15–34),
`pleBits` (8). Different bit-width combos need different compiled functions
(cached by `(mlpBits, pleBits)`).

### 4.2 Factory + caching

The Python uses `@lru_cache` on the factory functions to share one compiled
function across all layers with the same constants. Swift has no `@lru_cache`,
so we use a static dictionary cache:

```swift
private enum CompiledPreAttnSourceCache {
    struct Key: Hashable { let nHeads: Int; let headDim: Int; let nKvHeads: Int }
    nonisolated(unsafe) static var cache: [Key: ([MLXArray]) -> [MLXArray]] = [:]
    static func fn(nHeads: Int, headDim: Int, nKvHeads: Int) -> ([MLXArray]) -> [MLXArray] {
        let key = Key(nHeads: nHeads, headDim: headDim, nKvHeads: nKvHeads)
        if let cached = cache[key] { return cached }
        let f = compile { (inputs: [MLXArray]) -> [MLXArray] in /* ... */ }
        cache[key] = f
        return f
    }
}
```

Same pattern for `CompiledPreAttnKvsharedCache` (key: `(nHeads, headDim)`) and
`CompiledPostAttnCache` (key: `(mlpBits, pleBits)`).

`nonisolated(unsafe)`: single-threaded inference context (same pattern as the
existing `GemmaQMVKernelCache` and `CompiledSRQ`).

### 4.3 Weight extraction (`GemmaQuantizedLinear.nativeArgs`)

Add a public accessor to `GemmaQuantizedLinear` that returns the pre-converted
native weights + SRQ scales, caching on the module (mirrors Python
`_qlinear_native_args`):

```swift
extension GemmaQuantizedLinear {
    /// Returns `(packed, scales, biases, inScale, outScale)` for `quantizedMM`.
    /// Conversion is bit-exact via `mobileToMLX` and cached on the module.
    func nativeArgs() -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        if !_conversionDone { convertToMLXFormat(); _conversionDone = true }
        let packed = _mlxWeight!
        let scales = _mlxScales!
        let biases = _mlxBiases!
        let inS = inputActivationScale
        let outS = outputActivationScale
        return (packed, scales, biases, inS, outS)
    }
}
```

Note: `convertToMLXFormat()` already exists and handles the `inputDims % 128`
guard (returns nil `_mlxWeight` for unaligned dims → caller falls back to
eager). The `nativeArgs()` accessor just wraps it and returns the SRQ scales.

### 4.4 RoPE frequency extraction

The Python `_get_rope_freqs` extracts `rope._freqs` (for `ProportionalRoPE`) or
precomputes `base^(arange(0, dims, 2) / dims)` (for `nn.RoPE`). The Swift port
uses `RoPELayer` (from `initializeRope`). Need to extract the freqs array to
pass as an input to the compiled function.

Check whether `RoPELayer` exposes `_freqs` or precompute:
```swift
let freqs = MLX.power(MLXArray(rope.base, dtype: .float32),
    MLX.arange(0, rope.dims, step: 2, dtype: .float32) / Float(rope.dims))
```

### 4.5 Fused q/k/v native weights

The Python `_build_fused_qkv_native` concatenates q/k/v native weights
(`wq, scales, biases`) along the output dim and builds a per-row output SRQ
scale. The Swift port already has `gemmaBuildPerRowOutputScale`; need a
`buildFusedQKVNative(attn:)` that concatenates the *native* (uint32) weights.

### 4.6 Decoder layer native path

`Gemma4DecoderLayer.callAsFunction` gets a native compiled path (before the
existing eager path). Guard: no MoE, PLE present, all relevant linears are
`GemmaQuantizedLinear` with `inputDims % 128 == 0`. Lazily extract + cache args
on first call (mirrors Python `_get_native_args`).

```swift
// In Gemma4DecoderLayer.callAsFunction:
if let native = getNativeArgs() {   // lazily cached, returns nil if not usable
    let (isSource, preFn, preArgs, postFn, postArgs) = native
    let offset = mx.array(cache?.ropeOffset ?? 0)  // must be MLXArray, not Int
    if isSource {
        let (q, k, v) = preFn([x] + preArgs + [offset])
        let (k2, v2) = cache.update(keys: k, values: v)
        let attnOut = attnSDPA(q, k2, v2, mask)
        // cache sharedKV for KV-shared layers
    } else {
        let q = preFn([x] + preArgs + [offset])[0]
        let attnOut = attnSDPA(q, sharedKV.keys, sharedKV.values, mask)
    }
    let h = postFn([residual, attnOut, perLayerInput] + postArgs)[0]
    return (h, sharedKV, offset)
}
// ... existing eager path as fallback ...
```

### 4.7 Load-time precompilation + weight freeing

Mirror Python `precompile_native_functions`. Called after weights are loaded
and modules are replaced. Three steps:

1. **Convert + free mobile weights layer by layer** — for each decoder layer,
   call `nativeArgs()` on each `GemmaQuantizedLinear`, `eval` the native weights,
   then replace the mobile `weight`/`weightScale` with dummy arrays. Keeps the
   conversion peak at ~half-mobile + half-native.

2. **Precompile `compile` functions** — run dummy forward passes for common
   prompt lengths `(1, 16, 32, 64, 128, 256, 512)` with `eval` on the output.
   Use the **hybrid compile strategy** (from the Python Phase 7 peak-memory fix):
   full forward pass for shapes ≤ 32 (negligible activations, warms up MLX
   built-in ops), direct compiled-function calls for larger shapes (avoids
   ~0.8 GB activation spike). Direct compilation selects one representative
   for each unique pre- and post-attention signature, covering source/KV-shared
   and sliding/full-attention variants without running all 35 layers.

3. **Set flags** — pre-set `_weightsConverted`/`_mobileWeightsFreed` so the
   fallback weight-freeing path in the model is skipped.

**Hook point in Swift:** The loading flow is `Load.swift: loadWeights` →
`sanitize` → `quantize` → `update(parameters:)`. The precompilation must run
*after* `update(parameters:)` (weights loaded) and *after* module replacement
(in `sanitize`). Options:
- (a) A model-optional `postLoad()` protocol method called by `loadWeights`
  after `update(parameters:)`.
- (b) A type check in `loadWeights` (like Python's `_needs_native_precompile`
  flag): `if let model = model as? Gemma4NativePrecompilable { model.precompileNativeFunctions() }`.

Option (b) is simpler and mirrors the Python approach.

---

## 5. Swift-specific considerations

### 5.1 `compile` argument count

| Segment | Arg count | Form to use |
|---|---|---|
| Pre-attn source | 12 | Convenience overload (1–12 args) **or** array form |
| Pre-attn KV-shared | 10 | Convenience overload **or** array form |
| Post-attn | ~37 | **Array form only** (`[MLXArray] -> [MLXArray]`) |

For consistency and to avoid index errors, **recommend the array form for all
three**. The convenience overloads are cleaner but mixing forms is confusing.
The array form also makes the "pack args into a list" pattern uniform.

### 5.2 `quantizedMM` inside `compile` — **#1 risk, verify first**

`quantizedMM` calls the C `mlx_quantized_mm` API, which creates a traceable MLX
op. The Python `mx.quantized_matmul` works inside `mx.compile`. Swift's
`quantizedMM` should too, but **this must be verified with a spike test before
building the full plan** (Phase 0 below). If it doesn't trace, the entire
approach is blocked and we'd fall back to the current qmv + separate-compile
path.

### 5.3 `MLXFast.RoPE` inside `compile` — **#2 risk, verify first**

The Python `mx.fast.rope` works inside `mx.compile`. Swift's `MLXFast.RoPE`
should too. Verify in the same spike test.

### 5.4 `eps` and `bits` as compile-time constants

- `eps`: Already baked as `kRMSEps = 1e-6` (all layers assert match). Pass as a
  captured `Float` constant inside the `compile` closure — `MLXFast.rmsNorm(x,
  weight: w, eps: kRMSEps)`.
- `bits`: The `quantizedMM` `bits` parameter is an `Int`. Inside `compile`, it's
  captured as a constant. Different bit-width combos need different compiled
  functions (cached by `(mlpBits, pleBits)`). This matches the Python factory
  approach.

### 5.5 `shapeless` — same constraint as Python

- Pre-attention: **cannot** use `shapeless: true` (reads `.shape` for tensor
  slicing/reshaping — MLX can't infer slice output shapes with unknown dims).
- Post-attention: *could* use `shapeless: true` (no slicing) but Python found it
  causes a decode regression (~128 vs 155 tok/s) because the generic shapeless
  kernel is less optimized. **Use per-shape `compile` (default, `shapeless:
  false`) for all three.**

### 5.6 `offset` must be an `MLXArray`, not an `Int`

The Python Phase 7 found a critical bug: `offset=0` (Int, compile-time constant)
during precompilation vs `offset=mx.array(cache.offset)` (0-d array, runtime
input) during real generation compiled different versions. **Always use
`MLXArray` for offset** (e.g., `MLXArray(0)` or `MLXArray(cache.ropeOffset)`),
never a bare `Int`, so `compile` treats it as a runtime input.

### 5.7 SRQ precision

The Python `_srq` does all SRQ math in `float32` (`x.astype(mx.float32)`). The
Swift `applySRQ` uses `scale.asType(x.dtype)` (input dtype, which may be
`bfloat16`). For the compiled path, add a **float32 SRQ variant** that matches
the Python precision:

```swift
/// Compile-friendly SRQ in float32 (matches Python _srq and the qmv kernel).
private func srqF32(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
    let s32 = s.asType(.float32)
    let isZero = s32 .== 0
    let safe = MLX.where(isZero, MLXArray.ones(like: s32), s32)
    let q = MLX.clip(MLX.round(x.asType(.float32) / safe), min: -128, max: 127) * safe
    return MLX.where(isZero, x, q)
}
```

Use `srqF32` inside the compiled segments (not `applySRQ` which uses input
dtype).

### 5.8 `quantizedMM` `transpose` parameter

The Python `mx.quantized_matmul(h, wq, scales, biases, group_size=128, bits=N)`
defaults to `transpose=True` (weights stored as `[out, in]`, matmul does
`x @ w.T`). Swift's `quantizedMM` has `transpose: Bool = true` — same default.
The `mobileToMLX` output is `[out, n_uint32]` which matches `transpose: true`.

---

## 6. Implementation phases

### Phase 0 — Spike: verify `quantizedMM` + `MLXFast.RoPE` inside `compile` ✅ DONE

**Goal:** De-risk the two biggest unknowns before building anything.

**Result:** All 3 tests pass (`swift test --filter CompileQuantizedMMSpike`):
- ✅ `quantizedMM` + `MLXFast.rmsNorm` + SRQ fuse correctly inside `compile` — matches eager path for shapes `[1,4,128]`, `[1,1,128]`, `[2,8,128]` (max diff < 1e-4).
- ✅ `MLXFast.RoPE` with `MLXArray` offset traces and executes inside `compile` — matches eager path for shapes `[1,8,4,256]` (offset=0) and `[1,8,1,256]` (offset=5) (max diff < 1e-5).
- ✅ Mini pre-attention segment (rmsNorm → SRQ → quantizedMM → SRQ → reshape → rmsNorm → transpose → RoPE) compiles and matches eager path for shapes `[1,4,128]` and `[1,1,128]` (max diff < 1e-4).

**Conclusion:** Both `quantizedMM` and `MLXFast.RoPE` are compile-friendly in Swift. The compiled-fusion approach is feasible. Proceeding to Phase 1.

Create a throwaway test (`Tests/MLXLMTests/CompileQuantizedMMSpike.swift`):
1. Create tiny mobile-format weights (e.g., `[8, 128]` int4 + scale), convert
   via `mobileToMLX`.
2. Wrap `quantizedMM` + `MLXFast.rmsNorm` + `srqF32` in `compile { ... }`.
3. Call with two different shapes (e.g., `[1, 1, 128]` and `[1, 4, 128]`).
4. `eval` the output. Verify: (a) no crash, (b) output is correct (matches
   eager `quantizedMM`), (c) the second call with the same shape doesn't
   recompile (check it's fast).
5. Same for `MLXFast.RoPE` inside `compile` with an `MLXArray` offset.

**Exit criteria:** Both `quantizedMM` and `MLXFast.RoPE` trace and execute
correctly inside `compile`. If either fails, stop and reassess (the fallback is
the current qmv + separate-compile path, which already works).

### Phase 1 — Weight extraction + SRQ helper

1. Add `srqF32` to `GemmaMobileQuantization.swift` (compile-friendly float32 SRQ).
2. Add `nativeArgs()` accessor to `GemmaQuantizedLinear` (returns
   `(packed, scales, biases, inScale, outScale)`, cached).
3. Add `buildFusedQKVNative(attn:)` — concatenates q/k/v native weights +
   per-row output SRQ scale (mirrors Python `_build_fused_qkv_native`).
4. Add `getRopeFreqs(attn:)` — extracts rope freqs from `RoPELayer`.

**Test:** Unit test `nativeArgs()` bit-exactness (already covered by existing
`mobileToMLX` tests, but add a test that goes through `GemmaQuantizedLinear`).

### Phase 2 — Compiled pre-attention segments

1. Add `CompiledPreAttnSourceCache` (key: `(nHeads, headDim, nKvHeads)`) with
   the factory compiled function (array form, ~12 inputs).
2. Add `CompiledPreAttnKvsharedCache` (key: `(nHeads, headDim)`) with the
   factory compiled function (array form, ~10 inputs).
3. Both use `srqF32`, `MLXFast.rmsNorm(eps: kRMSEps)`, `quantizedMM(bits: 4,
   groupSize: 128)`, `MLXFast.RoPE(offset:, freqs:)`.

**Test:** Create a `GemmaQuantizedLinear` layer, extract args, call the
compiled pre-attn function, compare output to the eager path (input_layernorm →
qProj → norms → rope). Verify bit-exactness within float32 tolerance.

### Phase 3 — Compiled post-attention segment

1. Add `CompiledPostAttnCache` (key: `(mlpBits, pleBits)`) with the factory
   compiled function (array form, ~37 inputs).
2. Uses `srqF32`, `MLXFast.rmsNorm`, `quantizedMM(bits: mlpBits/pleBits)`,
   `geluApproximate`, `layerScalar`.

**Test:** Compare compiled post-attn output to the eager path (oProj → norm →
residual → mlp → norm → residual → PLE → norm → residual → layerScalar).

### Phase 4 — Wire into decoder layer ✅ DONE

1. ✅ Added `getNativeArgs()` to `Gemma4DecoderLayer` — lazily extracts + caches
   `(isSource, preFn, preArgs, postFn, postArgs)` or returns `nil` (not usable:
   MoE, non-gemma-quant, unaligned dims, no PLE).
2. ✅ Added the native compiled path to `Gemma4DecoderLayer.callAsFunction`
   (before the existing eager path). Handles KV-shared vs source layers. Keeps
   the eager path as fallback (`eagerCall`).
3. ✅ `offset` as `MLXArray` (never bare `Int`).
4. ✅ Fixed the A/B equivalence failure: the eager path's `CompiledSRQ` used
   bfloat16 SRQ while the native path used float32 SRQ. Updated `CompiledSRQ`
   and `GemmaQuantizedLinear.callAsFunction` (batch > 16) to use float32 SRQ +
   float32 `quantizedMM` input + cast back to original dtype, matching the native
   path's dtype flow. See §1a for details.

**Test:** ✅ `nativeCompiledPathMatchesEager` — 32-token fixed prompt, mean abs
   diff 0.30, 2/32 argmax mismatches (within tolerance: mean < 0.5, argmax ≤ 4).

### Phase 5 — Load-time precompilation + weight freeing

1. Add `precompileNativeFunctions(model:)` — layer-by-layer weight conversion +
   freeing + hybrid compile strategy (full pass ≤ 32, direct call > 32), warming
   `(1, 16, 32, 64, 128, 256, 512)` and every unique E2B source/shared +
   sliding/full pre- and post-attention signature.
2. Hook into the loading flow (after `update(parameters:)`). Use a
   `Gemma4NativePrecompilable` protocol or type check in `Load.swift`.
3. Free mobile weights after conversion (replace `weight`/`weightScale` with
   dummy arrays). Preserve SRQ scales (still needed).

**Test:** Verify representative selection covers all four E2B attention
combinations, peak memory is reduced (mobile weights freed), and first-run
prefill is stable (not variable like the no-precompile path).

### Phase 6 — Benchmark & validate

1. Benchmark: decode tok/s, prefill tok/s, peak memory, steady-state memory.
2. Compare to the current (qmv-based) path. Expect: decode +10%, prefill
   +100%+ (stable, not variable).
3. Run all existing tests (17 unit + integration). Add new tests for the
   compiled segments.
4. Cross-validate output with the Python `mlx-vlm` model (same prompt → same
   output, within float32 tolerance).

---

## 7. Testing & validation

### Unit tests (Phase 1–3)

| Test | What it verifies |
|---|---|
| `srqF32` matches `applySRQ` for calibrated layers | SRQ precision |
| `nativeArgs()` bit-exact vs `dequantizeWeight` | Weight conversion |
| `buildFusedQKVNative` concatenation correct | Fused qkv weights |
| Compiled pre-attn source output ≈ eager path | Fusion correctness |
| Compiled pre-attn kvshared output ≈ eager path | Fusion correctness |
| Compiled post-attn output ≈ eager path | Fusion correctness |
| `quantizedMM` inside `compile` (Phase 0 spike) | De-risking |

### Integration tests (Phase 4–5)

| Test | What it verifies |
|---|---|
| Model loads with compiled path | Module replacement + native args |
| Short prompt → coherent output | End-to-end correctness |
| Top-1 token matches eager path | Numerical equivalence |
| Prefill stable across runs (not variable) | Precompilation works |
| Peak memory ≤ current + small delta | Weight freeing works |
| All 17 existing unit tests still pass | No regression |

### Benchmark (Phase 6)

Use the existing `BenchmarkHelpers` or a CLI generate:
```
swift test --filter Gemma4MobileIntegrationTests  # correctness
# + a benchmark harness measuring decode/prefill tok/s and peak memory
```

Compare to the current qmv-based path. **Honesty requirement:** the user
explicitly challenged overstated numbers from a previous agent. Report 3-run
averages, note variance, and don't claim improvements without evidence.

---

## 8. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `quantizedMM` doesn't trace inside `compile` | Low | **Blocking** | Phase 0 spike verifies before building anything. Fallback: current qmv path (already works). |
| `MLXFast.RoPE` doesn't trace inside `compile` | Low | **Blocking** | Phase 0 spike. Fallback: keep RoPE eager (outside the compiled segment). |
| Post-attn arg count exceeds `compile` limits | None | N/A | Use the `[MLXArray]` array form (no arg limit). |
| `quantizedMM` `bits` not captured as constant | Low | Medium | If `compile` can't capture `Int` params, bake via a factory closure (one fn per bit combo). |
| Memory spike during precompilation | Medium | Medium | Hybrid compile strategy (full pass ≤ 32, direct call > 32) + layer-by-layer freeing (proven in Python). |
| Decode regression from per-shape recompile | Medium | Medium | Load-time precompilation for common shapes (1, 16, 32, 64, 128, 256, 512) and every E2B source/shared + sliding/full signature. |
| `shapeless: true` temptation | None | N/A | Don't use it (Python confirmed it causes decode regression + can't slice). |
| `offset` Int vs MLXArray mismatch | Medium | High (garbage output) | Always use `MLXArray` for offset (Python Phase 7 bug). |
| MoE layers (E-series) | None | N/A | Guard: native path only for non-MoE. MoE falls back to eager. |
| `lm_head` (2-bit, out=262144) | None | N/A | Stays on the qmv/eager path (not a decoder layer). Future: fused lm_head + top-k. |

---

## 9. File summary

| File | Action | Phase | Description |
|---|---|---|---|
| `Libraries/MLXLMCommon/GemmaMobileQuantization.swift` | **Modify** | 1 | Add `srqF32`, `nativeArgs()` accessor, `buildFusedQKVNative`, `getRopeFreqs`. |
| `Libraries/MLXLLM/Models/Gemma4Text.swift` | **Modify** | 2–5 | Add three compiled-segment caches, `getNativeArgs()` on `Gemma4DecoderLayer`, native compiled path in `callAsFunction`, `precompileNativeFunctions`. |
| `Libraries/MLXLMCommon/Load.swift` | **Modify** | 5 | Hook `precompileNativeFunctions` after `update(parameters:)`. |
| `Tests/MLXLMTests/CompileQuantizedMMSpike.swift` | **Create** | 0 | De-risking spike test. |
| `Tests/MLXLMTests/Gemma4MobileQuantizationTests.swift` | **Modify** | 1–3 | Add tests for `srqF32`, `nativeArgs`, compiled segments. |
| `Tests/MLXLMTests/Gemma4MobileIntegrationTests.swift` | **Modify** | 4–6 | Add compiled-path integration + benchmark tests. |

---

## 10. Implementation order (summary)

1. ✅ **Phase 0** — Spike: verify `quantizedMM` + `MLXFast.RoPE` inside `compile`.
2. ✅ **Phase 1** — `srqF32`, `nativeArgs()`, `buildFusedQKVNative`, `compiledRopeFreqs`.
3. ✅ **Phase 2** — Compiled pre-attention (source + KV-shared) factory + cache.
4. ✅ **Phase 3** — Compiled post-attention factory + cache.
5. ✅ **Phase 4** — `getNativeArgs()` + native path in `Gemma4DecoderLayer.callAsFunction` + A/B fix (float32 SRQ).
6. ✅ **Phase 5** — Load-time precompilation + weight freeing + load hook (`NativePrecompilable` protocol, `precompileNativeFunctions`, `freeMobileWeights`, `precompileAtLoad` A/B flag).
7. ⬜ **Phase 6** — Benchmark, validate, cross-check with Python.

---

## 11. References

- Python Phase 6/7 implementation: `mlx-vlm/mlx_vlm/models/gemma4/language.py`
  - `_srq` (line ~76), `_qlinear_native_args` (line ~90),
    `_get_rope_freqs` (line ~330), `_build_fused_qkv_native` (line ~350),
    `_get_compiled_pre_attn_source` (line ~380), `_get_compiled_pre_attn_kvshared`
    (line ~430), `_get_compiled_post_attn_mlp_ple` (line ~460),
    `DecoderLayer._get_native_args` (line ~882), `DecoderLayer.__call__` (line ~960),
    `precompile_native_functions` (line ~190).
- Python weight conversion: `mlx-vlm/mlx_vlm/quantization/gemma_mobile.py`
  (`mobile_to_mlx`).
- Python plan docs: `GEMMA4_QAT_MOBILE_LITERT_PLAN.md`, `GEMMA4_QAT_MOBILE_PERF_PLAN.md`.
- Swift port plan: `GEMMA4_QAT_MOBILE_SWIFT_PORT_PLAN.md` (the initial port).
- Swift porting guide: `Libraries/MLXLMCommon/Documentation.docc/porting.md`.
- MLX-Swift `compile`: `.build/checkouts/mlx-swift/Source/MLX/Transforms+Compile.swift`.
- MLX-Swift `quantizedMM`: `.build/checkouts/mlx-swift/Source/MLX/Ops.swift` (~line 2433).
- MLX-Swift `MLXFast`: `.build/checkouts/mlx-swift/Source/MLX/MLXFast.swift`.
