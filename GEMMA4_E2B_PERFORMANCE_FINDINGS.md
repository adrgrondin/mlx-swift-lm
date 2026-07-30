# Gemma 4 QAT mobile performance and compatibility findings

## Final decision

The retained optimization for the official checkpoint is deliberately generic:

```text
/Users/adrgrondin/Workspace/mlx-vlm/gemma-4-E2B-it-qat-mobile-mlx-mm
```

The implementation compiles three semantics-preserving elementwise fragments in the shared VLM Gemma 4 decoder:

- Residual + RMSNorm.
- GELU-approximate × the second gated input.
- PLE residual + RMSNorm + layer scalar.

There is **no model-ID, parameter-count, hidden-size, layer-count, E2B, or E4B dispatch condition**. In particular, the rejected implementation artifacts are gone:

- `gemma4IsOfficialE2BArchitecture`
- `enableE2BMobileOptimizations`
- `nativePostOptimizationSelected`
- the layers 30–34 native-tail selection
- `MLX_LM_NO_GEMMA_NATIVE_POST`

The RMSNorm helpers are used when the decoded `rms_norm_eps` equals the compiled constant `1e-6`; another epsilon uses the existing eager expression. GELU × multiply is shape-independent and is compiled for every dense Gemma 4 MLP. This follows the existing `MLXLLM` Gemma 4 implementation rather than introducing a checkpoint-specific policy.

Gemma mobile quantization remains independently format-driven. `GemmaQuantizedLinear` and `GemmaQuantizedEmbedding` replacement occurs only when `quantization_config.quant_method == "gemma"`, and only for modules supported by the checkpoint regexes and scale tensors. Ordinary affine quantization continues through the generic loader.

The retained memory correction is also format-driven. It reuses the mobile checkpoint's already-compatible 2/4-bit packed storage as the MLX `uint32` quantizedMM input and makes fused decode Q/K/V read the three source projections directly. It does not dispatch on E2B/E4B architecture or model identity.

## Mobile-QAT first-prompt memory correction

### Root cause

The official checkpoint loaded at 2,274,029,922 active MLX bytes, then retained 3,082,574,718 bytes after its first prompt: an 808,544,796-byte increase. KV cache and short-prompt activations were not large enough to explain it.

The increase came from two persistent conversions:

1. `mobileToMLX` rebuilt every aligned 2/4-bit mobile linear into a second packed `uint32` tensor for prefill `quantizedMM`. The mobile byte layout is already the same little-endian LSB-first bit layout expected by MLX; arithmetic byte recombination duplicated the packed weights without changing their bits. The original weights had to remain resident because decode QMV reads the mobile representation directly.
2. Fused Q/K/V decode concatenated Q, K, and V packed weights for every KV-owning layer and retained those concatenations after the first decoded token.

The checkpoint contains about 916.5 MiB of packed linear weights. The first conversion duplicated most of that storage; temporary conversion buffers also enlarged MLX's reusable buffer cache.

### Retained implementation

- For 2-bit and 4-bit aligned linears, `mobileToMLX` uses `weight.view(dtype: .uint32)` and reshape rather than arithmetic byte reconstruction. The quantizedMM input and decode QMV input now share the original packed allocation.
- Int8 conversion remains explicit because signed two's-complement bytes cannot be represented by one affine bias in MLX's unsigned 8-bit quantizedMM format.
- Fused Q/K/V remains one Metal kernel launch, but the kernel selects Q, K, or V from the three original buffers per output row. It also selects each projection's scalar output SRQ scale directly, preserving the per-projection scale correctness fix without a per-row concatenated scale tensor.
- The same generic quantization primitives are used by LLM and VLM Gemma-mobile paths. Ordinary affine Gemma 4 and non-Gemma models never enter these functions.

This follows the useful LiteRT-LM patterns rather than copying a backend-specific implementation: external/source quantized weights are supplied to compilation without unconditional host duplication, source-quantized FC/conv operations can remain enabled, constants can be shared, and weight/program conversion caches are managed separately. Relevant references are:

```text
/Users/adrgrondin/Workspace/LiteRT-LM/runtime/executor/llm_executor_settings_utils.cc
/Users/adrgrondin/Workspace/LiteRT-LM/runtime/executor/llm_litert_compiled_model_executor.cc
/Users/adrgrondin/Workspace/LiteRT-LM/runtime/executor/litert_compiled_model_executor_utils.cc
```

In particular, LiteRT-LM enables source-quantized FC/conv operations and constant sharing, optionally converts weights on GPU, and passes external weight sections by file descriptor into compilation. The MLX correction applies the same ownership principle locally: do not retain an identical packed constant solely to satisfy a different tensor dtype.

### Memory benchmark

All memory comparisons used the release harness in [BENCHMARKING.md](BENCHMARKING.md), the production auto/VLM path, 64 output tokens, one excluded shape warmup, and the official local mobile checkpoint. Values are MLX active/peak allocation, not process RSS.

| Workload | Previous peak | Selected peak | Reduction |
|---:|---:|---:|---:|
| 128 prompt tokens | 3,281,345,309 B | 2,510,404,891 B | **770,940,418 B (735.2 MiB, 23.49%)** |
| 512 prompt tokens | 3,852,171,849 B | 3,081,246,471 B | **770,925,378 B (735.2 MiB, 20.01%)** |
| 2048 prompt tokens | 3,910,299,403 B | 3,139,391,753 B | **770,907,650 B (735.2 MiB, 19.71%)** |

Steady active allocation before a measured 128-token run fell from 3,082,574,718 to 2,311,674,750 bytes, only 37,644,828 bytes above the 2,274,029,922-byte post-load state. The packed view accounts for 735.3 MB of the reduction and direct-buffer fused Q/K/V saves another 35.6 MB.

An app-like natural 40-token prompt with a 20 MiB MLX cache limit peaked at 2,384,931,662 bytes (2.221 GiB). The ordinary affine `mlx-community/gemma-4-E2B-it-qat-4bit` reference peaked at 3,846,606,871 bytes for the 128-token workload, so selected mobile QAT is 1,336,201,980 bytes (1,274.3 MiB, 34.74%) lower instead of converging on the same peak.

### Performance and correctness gate

The packed view is bit-exact and changes only conversion/ownership; the evaluated quantizedMM tensor has the same shape, dtype, and bytes as before. Existing int2/int4 conversion tests pass against dequantized reference values.

A controlled 512-token A/B compared the old concatenated fused-QKV cache with the direct-buffer fused kernel:

| Variant | Prefill | Decode | 512-token peak |
|---|---:|---:|---:|
| Concatenated Q/K/V control | 10,256.20 tok/s | 116.13 tok/s | 3,116,827,110 B |
| Direct-buffer Q/K/V | 10,240.01 tok/s | 115.92 tok/s | 3,081,241,062 B |
| Delta | -0.16% | **-0.18%** | **-35,586,048 B** |

The throughput difference is below the observed frequency/run variance, so the direct-buffer kernel is retained. The temporary control function and dispatch flag were removed.

A final post-cleanup Release confirmation used one warmup and three measured runs per shape. Every measured run generated 64 tokens and stopped at the requested length; low-power mode was off and the reported thermal state was nominal.

| Prompt | Final median prefill | Final median decode | Final median peak | Active before measured runs |
|---:|---:|---:|---:|---:|
| 128 | 3,850.68 tok/s | 119.65 tok/s | 2,510,404,891 B | 2,311,674,750 B |
| 512 | 10,281.12 tok/s | 115.89 tok/s | 3,081,245,751 B | 2,311,676,286 B |
| 2048 | 15,632.26 tok/s | 115.21 tok/s | 3,139,391,753 B | 2,311,686,526 B |

The final peaks reproduce the selected candidate within 720 bytes. These absolute throughput values were collected after the preceding test/build workload and are confirmation data, not a replacement for the adjacent controlled Q/K/V performance A/B.

Benchmark reports:

```text
.../scratchpads/dm1/mobile-memory-baseline.json
.../scratchpads/dm1/mobile-memory-packed-view.json
.../scratchpads/dm1/mobile-memory-split-qkv.json
.../scratchpads/dm1/mobile-memory-app-like.json
.../scratchpads/dm1/mobile-memory-final-confirmation.json
.../scratchpads/dm1/affine-qat-memory-reference.json
.../scratchpads/dm1/qkv-concatenated-control.json
.../scratchpads/dm1/qkv-split-candidate.json
```

## Why the earlier E2B-only implementation was rejected

The earlier candidate selected an exact E2B architecture and enabled a native `quantizedMM` post-attention graph only for decoder layers 30–34. That was not a legitimate architecture capability:

- The 30–34 cutoff came from an E2B quality sweep, not a Gemma 4 configuration field.
- It could not generalize naturally to E4B's 42-layer topology.
- It was unlike the repository's established format/capability dispatch.
- Extending the native path to every eligible E2B layer changed decode top-1 results too often.

That implementation and its tests were removed rather than renamed or expanded into another model-size table.

## Dispatch and model-family review

The Swift VLM implementation was reviewed against:

```text
/Users/adrgrondin/Workspace/mlx-vlm/mlx_vlm/models/gemma4/config.py
/Users/adrgrondin/Workspace/mlx-vlm/mlx_vlm/models/gemma4/language.py
/Users/adrgrondin/Workspace/mlx-vlm/mlx_vlm/models/gemma4/gemma4.py
/Users/adrgrondin/Workspace/mlx-vlm/mlx_vlm/models/gemma4_unified/config.py
/Users/adrgrondin/Workspace/mlx-vlm/mlx_vlm/models/gemma4_unified/gemma4_unified.py
```

The production implementation selects behavior from decoded topology and runtime module capabilities:

| Feature | Configuration/capability used |
|---|---|
| Decoder depth and dimensions | `num_hidden_layers`, hidden/intermediate sizes, head counts/dimensions |
| Sliding/full attention | Explicit `layer_types`, otherwise the decoded pattern |
| E-series KV sharing | `num_kv_shared_layers` and same-attention-type source mapping |
| Global K=V attention | `attention_k_eq_v` and `num_global_key_value_heads` |
| Per-layer embeddings (PLE) | `hidden_size_per_layer_input > 0` and the corresponding modules |
| Double-wide E-series MLP | `use_double_wide_mlp` plus whether the layer is KV-shared |
| MoE | `enable_moe_block`, expert count, top-k, and MoE intermediate size |
| Tied/untied output | `tie_word_embeddings` and presence of `lm_head` |
| Mobile-QAT module replacement | `quant_method: gemma`, module regexes, exclusions, and scale tensors |
| Fused mobile QKV | Actual `GemmaQuantizedLinear` modules, QMV eligibility, matching dimensions/bits/input scales |
| Compiled RMSNorm fragments | `rms_norm_eps == 1e-6`; eager fallback otherwise |

No exact architecture is used as an inference-kernel switch.

### Family matrix

| Family | Expected path | Important decoded features | Quantization behavior |
|---|---|---|---|
| Official E2B mobile QAT | `gemma4` → `MLXVLM.Gemma4` | 35 layers, 20 KV-shared, PLE, dense, untied head | `quant_method: gemma`; mixed 2/4/8-bit mobile modules |
| Official E4B mobile QAT | `gemma4` → `MLXVLM.Gemma4` | 42 layers, 18 KV-shared, PLE, dense, untied head | Same generic mobile replacement; published config resolves E4B MLP/attention to 4-bit and PLE linears to 8-bit |
| Standard E2B/E4B affine | `gemma4` → `MLXVLM.Gemma4` | E-series KV sharing/PLE from config | Generic `QuantizedLinear`/`QuantizedEmbedding`; never Gemma-mobile modules without `quant_method: gemma` |
| 12B QAT 4-bit | `gemma4_unified` → `MLXVLM.Gemma4Unified` | 48 layers, no KV sharing, no PLE, dense, global K=V | Ordinary affine mixed precision: 4-bit default and 8-bit MLP overrides |
| 31B | `gemma4` → `MLXVLM.Gemma4` | 60 layers, no KV sharing/PLE, dense, global K=V | Checkpoint-defined generic quantization |
| 26B-A4B | `gemma4` → `MLXVLM.Gemma4` | 30 layers, no KV sharing/PLE, 128 experts/top-8, global K=V | Checkpoint-defined generic quantization; MoE modules selected from config |

### E4B-specific check

The locally cached official E4B mobile config decodes as:

- Hidden size 2560.
- 42 decoder layers.
- Intermediate size 10240.
- 8 query heads and 2 KV heads.
- 18 KV-shared layers, so the shared tail starts naturally at layer `42 - 18 = 24`.
- PLE size 256.
- Dense, non-K=V E-series attention.
- Untied token embedding/output head.
- `quant_method: gemma`.

The implementation does not need to know that this model is called E4B. Its different topology and module bit patterns are consumed directly from the config. E4B weights are not present in the local snapshot, so full E4B mobile load/generation remains unverified; config decoding and quantization resolution pass.

## Python/Swift architecture parity findings

- **Layer layout:** explicit `layer_types` are consumed directly. Generated patterns are configuration-based.
- **Attention geometry:** full attention uses `global_head_dim`; sliding attention uses `head_dim`.
- **K=V:** full-attention K=V layers omit `v_proj`, derive values from raw K, apply K norm only to keys, and apply no-scale V norm to values.
- **KV sharing:** E-series shared layers reuse the latest earlier KV-owning layer of the same attention type. Shared layers declare no local K/V projection or K norm, and only KV-owning layers allocate caches.
- **Dense MLP:** GELU-approximate gated MLP and E-series double-wide rules match the Python implementation.
- **MoE:** the dense and sparse branches, router normalization/scaling, top-k selection/renormalization, SwitchGLU experts, expert weighting, and gate/up checkpoint splitting match the reference.
- **PLE:** token lookup, reshape, embedding/projection scaling, RMS normalization, `1/sqrt(2)` combination, per-layer gate/projection, residual, and layer scalar match.
- **Output:** tied models use the token embedding as a linear head. Official mobile E2B/E4B retain their independent quantized `lm_head`.
- **Softcapping:** logits use `tanh(logits / cap) * cap` when configured.
- **Sanitization:** rotary tensors are dropped, text KV-shared redundancies are removed without affecting tower keys, namespaces are normalized, MoE gate/up weights are split, and tied checkpoints discard separate LM-head weights.
- **Quantization separation:** top-level ordinary affine `quantization` and mobile `quantization_config.quant_method: gemma` remain distinct loading paths.

## Retained performance result

### Method

The acceptance comparison uses the release-mode harness documented in [BENCHMARKING.md](BENCHMARKING.md):

- Production auto/VLM factory path.
- Exact 128, 512, and 2048-token prompts.
- Greedy sampling (`temperature = 0`).
- 64 generated tokens.
- One excluded warmup and five measured runs per prompt length.
- Same model, cache, and generation settings.
- Low-power mode off; benchmark-reported thermal state nominal.
- Every measured run emitted 64 tokens with `stop_reason: length`.

The control temporarily changed only the three elementwise sites back to their eager expressions. Production source was restored before the reverse-order candidate run.

### Reverse-order confirmation

| Prompt | Eager prefill tok/s | Compiled prefill tok/s | Prefill delta | Eager decode tok/s | Compiled decode tok/s | Decode delta | Peak-memory delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4,921.54 | 5,116.74 | **+3.97%** | 147.20 | 151.66 | **+3.03%** | +6,680,787 B |
| 512 | 12,902.26 | 13,235.12 | **+2.58%** | 141.93 | 147.19 | **+3.70%** | +41,976,769 B |
| 2048 | 19,231.51 | 19,616.85 | **+2.00%** | 143.09 | 145.98 | **+2.01%** | +19,450,336 B |

Decision: **keep the generic compiled elementwise helpers**. They improve both prefill and decode in the causal reverse-order comparison. The MLX compile/cache cost increases peak allocation by approximately 6.4–40.0 MiB depending on prompt shape.

An earlier 512-token cooled comparison measured 127.66 → 130.21 decode tok/s (**+2.00%**) and 11,505.90 → 11,659.92 prefill tok/s (**+1.34%**), independently supporting the decision.

### Frequency-drift warning

A preceding candidate process measured 130–135 decode tok/s; the subsequent eager process measured 142–147 tok/s even though both reported `thermal_state: nominal`. Those cross-process absolute rates were not used as the performance delta. The immediately following reverse-order candidate reached 146–152 tok/s and produced the table above. macOS's coarse thermal label does not capture all power/frequency state, so short adjacent/reverse-order comparisons remain necessary.

Benchmark reports:

```text
.../scratchpads/dm1/e2b-generic-helpers-control-full.json
.../scratchpads/dm1/e2b-generic-helpers-reverse-full.json
.../scratchpads/dm1/e2b-generic-helpers-full.json
.../scratchpads/dm1/e2b-control-512.json
.../scratchpads/dm1/e2b-compiled-elementwise-512.json
```

## Commit `98af599` versus `HEAD` and working tree

A second comparison investigated whether commit `98af599d7fe35ca5afc192c9c83fe38c62c6a47f` (`Native MLX quantized_matmul`) was faster than `HEAD` (`afeef922596c921b1fb81a3295403651351839ee`) or the current working tree.

### Production auto/VLM result

All variants used the exact official local checkpoint, release builds, greedy generation, 64 output tokens, one excluded warmup, and five measured runs per prompt length. Each variant had an isolated Xcode derived-data directory. Builds were completed before the accepted comparison, processes were separated by cooldowns, low-power mode was off, thermal state was nominal, and every measured run ended with `64:length`.

| Prompt | `98af599` prefill/decode | `HEAD` prefill/decode | Working tree prefill/decode | `98af` decode vs `HEAD` | Working-tree decode vs `HEAD` |
|---:|---:|---:|---:|---:|---:|
| 128 | 4,329.59 / 132.63 | 4,448.00 / 132.46 | 4,603.34 / 135.84 | +0.13% | **+2.55%** |
| 512 | 11,597.33 / 128.76 | 11,546.36 / 128.78 | 11,735.30 / 131.65 | -0.01% | **+2.23%** |
| 2048 | 17,214.85 / 128.32 | 17,145.68 / 127.88 | 17,475.73 / 131.24 | +0.34% | **+2.62%** |

`98af599` and `HEAD` are effectively tied: the decode differences are between -0.01% and +0.34%, below the observed process-to-process variance. The working tree is consistently faster than `HEAD` at all three prompt lengths. Its prefill gains are +3.49%, +1.64%, and +1.92% respectively, with the previously documented MLX compile/cache allocation cost.

A separate cooled 512-token palindrome (`98af` → `HEAD` → working tree → working tree → `HEAD` → `98af`) produced these averages of the two process medians:

| Variant | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| `98af599` | 11,544.90 | 128.61 |
| `HEAD` | 11,574.62 | 128.87 |
| Working tree | **11,760.54** | **131.02** |

This independently confirms that `98af599` is not faster on the production factory path and that the generic working-tree helpers provide the retained gain.

### Why `98af599` could appear faster

The VLM `Gemma4.swift` source is byte-identical between `98af599` and `HEAD`. The later LLM native-precompile commits therefore do not execute when `--model-kind auto` resolves this checkpoint as `vlm`.

The one shared-kernel behavior change relevant to VLM is commit `0606a78` (`Fix fused QKV per-row output scaling`). `98af599` constructs a concatenated per-row output-scale vector for Q, K, and V, but its Metal kernel always reads `output_scale[0]`. That applies Q's static output scale to every K and V row. `HEAD` selects `output_scale[output_row]` for the fused-QKV specialization while retaining scalar scale lookup for ordinary QMV.

The old scalar read could theoretically save a tiny indexed lookup, but the cooled benchmark shows no measurable throughput cost for the correction. More importantly, the old operation is numerically wrong whenever Q/K/V output scales differ. The focused `fusedQKVUsesPerRowOutputScale` regression passes on the current implementation and checks the fused output against three separate projections with deliberately different scales.

The initial uncool comparison was misleading: its reports were `fair`, and decode declined within individual processes from approximately 119 to 89 tok/s at 512 tokens after three clean Xcode builds. Those reports were rejected rather than used to rank commits.

### Forced text-LLM result

For completeness, `98af599` and `HEAD` were also compared with `--model-kind llm`. The current working tree does not modify this LLM implementation, so it is source-equivalent to `HEAD` for this forced path. A cooled 512-token palindrome gave:

| Variant | Prefill tok/s | Decode tok/s | Measured peak | Reported model-weight bytes |
|---|---:|---:|---:|---:|
| `98af599` | **5,418.16** | 155.91 | 4,901,176,597 B | 2,087,286,934 B |
| `HEAD` | 5,349.75 | **156.22** | **4,236,805,902 B** | **1,423,229,391 B** |
| `HEAD` vs `98af` | -1.26% | +0.19% | -664,370,695 B | -664,057,543 B |

This is the only repeatable sense in which `98af599` was slightly faster: forced-LLM prefill was about 1.3% higher. The native inference graph itself is unchanged. Later commits precompile common shapes at load and free redundant mobile-format weights after native conversion, which changes compile/allocation state and recovers about 634 MiB of measured peak memory. Decode is not slower. This does not justify reverting the precompile work, and it does not affect the official checkpoint's production auto/VLM path.

Accepted comparison reports:

```text
.../scratchpads/dm1/compare-cooled-full-98af.json
.../scratchpads/dm1/compare-cooled-full-head.json
.../scratchpads/dm1/compare-cooled-full-worktree.json
.../scratchpads/dm1/compare-pal-1-98af.json
.../scratchpads/dm1/compare-pal-2-head.json
.../scratchpads/dm1/compare-pal-3-worktree.json
.../scratchpads/dm1/compare-pal-4-worktree.json
.../scratchpads/dm1/compare-pal-5-head.json
.../scratchpads/dm1/compare-pal-6-98af.json
.../scratchpads/dm1/compare-llm-pal-1-98af.json
.../scratchpads/dm1/compare-llm-pal-2-head.json
.../scratchpads/dm1/compare-llm-pal-3-head.json
.../scratchpads/dm1/compare-llm-pal-4-98af.json
```

Decision: **keep `HEAD` and the working-tree helpers**. Do not recover the old fused-QKV scalar-scale behavior.

## MLXChatExample throughput discrepancy

The approximately 132.63 tok/s figure was the 128-token case from the cooled `98af599` commit-comparison process. It used an exact 128-token `LMInput`, 64 forced output tokens, a shape-specific warmup, greedy sampling, and the median of five Release runs. It is not a universal application throughput target. The same official checkpoint has measured from the low 60s to the low 150s on this M5 Max as its GPU power/frequency state changed, sometimes while the benchmark-reported macOS thermal state remained `nominal`.

The implementation and application paths were checked directly:

- The MLXChat scheme in the examples worktree now launches the native macOS app with Xcode's `Release` configuration, whole-module compilation, and Metal fast math. A true Debug build measured only 43.57 tok/s, so an approximately 110 tok/s result is not explained by Debug overhead alone.
- MLXChat has a local package reference to this `mlx-swift-lm` worktree and its Release `MLXVLM` build records point to this checkout. However, the app's `Gemma4.o` was older than the final `Gemma4.swift` edit when the discrepancy was investigated. `MLXChatExample` was rebuilt successfully in Release, and the object now postdates the source.
- A disk checkpoint must be selected as a **Vision Model**. The picker entry named `gemma4:E2B:QAT` is `mlx-community/gemma-4-E2B-it-qat-4bit`, not `/Users/adrgrondin/Workspace/mlx-vlm/gemma-4-E2B-it-qat-mobile-mlx-mm`. Its loaded weight size is 3.46 GB rather than the official mobile checkpoint's 2.12 GB. It is therefore not an acceptable substitute even though a cooled 64-token trial happened to measure 172.02 tok/s.
- App benchmark mode uses natural chat-template prompts, one short-prompt warmup for the entire suite, up to 256 output tokens, five runs for each of three prompts, arithmetic means, and a 20 MiB MLX buffer-cache limit. The accepted commit comparison uses exact 128/512/2048-token inputs, a warmup for every shape, 64 output tokens, medians, and the default MLX cache limit.
- Chat mode additionally carries conversation history, samples at temperature 0.7, detokenizes text, and updates SwiftUI. These are production semantics, not the commit-comparison microbenchmark workload.

Controlled release experiments ruled out several suspected intrinsic penalties for the official local mobile checkpoint:

| Experiment | Decode result | Conclusion |
|---|---:|---|
| Natural 17-token app short prompt, 20 MiB cache, max 256 output | 133.64 tok/s median; model stopped after 8 tokens | The application cache limit and normal chat-template processor do not inherently reduce decode to 110. |
| Decoded-text `generate` stream, exact 128 input, 256 output | 133.69 tok/s median | Per-token detokenization/chunk production was not a measurable penalty for this workload. |
| Categorical sampling at temperature 0.7, exact 128 input, 64 output | 134.60 tok/s median | Sampling was only 0.6% below the adjacent 135.36 tok/s greedy control. |
| Fifteen consecutive 256-token runs from a cooled/high-frequency state | 132.22 tok/s median | A long suite can remain fast when it starts in the favorable hardware state. |
| Same sustained workload after earlier GPU-heavy work | 103.18 tok/s first run, 62.02 tok/s median | Prior GPU load can dominate the absolute result. |
| Natural medium prompt immediately after the sustained load | 86.75 tok/s first run, 83.63 tok/s median | Frequency recovery can lag process completion even when macOS reports `nominal`. |

The temporary raw/text and temperature changes used for those A/Bs were fully restored. Reports are retained in the session scratchpad as:

```text
.../scratchpads/dm1/app-short-raw-cache20.json
.../scratchpads/dm1/exact128-text-cache20.json
.../scratchpads/dm1/categorical-07-cooled.json
.../scratchpads/dm1/greedy-cooled-after-categorical.json
.../scratchpads/dm1/sustained-from-cooled.json
.../scratchpads/dm1/sustained-app-duration.json
.../scratchpads/dm1/app-medium.json
.../scratchpads/dm1/debug-build.json
.../scratchpads/dm1/app-builtin-qat-cooled.json
```

Conclusion: there is no reproduced 17% MLXChat code-path penalty. The actionable differences are checkpoint identity, a previously stale app package build, a non-equivalent workload/aggregation method, and highly variable GPU power/frequency state. For a valid application comparison, rebuild and relaunch the Release app, select `gemma-4-E2B-it-qat-mobile-mlx-mm (Disk, Vision)`, begin from the same cooled state, and compare the copied per-run prompt/output counts rather than only the displayed arithmetic average. An exact final diagnosis of a particular 110 tok/s app run requires that copied report because it identifies actual token counts and run-to-run drift.

## Rejected performance candidates

### E2B-only native post-attention tail

The former layers 30–34 candidate improved decode by approximately 2.6–3.0% over a helper-enabled eager control, but it was rejected because its layer range was an E2B-specific quality heuristic rather than a runtime capability.

Its 16-position forced-decode result was:

- Mean absolute logit difference: 0.4859806.
- Maximum absolute logit difference: 3.173555.
- Top-1 mismatches: 1/16.

Those numbers do not justify encoding the E2B layer count or tail indices into shared Gemma 4 code. The implementation, override seam, environment variable, and dedicated test were removed.

### Native post-attention graph on every E2B layer

This had higher throughput potential but failed the numerical gate against the production QMV decode path:

- Mean absolute logit difference: 0.6290375.
- Maximum absolute logit difference: 4.2224674.
- Top-1 mismatches: 6/16.

The primary difference was repeated QMV-versus-native-`quantizedMM` accumulation across all 35 layers, not a missing architecture check. A generic capability check alone would therefore not make this candidate safe.

### File-backed token and PLE embeddings

Mapping packed token/PLE tables and copying selected rows from CPU memory saved about 1.18 GiB of peak MLX allocation but regressed the cooled 512-token workload:

| Variant | Prefill tok/s | Decode tok/s | Peak MLX memory |
|---|---:|---:|---:|
| Resident control | 11,546.11 | 127.81 | 3,810,177,532 B |
| File-backed token + PLE | 10,501.71 | 89.02 | 2,548,476,490 B |
| Delta | -9.05% | **-30.35%** | -1,261,701,042 B |

Decision: **reject and remove**. LiteRT's dedicated external-weight embedding operators, NEON unpacking, prefetch, persistent buffers, and direct tensor writes do not map to CPU row extraction followed by a new MLX allocation every token.

### Gate/up QMV fusion

A no-copy paired gate/up kernel measured 124.15 versus 127.66 decode tok/s at 512 tokens (**-2.75%**) with no memory gain. A concatenated-weight form duplicated roughly 320 MB of packed weights and was thermally inconsistent.

Decision: **reject and remove**.

### Replacing the production VLM path with the LLM native path

Forcing the text-only native LLM path demonstrated much faster decode but slower prefill and higher long-prompt peak allocation. It was useful as an optimization reference, not as a replacement for the production multimodal factory path.

## Google QAT and LiteRT-LM reference findings

Google's Gemma 4 QAT announcement emphasizes:

- Static activation scales to remove dynamic-scale work.
- Channel-wise weight quantization for mobile accelerators.
- Targeted 2-bit compression where generation memory/bandwidth benefits most.
- Quantized embeddings and KV caches.
- QAT quality advantages over same-bit PTQ.

Reference: [Gemma 4 QAT models: Optimizing model compression for mobile and laptop efficiency](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/).

The local LiteRT-LM review found metadata/capability-driven backend policy, not an E2B/E4B kernel whitelist. Relevant patterns include:

- Separate prefill/decode execution and latency accounting.
- Model-packaged backend constraints and preferred activation types.
- Delegate capability and source-quantized FC/conv controls.
- Weight/program caches and GPU weight conversion.
- External-weight embedding lookup with platform-specific packed-row kernels.
- Persistent/shared buffers and hardware KV-cache update paths.

Relevant local files:

```text
/Users/adrgrondin/Workspace/LiteRT-LM/runtime/engine/engine_settings.cc
/Users/adrgrondin/Workspace/LiteRT-LM/runtime/executor/llm_executor_settings_utils.cc
/Users/adrgrondin/Workspace/LiteRT-LM/runtime/executor/llm_litert_compiled_model_executor.cc
/Users/adrgrondin/Workspace/LiteRT-LM/runtime/components/embedding_lookup/embedding_lookup_text.cc
```

LiteRT-LM does not provide a directly comparable Apple MLX benchmark or a reusable MLX kernel. Its external-embedding design is architectural guidance, not evidence that CPU mmap row copying is faster in this runtime.

## Compatibility validation

Focused tests passed:

- `Gemma4MobileVLMIntegrationTests`: 4/4.
  - Real official mobile E2B load and finite text forward.
  - Real standard affine E2B load with ordinary quantized modules.
  - Cached official E4B mobile config and module-bit resolution.
  - Processor-config fallback.
- `Gemma4ModelFamilyTests`: 4/4.
  - Public E4B topology config.
  - Public 31B dense config.
  - Public 26B-A4B MoE config.
  - Tiny dense and MoE module construction plus finite forwards.
- `Gemma4UnifiedTests`: 8/8, including the public 12B mixed-affine config fixture.
- `Gemma4MobileQuantizationTests`: 18/18.
- `Gemma4KVSharedLoadTests`: 2/2.
- `Gemma4TextTests`: 2/2, including K=V and quantized shared KV cache.
- `Gemma4AssistantDraftModelTests`: 8/8.

A release factory/generation smoke on the real cached `mlx-community/gemma-4-E2B-it-4bit` checkpoint also passed with the generic helpers enabled:

- Resolved as VLM.
- Loaded the 2.74 GB checkpoint.
- Generated all four requested tokens with `stop_reason: length`.
- Report: `.../scratchpads/dm1/standard-e2b-generic-helper-compat.json`.

`git diff --check` passes. The staged benchmark harness and its package/README changes were not edited during this correction.

### Validation limits

- The local E4B mobile cache contains config/tokenizer sidecars but no model weights, so real E4B mobile load/generation could not be run.
- Public 12B, 31B, and 26B-A4B weights are not locally cached. Their exact architecture-critical configs and synthetic topology/forward paths are covered, not full checkpoint loads.
- The regular `Gemma4` Swift VLM currently supports text and vision and filters regular audio-tower weights. The Python regular model supports audio. This is a pre-existing model capability gap, not introduced by the retained optimization.

## Remaining bottlenecks and next candidates

1. **QMV/native numerical alignment.** The native compiled graph has meaningful decode potential, but accumulation differences versus the production QMV path must be reduced before enabling it broadly. Fix the kernel-level discrepancy rather than selecting model layers.
2. **Stage-level profiling.** Add opt-in timing around embedding/PLE, attention projections, SDPA, MLP, cache update, and LM head. Aggregate prefill/decode numbers do not identify the next bottleneck reliably.
3. **Zero-copy/GPU embedding gather.** Revisit external packed embeddings only if MLX can import mapped buffers without copy or a GPU kernel can gather packed rows directly.
4. **QMV shape tuning.** Evaluate threadgroup/output tiling for 2-bit MLP, 4-bit attention, and 8-bit PLE separately. Do not hardcode one checkpoint's dimensions or accept a change without release A/B and numerical checks.
5. **E4B real-weight validation.** Run the same mobile replacement/load/forward and benchmark suite when E4B mobile weights are available.
6. **Large-family real loads.** Add cache-gated 12B/31B/26B-A4B load/generation tests as local or CI capacity permits.
7. **MTP/speculative decoding.** This is a separate checkpoint/runtime feature and should not be disguised as an E2B base-model optimization.
8. **Modality pruning as an explicit deployment mode.** Text-only deployment can save memory, but the official target here remains multimodal; modalities must not be silently removed.

## Current memory-correction worktree files

- [GEMMA4_E2B_PERFORMANCE_FINDINGS.md](GEMMA4_E2B_PERFORMANCE_FINDINGS.md)
- [Libraries/MLXLMCommon/GemmaMobileQuantization.swift](Libraries/MLXLMCommon/GemmaMobileQuantization.swift)
- [Libraries/MLXVLM/Models/Gemma4.swift](Libraries/MLXVLM/Models/Gemma4.swift)
- [Libraries/MLXLLM/Models/Gemma4Text.swift](Libraries/MLXLLM/Models/Gemma4Text.swift)
- [Tests/MLXLMTests/Gemma4MobileQuantizationTests.swift](Tests/MLXLMTests/Gemma4MobileQuantizationTests.swift)
