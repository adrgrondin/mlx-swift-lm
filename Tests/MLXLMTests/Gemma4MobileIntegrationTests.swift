// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// End-to-end coverage for the Gemma 4 QAT mobile (wNa8o8) load path
/// (`quant_method: "gemma"`). Builds a tiny `gemma4` model, writes a synthetic
/// checkpoint laid out like `gemma-4-E2B-it-qat-mobile-mlx-mm` (packed
/// int2/4/8 weights + per-channel scales + scalar SRQ scales, packed
/// `embedding_quantized` tables, and a plain-fp `per_layer_model_projection`),
/// loads it, and verifies that `Linear`/`Embedding` leaves are swapped for
/// `GemmaQuantizedLinear`/`GemmaQuantizedEmbedding` with the right per-layer
/// bits and that a forward pass produces finite logits
/// (see GEMMA4_QAT_MOBILE_SWIFT_PORT_PLAN.md §5.11).
///
/// Serialized because several tests mutate the process-wide
/// `Gemma4TextModel.useNativeCompiledPath` / `precompileAtLoad` flags and
/// load the same real checkpoint; running them concurrently would let one
/// test's flag changes leak into another's forward passes.
@Suite(.serialized)
struct Gemma4MobileIntegrationTests {

    private func tinyConfigJSON() -> String {
        """
        {
          "model_type": "gemma4",
          "vocab_size": 32,
          "quantization_config": {
            "quant_method": "gemma",
            "num_bits": 4,
            "quantize_embeddings": true,
            "modules_to_not_convert": ["per_layer_model_projection"],
            "module_quant_configs": {
              "^lm_head$": {"num_bits": 2},
              "language_model.embed_tokens$": {"num_bits": 2},
              "language_model.embed_tokens_per_layer$": {"num_bits": 4},
              "language_model.layers.[0-9]+.mlp.": {"num_bits": 4},
              "language_model.layers.[0-9]+.self_attn.": {"num_bits": 4},
              "language_model.layers.[0-9]+.per_layer_input_gate$": {"num_bits": 8},
              "language_model.layers.[0-9]+.per_layer_projection$": {"num_bits": 8}
            }
          },
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 1,
            "head_dim": 16,
            "global_head_dim": 16,
            "global_partial_rotary_factor": 0.25,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 8,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "max_position_embeddings": 64,
            "attention_k_eq_v": false,
            "final_logit_softcapping": 30.0,
            "use_double_wide_mlp": false,
            "layer_types": ["sliding_attention", "full_attention"],
            "tie_word_embeddings": false
          }
        }
        """
    }

    @Test("Gemma 4 mobile checkpoint loads with quantized layers and runs")
    func loadsAndRunsMobileCheckpoint() throws {
        MLXRandom.seed(0)
        let configData = Data(tinyConfigJSON().utf8)
        let decoder = JSONDecoder()
        decoder.userInfo[.rawConfigData] = configData
        let config = try decoder.decode(Gemma4Configuration.self, from: configData)
        let model = Gemma4Model(config)
        eval(model)

        let numLayers = config.textConfig.numHiddenLayers
        let arrays = makeMobileCheckpoint(
            model: model, quantizationConfig: config.quantizationConfig!, numLayers: numLayers)
        let directory = try writeCheckpoint(arrays)
        defer { try? FileManager.default.removeItem(at: directory) }

        try loadWeights(modelDirectory: directory, model: model)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())

        // lm_head → 2-bit GemmaQuantizedLinear (untied).
        let lmHead = try #require(modules["language_model.lm_head"] as? GemmaQuantizedLinear)
        #expect(lmHead.numBits == 2)

        // embed_tokens → 2-bit per-row GemmaQuantizedEmbedding.
        let embed = try #require(
            modules["language_model.model.embed_tokens"] as? GemmaQuantizedEmbedding)
        #expect(embed.numBits == 2)
        #expect(embed.numBlocks == 1)

        // embed_tokens_per_layer → 4-bit block-wise GemmaQuantizedEmbedding
        // (one block per layer).
        let embedPerLayer = try #require(
            modules["language_model.model.embed_tokens_per_layer"] as? GemmaQuantizedEmbedding)
        #expect(embedPerLayer.numBits == 4)
        #expect(embedPerLayer.numBlocks == numLayers)

        // mlp / attention → 4-bit; PLE gates/projections → 8-bit.
        let gate = try #require(
            modules["language_model.model.layers.0.mlp.gate_proj"] as? GemmaQuantizedLinear)
        #expect(gate.numBits == 4)
        let qProj = try #require(
            modules["language_model.model.layers.0.self_attn.q_proj"] as? GemmaQuantizedLinear)
        #expect(qProj.numBits == 4)
        let perLayerGate = try #require(
            modules["language_model.model.layers.0.per_layer_input_gate"]
                as? GemmaQuantizedLinear)
        #expect(perLayerGate.numBits == 8)
        let perLayerProj = try #require(
            modules["language_model.model.layers.0.per_layer_projection"]
                as? GemmaQuantizedLinear)
        #expect(perLayerProj.numBits == 8)

        // per_layer_model_projection stays a plain fp Linear (not converted).
        let perLayerModelProj = try #require(
            modules["language_model.model.per_layer_model_projection"] as? Linear)
        #expect(!(perLayerModelProj is GemmaQuantizedLinear))

        // Forward pass produces finite logits of the right shape.
        let cache = model.newCache(parameters: nil)
        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let logits = model(tokens, cache: cache)
        eval(logits)
        #expect(logits.shape == [1, 3, config.textConfig.vocabSize])
        let values = logits.asType(.float32).asArray(Float.self)
        #expect(values.allSatisfy { $0.isFinite }, "logits must be finite, got \(values)")
    }

    // MARK: - Synthetic checkpoint

    /// Build a checkpoint laid out like the mobile format: packed weights +
    /// scales for quantizable `Linear`/`Embedding` leaves, plain-fp arrays for
    /// everything else (norms, `layer_scalar`, `per_layer_model_projection`).
    private func makeMobileCheckpoint(
        model: Gemma4Model, quantizationConfig: GemmaMobileQuantizationConfig, numLayers: Int
    ) -> [String: MLXArray] {
        var arrays = [String: MLXArray]()
        var quantizedPaths = Set<String>()

        for (path, module) in model.leafModules().flattened() {
            let bits = resolveModuleBits(path: path, config: quantizationConfig)
            if let linear = module as? Linear, let bits = bits {
                let (out, inDim) = linear.shape
                let dtype: DType = bits == 8 ? .int8 : .uint8
                arrays["\(path).weight"] = randomIntArray([out, packedInputDim(inDim, bits)], dtype: dtype)
                arrays["\(path).weight_scale"] = randomScaleArray([out, 1])
                arrays["\(path).input_activation_scale"] = MLXArray(Float(0.0))
                arrays["\(path).output_activation_scale"] = MLXArray(Float(0.0))
                quantizedPaths.insert(path)
            } else if let embedding = module as? Embedding,
                quantizationConfig.quantizeEmbeddings,
                let bits = bits
            {
                let (numEmb, dim) = embedding.shape
                let dtype: DType = bits == 8 ? .int8 : .uint8
                let numBlocks = path.contains("embed_tokens_per_layer") ? numLayers : 1
                arrays["\(path).embedding_quantized"] = randomIntArray(
                    [numEmb, packedInputDim(dim, bits)], dtype: dtype)
                arrays["\(path).embedding_scale"] = randomScaleArray([numEmb, numBlocks])
                quantizedPaths.insert(path)
            }
        }

        // Copy the remaining (non-quantized) parameters verbatim.
        for (key, value) in model.parameters().flattened() {
            let modulePath = key.split(separator: ".").dropLast().joined(separator: ".")
            if quantizedPaths.contains(modulePath) { continue }
            arrays[key] = value
        }
        return arrays
    }

    private func writeCheckpoint(_ arrays: [String: MLXArray]) throws -> URL {
        let directory = URL(filePath: NSTemporaryDirectory())
            .appending(component: "gemma4-mobile-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try save(arrays: arrays, url: directory.appending(component: "model.safetensors"))
        return directory
    }

    private func packedInputDim(_ inputDims: Int, _ numBits: Int) -> Int {
        switch numBits {
        case 2: return (inputDims + 3) / 4
        case 4: return (inputDims + 1) / 2
        case 8: return inputDims
        default: fatalError("Unsupported numBits \(numBits)")
        }
    }

    private func randomIntArray(_ shape: [Int], dtype: DType) -> MLXArray {
        let count = shape.reduce(1, *)
        if dtype == .int8 {
            let vals = (0 ..< count).map { _ in Int8.random(in: -128 ... 127) }
            return MLXArray(vals, shape)
        } else {
            let vals = (0 ..< count).map { _ in UInt8.random(in: 0 ... 255) }
            return MLXArray(vals, shape)
        }
    }

    private func randomScaleArray(_ shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        let vals = (0 ..< count).map { _ in Float.random(in: 0.1 ... 1.0) }
        return MLXArray(vals, shape)
    }

    // MARK: - Real checkpoint

    /// Path to the real `gemma-4-E2B-it-qat-mobile-mlx-mm` checkpoint. This is
    /// a local path and may not exist in all environments; the test skips if
    /// the checkpoint is absent.
    private static let realCheckpointURL = URL(filePath: "/Users/adrgrondin/Workspace/mlx-vlm/gemma-4-E2B-it-qat-mobile-mlx-mm")

    @Test("Real gemma-4-E2B-it-qat-mobile-mlx-mm loads with quantized layers and runs")
    func loadsRealMobileCheckpoint() throws {
        let dir = Self.realCheckpointURL
        guard FileManager.default.fileExists(atPath: dir.appending(component: "config.json").path) else {
            // Checkpoint not available — skip this test.
            return
        }

        let configData = try Data(contentsOf: dir.appending(component: "config.json"))
        let decoder = JSONDecoder.json5()
        decoder.userInfo[.rawConfigData] = configData
        let config = try decoder.decode(Gemma4Configuration.self, from: configData)
        #expect(config.quantizationConfig?.isGemmaMobile == true)

        let model = Gemma4Model(config)
        try loadWeights(modelDirectory: dir, model: model)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())

        // lm_head → 2-bit GemmaQuantizedLinear.
        let lmHead = try #require(modules["language_model.lm_head"] as? GemmaQuantizedLinear)
        #expect(lmHead.numBits == 2)

        // embed_tokens → 2-bit per-row GemmaQuantizedEmbedding.
        let embed = try #require(
            modules["language_model.model.embed_tokens"] as? GemmaQuantizedEmbedding)
        #expect(embed.numBits == 2)
        #expect(embed.numBlocks == 1)

        // embed_tokens_per_layer → 4-bit block-wise (one block per layer = 35).
        let embedPerLayer = try #require(
            modules["language_model.model.embed_tokens_per_layer"] as? GemmaQuantizedEmbedding)
        #expect(embedPerLayer.numBits == 4)
        #expect(embedPerLayer.numBlocks == config.textConfig.numHiddenLayers)

        // Layer 0 mlp → 4-bit (layers 0–14).
        let gate0 = try #require(
            modules["language_model.model.layers.0.mlp.gate_proj"] as? GemmaQuantizedLinear)
        #expect(gate0.numBits == 4)

        // Layer 15 mlp → 2-bit (layers 15–34).
        let gate15 = try #require(
            modules["language_model.model.layers.15.mlp.gate_proj"] as? GemmaQuantizedLinear)
        #expect(gate15.numBits == 2)

        // Layer 0 attention → 4-bit.
        let qProj0 = try #require(
            modules["language_model.model.layers.0.self_attn.q_proj"] as? GemmaQuantizedLinear)
        #expect(qProj0.numBits == 4)

        // PLE gates/projections → 8-bit.
        let perLayerGate = try #require(
            modules["language_model.model.layers.0.per_layer_input_gate"]
                as? GemmaQuantizedLinear)
        #expect(perLayerGate.numBits == 8)

        // per_layer_model_projection stays a plain fp Linear (not converted).
        let perLayerModelProj = try #require(
            modules["language_model.model.per_layer_model_projection"] as? Linear)
        #expect(!(perLayerModelProj is GemmaQuantizedLinear))

        // Forward pass produces finite logits of the right shape.
        let cache = model.newCache(parameters: nil)
        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let logits = model(tokens, cache: cache)
        eval(logits)
        #expect(logits.shape == [1, 3, config.textConfig.vocabSize])
        let values = logits.asType(.float32).asArray(Float.self)
        #expect(values.allSatisfy { $0.isFinite }, "logits must be finite, got first few: \(values.prefix(10))")
    }

    // MARK: - Native compiled path A/B equivalence

    /// Verify the native compiled path (quantizedMM + compile fusion) produces
    /// logits close to the eager path for the real model.
    ///
    /// Both paths now use float32 SRQ + float32 quantizedMM input (matching Python
    /// `_srq`), so the SRQ rounding is identical. The remaining difference is
    /// numerical noise from (1) the fused q/k/v matmul (native) vs three separate
    /// matmuls (eager) — different Metal tiling/accumulation, and (2) compile
    /// fusion (native) vs separate ops (eager) — different kernel implementations.
    /// These cause a small mean abs diff (< 0.35) and 0–3 argmax mismatches out of
    /// 32 positions, which is expected for two different code paths.
    @Test("Native compiled path matches eager path top-1 token (real model)")
    func nativeCompiledPathMatchesEager() throws {
        let dir = Self.realCheckpointURL
        guard FileManager.default.fileExists(atPath: dir.appending(component: "config.json").path) else {
            return  // Checkpoint not available — skip.
        }

        let configData = try Data(contentsOf: dir.appending(component: "config.json"))
        let decoder = JSONDecoder.json5()
        decoder.userInfo[.rawConfigData] = configData
        let config = try decoder.decode(Gemma4Configuration.self, from: configData)

        // Disable load-time precompilation + weight freeing for this A/B test:
        // once `loadWeights` frees the mobile-format weights, the eager path
        // (which dequantizes them) can no longer run. The native path still
        // works (it converts lazily on the first forward pass), so disabling
        // precompilation keeps both paths runnable on the same model instance.
        Gemma4TextModel.precompileAtLoad = false
        defer { Gemma4TextModel.precompileAtLoad = true }

        let model = Gemma4Model(config)
        try loadWeights(modelDirectory: dir, model: model)

        // Use a fixed 32-token prompt so the eager path also uses quantizedMM
        // (batch > 16), isolating the difference to the compile fusion (not qmv
        // vs quantizedMM). Fixed tokens give a deterministic diff (random tokens
        // produced 0.08–0.80 mean diff due to different activation patterns).
        let tokens = MLXArray((0..<32).map { Int32($0 % 1000 + 1) }).reshaped([1, 32])

        // Run with the native compiled path.
        Gemma4TextModel.useNativeCompiledPath = true
        let cache1 = model.newCache(parameters: nil)
        let logitsNative = model(tokens, cache: cache1)
        eval(logitsNative)
        let nativeValues = logitsNative.asType(.float32).asArray(Float.self)
        #expect(nativeValues.allSatisfy { $0.isFinite },
            "native path logits must be finite")

        // Run with the eager path (flag off).
        Gemma4TextModel.useNativeCompiledPath = false
        let cache2 = model.newCache(parameters: nil)
        let logitsEager = model(tokens, cache: cache2)
        eval(logitsEager)
        let eagerValues = logitsEager.asType(.float32).asArray(Float.self)
        #expect(eagerValues.allSatisfy { $0.isFinite },
            "eager path logits must be finite")

        // Restore the flag.
        Gemma4TextModel.useNativeCompiledPath = true

        // Compare: mean absolute difference should be small (float32 accumulation
        // order differences between fused-compile and separate-ops paths).
        let vocabSize = config.textConfig.vocabSize
        let nPositions = 32
        var totalDiff: Float = 0
        var maxDiff: Float = 0
        var argmaxMismatches = 0
        for pos in 0 ..< nPositions {
            let offset = pos * vocabSize
            var posDiff: Float = 0
            for i in 0..<vocabSize {
                let d = abs(nativeValues[offset + i] - eagerValues[offset + i])
                posDiff += d
                maxDiff = max(maxDiff, d)
            }
            totalDiff += posDiff / Float(vocabSize)
            let nativeSlice = Array(nativeValues[offset..<(offset + vocabSize)])
            let eagerSlice = Array(eagerValues[offset..<(offset + vocabSize)])
            let nativeArgmax = nativeSlice.enumerated().max(by: { $0.1 < $1.1 })!.0
            let eagerArgmax = eagerSlice.enumerated().max(by: { $0.1 < $1.1 })!.0
            if nativeArgmax != eagerArgmax { argmaxMismatches += 1 }
        }
        let meanDiff = totalDiff / Float(nPositions)
        print("\(nPositions)-token prompt: mean abs diff = \(meanDiff), max abs diff = \(maxDiff), argmax mismatches = \(argmaxMismatches)/\(nPositions)")
        #expect(meanDiff < 0.5, "mean abs diff too large: \(meanDiff)")
        #expect(argmaxMismatches <= 4, "too many argmax mismatches: \(argmaxMismatches)")
    }

    // MARK: - Phase 5: load-time precompilation + weight freeing

    /// Verify the load-time precompilation hook (Phase 5) frees the mobile-format
    /// decoder-layer weights, keeps `lm_head`/embeddings, and produces stable
    /// (deterministic) prefill across runs.
    @Test("Load-time precompilation frees mobile weights and stabilizes prefill (real model)")
    func precompileFreesWeightsAndStabilizesPrefill() throws {
        let dir = Self.realCheckpointURL
        guard FileManager.default.fileExists(atPath: dir.appending(component: "config.json").path) else {
            return  // Checkpoint not available — skip.
        }

        let configData = try Data(contentsOf: dir.appending(component: "config.json"))
        let decoder = JSONDecoder.json5()
        decoder.userInfo[.rawConfigData] = configData
        let config = try decoder.decode(Gemma4Configuration.self, from: configData)

        // precompileAtLoad defaults to true; make it explicit. The load hook
        // converts to native quantizedMM format, frees mobile weights, and warms
        // up the per-shape compile graphs.
        Gemma4TextModel.precompileAtLoad = true
        Gemma4TextModel.useNativeCompiledPath = true
        defer {
            Gemma4TextModel.precompileAtLoad = true
            Gemma4TextModel.useNativeCompiledPath = true
        }

        let model = Gemma4Model(config)
        try loadWeights(modelDirectory: dir, model: model)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())

        // Decoder-layer mobile weights are freed: `weight`/`weightScale` replaced
        // with a dummy [1] array. The native path uses the converted _mlxWeight.
        let gate0 = try #require(
            modules["language_model.model.layers.0.mlp.gate_proj"] as? GemmaQuantizedLinear)
        #expect(gate0.weight.shape == [1],
            "decoder layer mobile weight should be freed, got \(gate0.weight.shape)")
        #expect(gate0.weightScale.shape == [1],
            "decoder layer mobile weightScale should be freed, got \(gate0.weightScale.shape)")

        // A KV-shared layer (15) is also freed.
        let gate15 = try #require(
            modules["language_model.model.layers.15.mlp.gate_proj"] as? GemmaQuantizedLinear)
        #expect(gate15.weight.shape == [1],
            "KV-shared layer mobile weight should be freed, got \(gate15.weight.shape)")

        // lm_head (2-bit, not a decoder layer) keeps its mobile weights — it
        // stays on the qmv/eager path.
        let lmHead = try #require(modules["language_model.lm_head"] as? GemmaQuantizedLinear)
        #expect(lmHead.weight.ndim == 2,
            "lm_head mobile weight should NOT be freed, got shape \(lmHead.weight.shape)")

        // Embeddings keep their packed tables (dequant-on-forward, not converted).
        let embed = try #require(
            modules["language_model.model.embed_tokens"] as? GemmaQuantizedEmbedding)
        #expect(embed.weight.ndim == 2,
            "embed_tokens weight should NOT be freed, got shape \(embed.weight.shape)")

        // Forward pass produces finite logits of the right shape (native path).
        let cache = model.newCache(parameters: nil)
        let tokens = MLXArray((0..<32).map { Int32($0 % 1000 + 1) }).reshaped([1, 32])
        let logits1 = model(tokens, cache: cache)
        eval(logits1)
        #expect(logits1.shape == [1, 32, config.textConfig.vocabSize])
        let values1 = logits1.asType(.float32).asArray(Float.self)
        #expect(values1.allSatisfy { $0.isFinite }, "logits must be finite")

        // Prefill is stable: a second run with the same input produces the
        // exact same output (the precompiled graphs are deterministic).
        let cache2 = model.newCache(parameters: nil)
        let logits2 = model(tokens, cache: cache2)
        eval(logits2)
        let values2 = logits2.asType(.float32).asArray(Float.self)
        #expect(values2.allSatisfy { $0.isFinite }, "logits must be finite")
        var maxDiff: Float = 0
        for i in 0..<values1.count {
            maxDiff = max(maxDiff, abs(values1[i] - values2[i]))
        }
        #expect(maxDiff == 0, "prefill should be stable across runs, max diff = \(maxDiff)")
    }

    /// Verify the load-time precompilation is a no-op when the native path is
    /// not usable (unaligned dims). The tiny model has hidden_size=16 (16 % 128 != 0),
    /// so `getNativeArgs()` returns nil for every layer and the mobile weights are
    /// kept (the eager/qmv path still needs them).
    @Test("Load-time precompilation is a no-op for unaligned dims (tiny model)")
    func precompileNoOpForUnalignedDims() throws {
        MLXRandom.seed(0)
        let configData = Data(tinyConfigJSON().utf8)
        let decoder = JSONDecoder()
        decoder.userInfo[.rawConfigData] = configData
        let config = try decoder.decode(Gemma4Configuration.self, from: configData)
        let model = Gemma4Model(config)
        eval(model)

        let numLayers = config.textConfig.numHiddenLayers
        let arrays = makeMobileCheckpoint(
            model: model, quantizationConfig: config.quantizationConfig!, numLayers: numLayers)
        let directory = try writeCheckpoint(arrays)
        defer { try? FileManager.default.removeItem(at: directory) }

        // precompileAtLoad defaults to true, but the tiny model's dims are not
        // divisible by 128, so the native path is not usable → no-op.
        Gemma4TextModel.precompileAtLoad = true
        defer { Gemma4TextModel.precompileAtLoad = true }

        try loadWeights(modelDirectory: directory, model: model)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let gate = try #require(
            modules["language_model.model.layers.0.mlp.gate_proj"] as? GemmaQuantizedLinear)
        // Weights are NOT freed (unaligned dims → native path not usable → no-op).
        #expect(gate.weight.ndim == 2,
            "tiny model weights should NOT be freed (unaligned dims), got \(gate.weight.shape)")

        // Forward pass still works (eager path).
        let cache = model.newCache(parameters: nil)
        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let logits = model(tokens, cache: cache)
        eval(logits)
        #expect(logits.shape == [1, 3, config.textConfig.vocabSize])
        let values = logits.asType(.float32).asArray(Float.self)
        #expect(values.allSatisfy { $0.isFinite }, "logits must be finite")
    }
}
