import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Parity tests for the single-token decode Metal kernels in
/// `MapleKernels.swift`. Each kernel must match the portable implementation it
/// replaces; the model-level test also verifies the probes latch on and that
/// forcing every fast path off reproduces the same decode logits.
final class MapleKernelTests: XCTestCase {
    /// Kernel-eligible configuration: hidden 256 (2 heads x 128, rotary 64),
    /// 32 experts with top-8 routing, one sliding and one full layer.
    private func kernelConfiguration(
        layerTypes: [String] = ["sliding_attention", "full_attention"]
    ) throws -> MapleConfiguration {
        let types = layerTypes.map { "\"\($0)\"" }.joined(separator: ",")
        let json = """
            {
              "model_type": "maple",
              "hidden_size": 256,
              "intermediate_size": 512,
              "moe_intermediate_size": 128,
              "num_hidden_layers": \(layerTypes.count),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 128,
              "num_experts": 32,
              "num_experts_per_tok": 8,
              "first_k_dense_replace": 0,
              "partial_rotary_factor": 0.5,
              "vocab_size": 64,
              "sliding_window": 8,
              "layer_types": [\(types)]
            }
            """
        return try JSONDecoder().decode(MapleConfiguration.self, from: Data(json.utf8))
    }

    func testAddRMSNormKernelMatchesPortable() throws {
        let weight = (MLXRandom.normal([2048]) + 1).asType(.bfloat16)
        XCTAssertTrue(
            MapleAddRMSNormKernel.probe(
                dimensions: 2048, dtype: .bfloat16, weight: weight, eps: 1e-6))
        XCTAssertFalse(MapleAddRMSNormKernel.supports(dimensions: 2000))

        let x = MLXRandom.normal([1, 1, 2048]).asType(.bfloat16)
        let r = MLXRandom.normal([1, 1, 2048]).asType(.bfloat16)
        let (h, hn) = MapleAddRMSNormKernel.callAsFunction(x, r, weight, eps: 1e-6)
        let referenceH = x + r
        let referenceHN = MLXFast.rmsNorm(
            referenceH.asType(.float32), weight: weight.asType(.float32), eps: 1e-6
        ).asType(.bfloat16)
        eval(h, hn, referenceH, referenceHN)
        XCTAssertEqual(h.dtype, .bfloat16)
        XCTAssertTrue(h.allClose(referenceH, rtol: 2e-2, atol: 2e-2).item(Bool.self))
        XCTAssertTrue(hn.allClose(referenceHN, rtol: 2e-2, atol: 2e-2).item(Bool.self))
    }

    func testFusedQKNormRoPEMatchesPortableAtNonzeroOffset() throws {
        let config = try kernelConfiguration()
        for layerType in ["sliding_attention", "full_attention"] {
            let attention = MapleAttention(config, layerType: layerType)
            XCTAssertTrue(attention.fusedQKEligible, layerType)
            let qk = MLXRandom.normal([3, 128]).asType(.bfloat16)
            for offset in [0, 7, 129] {
                let fused = attention.fusedQKDecode(qk, offset: offset)
                let reference = attention.referenceQKDecode(qk, offset: offset)
                eval(fused, reference)
                XCTAssertTrue(
                    fused.allClose(reference, rtol: 2e-2, atol: 2e-2).item(Bool.self),
                    "\(layerType) at offset \(offset)")
            }
        }
    }

    func testFusedRouterMatchesPortable() throws {
        let gate = MapleGate(try kernelConfiguration())
        try gate.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXRandom.normal([32, 256])
            ]), verify: [])
        let x = MLXRandom.normal([1, 1, 256]).asType(.bfloat16)

        let (indices, scores) = gate.fusedCall(x)
        let (referenceIndices, referenceScores) = gate.referenceCall(x)
        eval(indices, scores, referenceIndices, referenceScores)

        XCTAssertEqual(indices.shape, referenceIndices.shape)
        XCTAssertEqual(indices.dtype, .int32)
        XCTAssertEqual(scores.dtype, .float32)
        XCTAssertEqual(Set(indices.asArray(Int32.self)), Set(referenceIndices.asArray(Int32.self)))
        XCTAssertTrue(
            sorted(scores, axis: -1)
                .allClose(sorted(referenceScores, axis: -1), rtol: 1e-5, atol: 1e-5)
                .item(Bool.self))
        XCTAssertEqual(scores.sum().item(Float.self), 1, accuracy: 1e-5)
    }

    func testDecodeFastPathsMatchPortableLogits() throws {
        let model = MapleModel(try kernelConfiguration())
        // Random router weights so the top-8 boundary has no ties between the
        // fused and portable selection.
        try model.update(
            parameters: ModuleParameters.unflattened([
                "model.layers.0.mlp.gate.weight": MLXRandom.normal([32, 256]),
                "model.layers.1.mlp.gate.weight": MLXRandom.normal([32, 256]),
            ]), verify: [])
        let prompt = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)
        let token = MLXArray([6] as [Int32]).reshaped(1, 1)

        // Fast-path run: the first decode probes and latches every kernel on.
        let fusedCache = model.newCache(parameters: nil)
        _ = model(prompt, cache: fusedCache)
        let fusedLogits = model(token, cache: fusedCache)
        eval(fusedLogits)

        XCTAssertEqual(model.model.fusedAddNorm, true)
        for layer in model.model.layers {
            XCTAssertEqual(layer.selfAttention.fusedQK, true)
            XCTAssertEqual(
                (layer.mlp as! MapleSparseMoeBlock).gate.fusedRouter, true)
        }

        // Portable run with identical cache state and every fast path off.
        let portableCache = model.newCache(parameters: nil)
        _ = model(prompt, cache: portableCache)
        model.disableFusedDecodeKernels()
        let portableLogits = model(token, cache: portableCache)
        eval(portableLogits)

        XCTAssertTrue(
            fusedLogits.asType(.float32)
                .allClose(portableLogits.asType(.float32), rtol: 1e-2, atol: 1e-2)
                .item(Bool.self))
    }
}
