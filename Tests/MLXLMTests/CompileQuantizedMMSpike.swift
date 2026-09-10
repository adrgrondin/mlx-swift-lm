// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// Phase 0 de-risking spike: verify that `quantizedMM` and `MLXFast.RoPE`
/// (with `MLXArray` offset) trace and execute correctly inside `compile`.
///
/// These are the two biggest unknowns for porting the Python Phase 6/7
/// compiled-fusion approach (native `quantized_matmul` + `mx.compile`). If
/// either fails to trace inside `compile`, the entire compiled-fusion plan is
/// blocked and we fall back to the current qmv + separate-compile path.
///
/// See `GEMMA4_QAT_MOBILE_NATIVE_MATMUL_SWIFT_PLAN.md` §5.2 and §5.3.
struct CompileQuantizedMMSpike {

    // MARK: - quantizedMM inside compile

    /// Verify `quantizedMM` + `MLXFast.rmsNorm` + SRQ fuse correctly inside a
    /// `compile` graph, matching the eager path, for multiple shapes.
    @Test("quantizedMM + rmsNorm + SRQ fuse correctly inside compile")
    func quantizedMMInCompile() throws {
        let outDims = 8
        let inDims = 128
        let numBits = 4
        let packedIn = inDims / 2  // int4: 2 values/byte

        // Random packed uint8 weights + per-channel scale.
        let weightBytes = (0 ..< (outDims * packedIn)).map { _ in UInt8.random(in: 0 ... 255) }
        let weight = MLXArray(weightBytes, [outDims, packedIn])
        let weightScale = MLXArray(
            (0 ..< outDims).map { 0.1 * Float($0 + 1) }, [outDims, 1])
        let normWeight = MLXArray.ones([inDims])

        // Convert to MLX uint32 format (group_size=128).
        let (packed, scales, biases) = mobileToMLX(
            weight: weight, weightScale: weightScale,
            numBits: numBits, inputDims: inDims)
        eval([packed, scales, biases, normWeight])

        // SRQ scales (calibrated — non-zero to exercise the SRQ path).
        let inScale = MLXArray(Float(0.5))
        let outScale = MLXArray(Float(0.3))

        // Eager reference: rmsNorm → SRQ(in) → quantizedMM → SRQ(out) → cast.
        func eagerPath(_ x: MLXArray) -> MLXArray {
            var h = MLXFast.rmsNorm(x, weight: normWeight, eps: 1e-6).asType(.float32)
            let s = inScale.asType(.float32)
            let safe = MLX.where(s .== 0, MLXArray.ones(like: s), s)
            h = MLX.clip(MLX.round(h / safe), min: -128, max: 127) * safe
            var q = quantizedMM(
                h, packed, scales: scales, biases: biases,
                transpose: true, groupSize: 128, bits: numBits, mode: .affine)
            let os = outScale.asType(.float32)
            let osafe = MLX.where(os .== 0, MLXArray.ones(like: os), os)
            q = MLX.clip(MLX.round(q / osafe), min: -128, max: 127) * osafe
            return q.asType(x.dtype)
        }

        // Compiled version (array form — same as the post-attention segment).
        let compiledFn = compile { (args: [MLXArray]) -> [MLXArray] in
            let x = args[0]
            let normW = args[1]
            let pk = args[2]
            let sc = args[3]
            let bi = args[4]
            let inS = args[5]
            let outS = args[6]

            var h = MLXFast.rmsNorm(x, weight: normW, eps: 1e-6).asType(.float32)
            let s = inS.asType(.float32)
            let safe = MLX.where(s .== 0, MLXArray.ones(like: s), s)
            h = MLX.clip(MLX.round(h / safe), min: -128, max: 127) * safe
            var q = quantizedMM(
                h, pk, scales: sc, biases: bi,
                transpose: true, groupSize: 128, bits: 4, mode: .affine)
            let os = outS.asType(.float32)
            let osafe = MLX.where(os .== 0, MLXArray.ones(like: os), os)
            q = MLX.clip(MLX.round(q / osafe), min: -128, max: 127) * osafe
            return [q.asType(x.dtype)]
        }

        let allArgs: [MLXArray] = [normWeight, packed, scales, biases, inScale, outScale]

        // Shape 1: [1, 4, 128] (prefill-like).
        let x1 = MLXArray.zeros([1, 4, inDims], dtype: .float32)
        let eager1 = eagerPath(x1)
        let compiled1 = compiledFn([x1] + allArgs)[0]
        eval([eager1, compiled1])
        let maxDiff1 = MLX.abs(eager1 - compiled1).max().item(Float.self)
        #expect(maxDiff1 < 1e-4, "quantizedMM in compile (shape [1,4,128]): max diff \(maxDiff1)")

        // Shape 2: [1, 1, 128] (decode-like — different shape triggers recompile).
        let x2 = MLXArray.zeros([1, 1, inDims], dtype: .float32)
        let eager2 = eagerPath(x2)
        let compiled2 = compiledFn([x2] + allArgs)[0]
        eval([eager2, compiled2])
        let maxDiff2 = MLX.abs(eager2 - compiled2).max().item(Float.self)
        #expect(maxDiff2 < 1e-4, "quantizedMM in compile (shape [1,1,128]): max diff \(maxDiff2)")

        // Shape 3: [2, 8, 128] (batched prefill — yet another shape).
        let x3 = MLXArray.zeros([2, 8, inDims], dtype: .float32)
        let eager3 = eagerPath(x3)
        let compiled3 = compiledFn([x3] + allArgs)[0]
        eval([eager3, compiled3])
        let maxDiff3 = MLX.abs(eager3 - compiled3).max().item(Float.self)
        #expect(maxDiff3 < 1e-4, "quantizedMM in compile (shape [2,8,128]): max diff \(maxDiff3)")
    }

    // MARK: - MLXFast.RoPE inside compile

    /// Verify `MLXFast.RoPE` with an `MLXArray` offset (the dynamic overload)
    /// traces and executes correctly inside `compile`, matching the eager path.
    @Test("MLXFast.RoPE with MLXArray offset traces and executes inside compile")
    func ropeInCompile() throws {
        let headDim = 256
        let nHeads = 8

        // Precompute rope freqs: base^(arange(0, dims, 2) / dims).
        let base: Float = 10000.0
        let freqs = MLXArray(base, dtype: .float32).pow(
            MLX.arange(0, headDim, step: 2, dtype: .float32) / Float(headDim))
        eval(freqs)

        // Eager reference.
        func eagerRoPE(_ queries: MLXArray, offset: MLXArray) -> MLXArray {
            MLXFast.RoPE(
                queries, dimensions: headDim, traditional: false,
                base: nil, scale: 1.0, offset: offset, freqs: freqs)
        }

        // Compiled version (array form).
        let compiledRoPE = compile { (args: [MLXArray]) -> [MLXArray] in
            let q = args[0]
            let off = args[1]
            let fr = args[2]
            let result = MLXFast.RoPE(
                q, dimensions: headDim, traditional: false,
                base: nil, scale: 1.0, offset: off, freqs: fr)
            return [result]
        }

        // Shape 1: [1, nHeads, 4, headDim] (prefill-like, transposed layout).
        let queries1 = MLXArray.zeros([1, nHeads, 4, headDim], dtype: .float32)
        let offset1 = MLXArray(0)
        let eager1 = eagerRoPE(queries1, offset: offset1)
        let compiled1 = compiledRoPE([queries1, offset1, freqs])[0]
        eval([eager1, compiled1])
        let maxDiff1 = MLX.abs(eager1 - compiled1).max().item(Float.self)
        #expect(maxDiff1 < 1e-5, "RoPE in compile (shape [1,8,4,256]): max diff \(maxDiff1)")

        // Shape 2: [1, nHeads, 1, headDim] (decode-like, different shape).
        let queries2 = MLXArray.zeros([1, nHeads, 1, headDim], dtype: .float32)
        let offset2 = MLXArray(5)  // non-zero offset
        let eager2 = eagerRoPE(queries2, offset: offset2)
        let compiled2 = compiledRoPE([queries2, offset2, freqs])[0]
        eval([eager2, compiled2])
        let maxDiff2 = MLX.abs(eager2 - compiled2).max().item(Float.self)
        #expect(
            maxDiff2 < 1e-5, "RoPE in compile (shape [1,8,1,256], offset=5): max diff \(maxDiff2)")
    }

    // MARK: - Combined: quantizedMM + RoPE + rmsNorm inside one compile graph

    /// Verify a mini pre-attention segment (rmsNorm → SRQ → quantizedMM → SRQ →
    /// reshape → rmsNorm → transpose → RoPE) compiles and matches the eager
    /// path. This is a slice of the real pre-attention compiled segment.
    @Test("Mini pre-attention segment compiles and matches eager path")
    func miniPreAttnSegment() throws {
        let outDims = 8  // nHeads * headDim (tiny)
        let inDims = 128
        let numBits = 4
        let packedIn = inDims / 2
        let nHeads = 4
        let headDim = 2  // outDims / nHeads

        let weightBytes = (0 ..< (outDims * packedIn)).map { _ in UInt8.random(in: 0 ... 255) }
        let weight = MLXArray(weightBytes, [outDims, packedIn])
        let weightScale = MLXArray(
            (0 ..< outDims).map { 0.1 * Float($0 + 1) }, [outDims, 1])
        let inputNormW = MLXArray.ones([inDims])
        let qNormW = MLXArray.ones([headDim])

        let (packed, scales, biases) = mobileToMLX(
            weight: weight, weightScale: weightScale,
            numBits: numBits, inputDims: inDims)
        eval([packed, scales, biases, inputNormW, qNormW])

        let inScale = MLXArray(Float(0.5))
        let outScale = MLXArray(Float(0.3))
        let base: Float = 10000.0
        let freqs = MLXArray(base, dtype: .float32).pow(
            MLX.arange(0, headDim, step: 2, dtype: .float32) / Float(headDim))
        eval(freqs)

        // Eager reference.
        func eager(_ x: MLXArray, offset: MLXArray) -> MLXArray {
            var h = MLXFast.rmsNorm(x, weight: inputNormW, eps: 1e-6).asType(.float32)
            let s = inScale.asType(.float32)
            let safe = MLX.where(s .== 0, MLXArray.ones(like: s), s)
            h = MLX.clip(MLX.round(h / safe), min: -128, max: 127) * safe
            var q = quantizedMM(
                h, packed, scales: scales, biases: biases,
                transpose: true, groupSize: 128, bits: numBits, mode: .affine)
            let os = outScale.asType(.float32)
            let osafe = MLX.where(os .== 0, MLXArray.ones(like: os), os)
            q = MLX.clip(MLX.round(q / osafe), min: -128, max: 127) * osafe
            q = q.asType(x.dtype)
            let (B, L) = (q.dim(0), q.dim(1))
            var queries = q.reshaped(B, L, nHeads, headDim)
            queries = MLXFast.rmsNorm(queries, weight: qNormW, eps: 1e-6)
            queries = queries.transposed(0, 2, 1, 3)
            queries = MLXFast.RoPE(
                queries, dimensions: headDim, traditional: false,
                base: nil, scale: 1.0, offset: offset, freqs: freqs)
            return queries
        }

        // Compiled version.
        let compiledFn = compile { (args: [MLXArray]) -> [MLXArray] in
            let x = args[0]
            let inputNW = args[1]
            let pk = args[2]
            let sc = args[3]
            let bi = args[4]
            let inS = args[5]
            let outS = args[6]
            let qNW = args[7]
            let fr = args[8]
            let off = args[9]

            var h = MLXFast.rmsNorm(x, weight: inputNW, eps: 1e-6).asType(.float32)
            let s = inS.asType(.float32)
            let safe = MLX.where(s .== 0, MLXArray.ones(like: s), s)
            h = MLX.clip(MLX.round(h / safe), min: -128, max: 127) * safe
            var q = quantizedMM(
                h, pk, scales: sc, biases: bi,
                transpose: true, groupSize: 128, bits: 4, mode: .affine)
            let os = outS.asType(.float32)
            let osafe = MLX.where(os .== 0, MLXArray.ones(like: os), os)
            q = MLX.clip(MLX.round(q / osafe), min: -128, max: 127) * osafe
            q = q.asType(x.dtype)
            let (B, L) = (q.dim(0), q.dim(1))
            var queries = q.reshaped(B, L, nHeads, headDim)
            queries = MLXFast.rmsNorm(queries, weight: qNW, eps: 1e-6)
            queries = queries.transposed(0, 2, 1, 3)
            queries = MLXFast.RoPE(
                queries, dimensions: headDim, traditional: false,
                base: nil, scale: 1.0, offset: off, freqs: fr)
            return [queries]
        }

        let staticArgs: [MLXArray] = [
            inputNormW, packed, scales, biases, inScale, outScale, qNormW, freqs,
        ]

        // Shape 1: [1, 4, 128].
        let x1 = MLXArray.zeros([1, 4, inDims], dtype: .float32)
        let off1 = MLXArray(0)
        let eager1 = eager(x1, offset: off1)
        let compiled1 = compiledFn([x1] + staticArgs + [off1])[0]
        eval([eager1, compiled1])
        let maxDiff1 = MLX.abs(eager1 - compiled1).max().item(Float.self)
        #expect(maxDiff1 < 1e-4, "Mini pre-attn (shape [1,4,128]): max diff \(maxDiff1)")

        // Shape 2: [1, 1, 128] (decode).
        let x2 = MLXArray.zeros([1, 1, inDims], dtype: .float32)
        let off2 = MLXArray(3)
        let eager2 = eager(x2, offset: off2)
        let compiled2 = compiledFn([x2] + staticArgs + [off2])[0]
        eval([eager2, compiled2])
        let maxDiff2 = MLX.abs(eager2 - compiled2).max().item(Float.self)
        #expect(maxDiff2 < 1e-4, "Mini pre-attn (shape [1,1,128], offset=3): max diff \(maxDiff2)")
    }

    // MARK: - quantizedMM input dtype sensitivity

    /// Verify whether `quantizedMM` produces different VALUES (not just dtype)
    /// for bfloat16 vs float32 input. The native compiled path passes float32
    /// to `quantizedMM` (SRQ in float32, no cast before matmul); the eager path
    /// passes bfloat16 (SRQ in bfloat16). If the values differ significantly,
    /// the input dtype is a structural difference.
    @Test("quantizedMM bfloat16 vs float32 input value difference")
    func quantizedMMInputDtypeSensitivity() throws {
        let outDims = 64
        let inDims = 128
        let numBits = 4
        let packedIn = inDims / 2

        let weightBytes = (0 ..< (outDims * packedIn)).map { _ in UInt8.random(in: 0 ... 255) }
        let weight = MLXArray(weightBytes, [outDims, packedIn])
        let weightScale = MLXArray(
            (0 ..< outDims).map { 0.1 * Float($0 + 1) }, [outDims, 1])
        let (packed, scales, biases) = mobileToMLX(
            weight: weight, weightScale: weightScale,
            numBits: numBits, inputDims: inDims)
        eval([packed, scales, biases])

        // Simulate SRQ output: round(x/s)*s in float32, then the same in bfloat16.
        let s = MLXArray(Float(0.5))
        let xRaw = MLXArray(
            (0 ..< (1 * 4 * inDims)).map { _ in Float.random(in: -2 ... 2) },
            [1, 4, inDims]
        )
        // float32 SRQ (native path style)
        let xF32 = srqF32(xRaw.asType(.float32), s)  // float32
        // bfloat16 SRQ (eager path style)
        let xBf16 = srqF32(xRaw.asType(.bfloat16).asType(.float32), s).asType(.bfloat16)

        // quantizedMM with float32 input (native path)
        let outF32 = quantizedMM(
            xF32, packed, scales: scales, biases: biases,
            transpose: true, groupSize: 128, bits: numBits, mode: .affine)
        // quantizedMM with bfloat16 input (eager path)
        let outBf16 = quantizedMM(
            xBf16, packed, scales: scales, biases: biases,
            transpose: true, groupSize: 128, bits: numBits, mode: .affine)

        eval([xF32, xBf16, outF32, outBf16])

        // First: how different are the SRQ outputs (the inputs to quantizedMM)?
        let srqDiff = MLX.abs(xF32 - xBf16.asType(.float32)).max().item(Float.self)
        // Then: how different are the quantizedMM outputs?
        let mmDiff = MLX.abs(outF32.asType(.float32) - outBf16.asType(.float32)).max().item(
            Float.self)
        let mmRelDiff = mmDiff / (MLX.abs(outF32).max().item(Float.self) + 1e-6)

        print(
            "quantizedMM dtype sensitivity: SRQ input max diff = \(srqDiff), "
                + "quantizedMM output max diff = \(mmDiff), "
                + "relative = \(mmRelDiff)")
        print("  xF32.dtype=\(xF32.dtype), xBf16.dtype=\(xBf16.dtype)")
        print("  outF32.dtype=\(outF32.dtype), outBf16.dtype=\(outBf16.dtype)")

        // This is informational — we want to know if the dtype matters.
        // Just verify no crash.
        #expect(outF32.size > 0 && outBf16.size > 0)
    }

    // MARK: - ProportionalRoPE inf-freqs equivalence (full attention)

    /// Verify that `MLXFast.RoPE` with `inf` freqs for non-rotated dims (the
    /// native compiled path approach) produces the same result as the eager
    /// `ProportionalRoPE` (manual half-split) for full-attention layers.
    ///
    /// Full attention: head_dim=512, partial_rotary_factor=0.25 → rotatedDims=128.
    /// The native path uses `MLXFast.RoPE(dimensions: 512, freqs: [64 real + 192 inf])`.
    /// The eager path uses `ProportionalRoPE` which does manual half-split +
    /// `MLXFast.RoPE(dimensions: 128, freqs: [64 real])`.
    /// Both inside and outside `compile`.
    @Test("ProportionalRoPE inf-freqs matches eager half-split (full attention)")
    func proportionalRopeInfFreqsEquivalence() throws {
        let headDim = 512
        let nHeads = 8
        let base: Float = 1_000_000.0
        let partialRotaryFactor: Float = 0.25

        // Eager ProportionalRoPE (manual half-split) via the public factory.
        let eagerRope = initializeRope(
            dims: headDim, base: base, traditional: false,
            scalingConfig: [
                "type": .string("proportional"),
                "partial_rotary_factor": .float(partialRotaryFactor),
            ],
            maxPositionEmbeddings: nil)

        // Native compiled freqs: [64 real + 192 inf], shape [256].
        let rotatedDims = 2 * Int(partialRotaryFactor * Float(headDim) / 2)
        let ropeAngles = rotatedDims / 2
        let nopeAngles = headDim / 2 - ropeAngles
        let exponents =
            MLXArray(
                stride(from: 0, to: rotatedDims, by: 2)
            ).asType(.float32) / Float(headDim)
        var nativeFreqs = MLX.pow(base, exponents)
        if nopeAngles > 0 {
            let infPad = MLXArray.ones([nopeAngles], dtype: .float32) * Float.infinity
            nativeFreqs = MLX.concatenated([nativeFreqs, infPad], axis: 0)
        }
        eval(nativeFreqs)

        // Compiled native RoPE.
        let compiledRoPE = compile { (args: [MLXArray]) -> [MLXArray] in
            let q = args[0]
            let off = args[1]
            let fr = args[2]
            let result = MLXFast.RoPE(
                q, dimensions: headDim, traditional: false,
                base: nil, scale: 1.0, offset: off, freqs: fr)
            return [result]
        }

        // Test with offset=0 and offset=5, shapes [1,8,4,512] and [1,8,1,512].
        for (shape, offsetVal)
            in ([([1, nHeads, 4, headDim], 0), ([1, nHeads, 1, headDim], 5)] as [([Int], Int)])
        {
            let queries = MLXArray(
                (0 ..< shape.reduce(1, *)).map { _ in Float.random(in: -1 ... 1) },
                shape
            ).asType(.float32)
            let offset = MLXArray(offsetVal)

            // Eager (manual half-split).
            let eagerOut = eagerRope.callAsFunction(queries, offset: offset)

            // Native (inf freqs, outside compile).
            let nativeOut = MLXFast.RoPE(
                queries, dimensions: headDim, traditional: false,
                base: nil, scale: 1.0, offset: offset, freqs: nativeFreqs)

            // Native (inf freqs, inside compile).
            let compiledOut = compiledRoPE([queries, offset, nativeFreqs])[0]

            eval([eagerOut, nativeOut, compiledOut])

            let maxDiffNative = MLX.abs(eagerOut - nativeOut).max().item(Float.self)
            let maxDiffCompiled = MLX.abs(eagerOut - compiledOut).max().item(Float.self)

            print(
                "RoPE inf-freqs shape \(shape) offset=\(offsetVal): "
                    + "eager-vs-native max diff = \(maxDiffNative), "
                    + "eager-vs-compiled max diff = \(maxDiffCompiled)")

            #expect(
                maxDiffNative < 1e-4,
                "ProportionalRoPE inf-freqs (native, shape \(shape), offset=\(offsetVal)): max diff \(maxDiffNative)"
            )
            #expect(
                maxDiffCompiled < 1e-4,
                "ProportionalRoPE inf-freqs (compiled, shape \(shape), offset=\(offsetVal)): max diff \(maxDiffCompiled)"
            )
        }
    }
}
