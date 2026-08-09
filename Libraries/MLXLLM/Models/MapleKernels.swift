// Copyright © 2026 DeepGrove AI.
//
// Single-token decode fast paths for Maple, ported from the hand-written Metal
// kernels in https://github.com/deepgrove-ai/mlx-lm (commit
// eba96c16158f032821b0bf374ea1421cfddef0a9). Decode is bounded by the serial
// dispatch chain, so each kernel folds several portable ops into one dispatch.
//
// Every fast path keeps the portable implementation as the source of truth and
// is used only after a one-time probe compares it against that implementation
// on live weights/dtypes. Any failure (compile error, shape mismatch, numerical
// drift) permanently latches the model back to the portable path.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Runtime gate for the fused decode kernels. Set `MLX_MAPLE_FUSED_KERNELS=0`
/// to force the portable decode path (useful for A/B benchmarking and for
/// diagnosing a suspected kernel issue). Any other value (or unset) leaves the
/// kernels enabled; each fast path still has to pass its one-time probe.
let mapleFusedKernelsEnabled: Bool =
    ProcessInfo.processInfo.environment["MLX_MAPLE_FUSED_KERNELS"] != "0"

/// One-time numerical probe for a hand-written kernel against its portable
/// reference. Mirrors the reference `_matches`: both callables return the same
/// number of arrays, compared in float32.
func mapleKernelMatches(
    tolerance: Double = 2e-2,
    fast: () throws -> [MLXArray],
    reference: () throws -> [MLXArray]
) -> Bool {
    do {
        return try withError {
            let got = try fast()
            let want = try reference()
            eval(got)
            eval(want)
            guard got.count == want.count else { return false }
            for (g, w) in zip(got, want) {
                guard g.shape == w.shape else { return false }
                let close = g.asType(.float32).allClose(
                    w.asType(.float32), rtol: tolerance, atol: tolerance)
                eval(close)
                guard close.item(Bool.self) else { return false }
            }
            return true
        }
    } catch {
        return false
    }
}

// MARK: - Residual add + RMSNorm

/// Residual add + RMSNorm in ONE dispatch for single-token decode.
///
/// Emits both `h = x + r` (rounded once, like the portable bf16 add) and
/// `hn = rmsnorm(h)` with the float32 weight multiply of ``MapleRMSNorm``.
enum MapleAddRMSNormKernel {
    /// eps-keyed kernel cache; one compilation per distinct RMSNorm epsilon.
    private final class KernelCache: @unchecked Sendable {
        private let lock = NSLock()
        private var kernels: [Float: MLXFast.MLXFastKernel] = [:]

        func kernel(eps: Float, make: () -> MLXFast.MLXFastKernel) -> MLXFast.MLXFastKernel {
            lock.lock()
            defer { lock.unlock() }
            if let cached = kernels[eps] { return cached }
            let created = make()
            kernels[eps] = created
            return created
        }
    }

    private static let cache = KernelCache()

    /// The kernel assigns each of the 256 threads `DIM / 256` elements.
    static func supports(dimensions dim: Int) -> Bool {
        dim > 0 && dim % 256 == 0
    }

    private static func kernel(eps: Float) -> MLXFast.MLXFastKernel {
        cache.kernel(eps: eps) {
            makeKernel(eps: eps)
        }
    }

    private static func makeKernel(eps: Float) -> MLXFast.MLXFastKernel {
        let epsLiteral = String(format: "%.10ef", Double(eps))
        let source = """
            uint tid = thread_position_in_threadgroup.x;
            constexpr uint N = DIM;
            constexpr uint PT = N / 256u;
            float hb[PT];
            float ss = 0.0f;
            for (uint i = 0; i < PT; ++i) {
                uint j = tid * PT + i;
                float v = (float)x[j] + (float)r[j];
                T_ vb = (T_)v;              // one rounding, same as a bf16 add
                h_out[j] = vb;
                hb[i] = (float)vb;          // norm sees the rounded stream
                ss += hb[i] * hb[i];
            }
            ss = simd_sum(ss);
            threadgroup float sums[8];
            uint sg = tid / 32u;
            uint lane = tid % 32u;
            if (lane == 0u) sums[sg] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tot = 0.0f;
            for (uint i = 0; i < 8u; ++i) tot += sums[i];
            float scale = metal::rsqrt(tot / (float)N + \(epsLiteral));
            for (uint i = 0; i < PT; ++i) {
                uint j = tid * PT + i;
                hn_out[j] = (T_)(hb[i] * scale * (float)w[j]);
            }
            """
        let tag =
            String(format: "%.3e", Double(eps))
            .replacingOccurrences(of: ".", with: "_")
            .replacingOccurrences(of: "-", with: "m")
            .replacingOccurrences(of: "+", with: "p")
        return MLXFast.metalKernel(
            name: "maple_add_rms_norm_\(tag)",
            inputNames: ["x", "r", "w"],
            outputNames: ["h_out", "hn_out"],
            source: source)
    }

    /// Returns `(h, hn)`: the updated residual stream and its normed view.
    static func callAsFunction(_ h: MLXArray, _ r: MLXArray, _ w: MLXArray, eps: Float)
        -> (h: MLXArray, hn: MLXArray)
    {
        let outputs = kernel(eps: eps)(
            [h.reshaped(-1), r.reshaped(-1), w],
            template: [("T_", h.dtype), ("DIM", h.dim(-1))],
            grid: (256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [h.shape, h.shape],
            outputDTypes: [h.dtype, h.dtype])
        return (outputs[0], outputs[1])
    }

    /// Probe the kernel against the portable add + ``MapleRMSNorm`` semantics.
    static func probe(dimensions dim: Int, dtype: DType, weight: MLXArray, eps: Float) -> Bool {
        mapleKernelMatches(
            fast: {
                let x = MLXRandom.normal([1, 1, dim], key: MLXRandom.key(0)).asType(dtype)
                let r = MLXRandom.normal([1, 1, dim], key: MLXRandom.key(1)).asType(dtype)
                let (h, hn) = callAsFunction(x, r, weight, eps: eps)
                return [h, hn]
            },
            reference: {
                let x = MLXRandom.normal([1, 1, dim], key: MLXRandom.key(0)).asType(dtype)
                let r = MLXRandom.normal([1, 1, dim], key: MLXRandom.key(1)).asType(dtype)
                let h = x + r
                let hn = MLXFast.rmsNorm(
                    h.asType(.float32), weight: weight.asType(.float32), eps: eps
                ).asType(dtype)
                return [h, hn]
            })
    }
}

// MARK: - Fused per-head Q/K RMSNorm + partial RoPE

/// Fused per-head RMSNorm + partial RoPE for single-token decode.
///
/// One dispatch replaces q_norm, k_norm, and both rope applications. One
/// simdgroup per head: normalize `headDim` values, scale by the head's norm
/// weight, and rotate the first `ropeDim` dims (non-traditional pairing
/// `i, i + R/2`) at the given position. NoPE layers pass `ropeDim == 0`.
enum MapleQKNormRoPEKernel {
    static let kernel = MLXFast.metalKernel(
        name: "maple_qk_norm_rope",
        inputNames: ["x", "w", "inv_freq", "pos_eps"],
        outputNames: ["out"],
        source: """
            uint head = thread_position_in_grid.y;
            uint lane = thread_position_in_grid.x;

            constexpr int per_lane = HEAD_DIM / 32;
            const device T_* xh = x + head * HEAD_DIM;
            const device T_* wh = w + head * HEAD_DIM;
            device T_* oh = out + head * HEAD_DIM;

            float ss = 0.0f;
            for (int i = 0; i < per_lane; ++i) {
                float v = (float)xh[lane * per_lane + i];
                ss += v * v;
            }
            ss = simd_sum(ss);
            float pos = pos_eps[0];
            float eps = pos_eps[1];
            float scale = metal::rsqrt(ss / HEAD_DIM + eps);

            for (int i = 0; i < per_lane; ++i) {
                int j = lane * per_lane + i;
                float v = (float)xh[j] * scale * (float)wh[j];
                if (ROPE_DIM > 0 && j < ROPE_DIM) {
                    constexpr int rhalf = ROPE_DIM > 0 ? ROPE_DIM / 2 : 1;
                    int p = j < rhalf ? j : j - rhalf;
                    float theta = pos * inv_freq[p];
                    float c = metal::cos(theta);
                    float s = metal::sin(theta);
                    int j2 = j < rhalf ? j + rhalf : j - rhalf;
                    float u = (float)xh[j2] * scale * (float)wh[j2];
                    v = j < rhalf ? (v * c - u * s) : (v * c + u * s);
                }
                oh[j] = (T_)v;
            }
            """)

    /// Apply the fused norm/rotation to `qk` `[nHeads, headDim]`.
    ///
    /// - Parameters:
    ///   - qk: concatenated query/key rows, one per attention/kv head
    ///   - weights: per-head norm weights broadcast to `[nHeads, headDim]`
    ///   - invFreq: rope inverse frequencies (or a one-element dummy for NoPE)
    ///   - offset: rope position (cache offset)
    ///   - eps: RMSNorm epsilon
    ///   - headDim: per-head dimension; must be a multiple of 32
    ///   - ropeDim: rotated prefix length; 0 for NoPE layers
    static func callAsFunction(
        _ qk: MLXArray, weights: MLXArray, invFreq: MLXArray, offset: Int, eps: Float,
        headDim: Int, ropeDim: Int
    ) -> MLXArray {
        let posEps = MLXArray([Float(offset), eps] as [Float])
        return kernel(
            [qk, weights, invFreq, posEps],
            template: [("T_", qk.dtype), ("HEAD_DIM", headDim), ("ROPE_DIM", ropeDim)],
            grid: (32, qk.dim(0), 1),
            threadGroup: (32, 1, 1),
            outputShapes: [qk.shape],
            outputDTypes: [qk.dtype])[0]
    }
}

// MARK: - Fused MoE router

/// Router GEMV + float32 softmax + top-8 + renormalization in ONE dispatch.
///
/// `numExperts / 32` threadgroups each compute 32 logits in float32 and publish
/// them through an atomic-float scratch (plain device stores are not reliably
/// visible across threadgroups on Apple GPUs); the last threadgroup to arrive
/// does the softmax + top-8 + renorm.
///
/// `counter` is a persistent arrival counter private to each gate instance:
/// every dispatch must see it at zero, so the electing threadgroup resets it on
/// its way out. Election on a stale counter would read unwritten scratch, so
/// nothing else may share the buffer.
enum MapleFusedRouterKernel {
    static let kernel = MLXFast.metalKernel(
        name: "maple_fused_router",
        inputNames: ["x", "w", "ctr_in"],
        outputNames: ["out_indices", "out_scores", "logits_scratch"],
        source: """
        constexpr uint NE = NEXP;
        constexpr uint D = DIM;
        constexpr uint NTG = NE / 32u;
        constexpr uint TM = 4u;
        constexpr uint TN = 4u;
        constexpr uint BLOCKN = 32u * TN;
        constexpr uint NITER = D / BLOCKN;

        uint tid = thread_position_in_threadgroup.x;
        uint tgid = threadgroup_position_in_grid.x;
        uint n_threads = 256u;
        uint sg_id = tid / 32u;
        uint lane = tid % 32u;
        uint n_sg = n_threads / 32u;

        uint row0 = tgid * (n_sg * TM) + sg_id * TM;
        float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint bn = lane * TN;
        for (uint i = 0u; i < NITER; ++i) {
            float v[TN];
            for (uint tn = 0u; tn < TN; ++tn) v[tn] = float(x[bn + tn]);
            for (uint tm = 0u; tm < TM; ++tm) {
                const device T_* wrow = w + (ulong)(row0 + tm) * D;
                T_ inter[TN];
                for (uint tn = 0u; tn < TN; ++tn) inter[tn] = wrow[bn + tn];
                for (uint tn = 0u; tn < TN; ++tn) result[tm] += inter[tn] * v[tn];
            }
            bn += BLOCKN;
        }
        for (uint tm = 0u; tm < TM; ++tm) {
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }
        device atomic_float* ls = (device atomic_float*)logits_scratch;
        if (lane == 0u) {
            for (uint tm = 0u; tm < TM; ++tm) {
                atomic_store_explicit(&ls[row0 + tm], result[tm],
                                      memory_order_relaxed);
            }
        }

        threadgroup_barrier(mem_flags::mem_device);
        threadgroup uint last_flag;
        if (tid == 0u) {
            device atomic_uint* ctr = (device atomic_uint*)ctr_in;
            uint prev = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
            uint last = (prev == NTG - 1u) ? 1u : 0u;
            if (last == 1u) atomic_store_explicit(ctr, 0u, memory_order_relaxed);
            last_flag = last;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (last_flag == 0u) return;
        threadgroup_barrier(mem_flags::mem_device);

        float my_max = -1e30f;
        for (uint e = tid; e < NE; e += n_threads) {
            float v = atomic_load_explicit(&ls[e], memory_order_relaxed);
            if (v > my_max) my_max = v;
        }
        for (int off = 16; off > 0; off >>= 1) {
            float other = simd_shuffle_down(my_max, off);
            if (other > my_max) my_max = other;
        }
        threadgroup float sg_red[16];
        if (lane == 0u) sg_red[sg_id] = my_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float m = sg_red[0];
            for (uint s = 1u; s < n_sg; s++) if (sg_red[s] > m) m = sg_red[s];
            sg_red[0] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lmax = sg_red[0];

        threadgroup float scores[NE];
        float my_sum = 0.0f;
        for (uint e = tid; e < NE; e += n_threads) {
            float lv = atomic_load_explicit(&ls[e], memory_order_relaxed);
            float v = metal::exp(lv - lmax);
            scores[e] = v;
            my_sum += v;
        }
        for (int off = 16; off > 0; off >>= 1) {
            my_sum += simd_shuffle_down(my_sum, off);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) sg_red[sg_id] = my_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float ssum = sg_red[0];
            for (uint i = 1u; i < n_sg; i++) ssum += sg_red[i];
            sg_red[0] = ssum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_total = 1.0f / (sg_red[0] + 1e-20f);
        for (uint e = tid; e < NE; e += n_threads) {
            scores[e] = scores[e] * inv_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup int topk_idx[8];
        threadgroup float topk_val[8];
        threadgroup uint8_t used[NE];
        for (uint e = tid; e < NE; e += n_threads) used[e] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < 8; k++) {
            float my_best = -1e30f;
            int my_idx = 0;
            for (int e = int(tid); e < int(NE); e += int(n_threads)) {
                if (!used[e] && scores[e] > my_best) {
                    my_best = scores[e];
                    my_idx = e;
                }
            }
            for (int off = 16; off > 0; off >>= 1) {
                float other_v = simd_shuffle_down(my_best, off);
                int other_i = simd_shuffle_down(my_idx, off);
                if (other_v > my_best) { my_best = other_v; my_idx = other_i; }
            }
            threadgroup float sg_vals[16];
            threadgroup int sg_idxs[16];
            if (lane == 0u) { sg_vals[sg_id] = my_best; sg_idxs[sg_id] = my_idx; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float bv = sg_vals[0]; int bi = sg_idxs[0];
                for (uint s = 1u; s < n_sg; s++) {
                    if (sg_vals[s] > bv) { bv = sg_vals[s]; bi = sg_idxs[s]; }
                }
                topk_val[k] = bv; topk_idx[k] = bi;
                used[bi] = 1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid < 8u) {
            float sel_sum = 0.0f;
            for (int i = 0; i < 8; i++) sel_sum += topk_val[i];
            out_indices[tid] = topk_idx[tid];
            out_scores[tid] = float(topk_val[tid] / (sel_sum + 1e-20f));
        }
        """)

    /// The kernel hard-codes 8 selected experts, `numExperts / 32`
    /// threadgroups, and a 128-wide inner GEMV block.
    static func supports(topK: Int, numExperts: Int, hiddenSize: Int) -> Bool {
        topK == 8 && numExperts > 0 && numExperts % 32 == 0 && hiddenSize > 0
            && hiddenSize % 128 == 0
    }

    /// Route a single token. `x` must have exactly `hiddenSize` elements;
    /// returns `(indices, scores)` shaped `[..., topK]` like the portable gate.
    static func callAsFunction(
        _ x: MLXArray, weight: MLXArray, counter: MLXArray, topK: Int, numExperts: Int,
        hiddenSize: Int
    ) -> (indices: MLXArray, scores: MLXArray) {
        let outputs = kernel(
            [x.reshaped(-1), weight, counter],
            template: [("T_", weight.dtype), ("NEXP", numExperts), ("DIM", hiddenSize)],
            grid: ((numExperts / 32) * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[topK], [topK], [numExperts]],
            outputDTypes: [.int32, .float32, .float32])
        let shape = Array(x.shape.dropLast()) + [topK]
        return (outputs[0].reshaped(shape), outputs[1].reshaped(shape))
    }
}
