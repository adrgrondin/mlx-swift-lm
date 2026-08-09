// Copyright © 2026 Apple Inc.

import Foundation

/// A model that can precompile native (compiled) functions at load time.
///
/// Conforming models can precompile compiled Metal graphs and free converted
/// weights after loading, eliminating the per-shape JIT cost on the first
/// forward pass and reducing peak memory. `loadWeights` calls
/// ``precompileNativeFunctions()`` after weights are loaded and modules are
/// replaced (after `update(parameters:)` and `eval(model)`).
///
/// This is the load-time hook for the Gemma 4 QAT mobile native compiled path
/// (Phase 5 of `GEMMA4_QAT_MOBILE_NATIVE_MATMUL_SWIFT_PLAN.md`), mirroring the
/// Python `precompile_native_functions`. Implementations should be no-ops if
/// the native path is not usable (e.g. unaligned dims, MoE, no PLE).
public protocol NativePrecompilable {
    /// Precompile native compiled functions and free converted weights.
    ///
    /// Called by `loadWeights` after `update(parameters:)` and `eval(model)`.
    /// Implementations should be no-ops if the native path is not usable.
    func precompileNativeFunctions()
}
