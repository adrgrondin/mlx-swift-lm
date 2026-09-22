// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Default-on rollback switch for the Qwen 3.5/3.6 four-projection GDN fusion.
package let qwen35FourGDNEnabled: Bool = {
    let raw = ProcessInfo.processInfo.environment["MLX_QWEN_FOUR_GDN"]?
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return raw != "0" && raw != "false" && raw != "off"
}()

/// A fused quantized projection and checkpoint-shaped views into its storage.
///
/// The views let a model keep its public/checkpoint module topology without
/// retaining a second physical copy of the quantized weights.
package struct FusedQuantizedLinearProjection {
    package let fused: QuantizedLinear
    package let sourceViews: [QuantizedLinear]
}

/// Fusion failed before any source modules were replaced.
package struct FusedQuantizedLinearConstructionError: Error, CustomStringConvertible {
    package let underlyingError: any Error

    package var description: String {
        "unable to construct fused projection (\(underlyingError)); the original projections are unchanged"
    }
}

/// A fused projection could not replace its source modules atomically.
///
/// `rollbackError` is non-nil only when restoring the original source modules
/// also failed. The model must not be used in that case.
package struct FusedQuantizedLinearPreparationError: Error, CustomStringConvertible {
    package let installationError: any Error
    package let rollbackError: (any Error)?

    package var description: String {
        if let rollbackError {
            return "unable to install fused projection views (\(installationError)); "
                + "restoring the original projections also failed (\(rollbackError))"
        }
        return "unable to install fused projection views (\(installationError)); "
            + "the original projections were restored"
    }
}

/// Lifecycle state for a physical projection derived from several registered
/// quantized linears.
///
/// The cache is intentionally not synchronized. Its only mutating operations
/// run during model loading or an explicit model update, both of which require
/// exclusive access. Inference only reads ``fused`` after the model has been
/// published, avoiding a lock in the token-generation hot path.
package final class FusedQuantizedLinearProjectionCache {
    private enum State {
        case unprepared
        case preparing
        case ready
        case ineligible
        case failedRestoration(FusedQuantizedLinearPreparationError)
    }

    private var state = State.unprepared
    package private(set) var fused: QuantizedLinear?

    package init() {}

    package var isPrepared: Bool {
        if case .ready = state { return fused != nil }
        return false
    }

    /// Drop derived state after any source-module or source-parameter update.
    /// Invalidations caused by installing the cache's own storage-sharing views
    /// are ignored; the preparation transaction publishes `fused` only after
    /// every view has been installed successfully.
    package func invalidate() {
        if case .preparing = state { return }
        fused = nil
        state = .unprepared
    }

    /// Build and publish the fused projection once for the current source
    /// topology. A failed or ineligible attempt is terminal until a subsequent
    /// source update calls ``invalidate()``.
    @discardableResult
    package func prepare(
        enabled: Bool,
        linears: [Linear],
        materialize: ([MLXArray]) -> Void = { eval($0) },
        installSourceModules: ([Linear]) throws -> Void
    ) throws -> Bool {
        if case .failedRestoration(let error) = state { throw error }
        guard enabled else { return false }

        switch state {
        case .ready:
            return fused != nil
        case .preparing, .ineligible:
            return false
        case .failedRestoration(let error):
            throw error
        case .unprepared:
            break
        }

        state = .preparing
        defer {
            if case .preparing = state {
                fused = nil
                state = .ineligible
            }
        }
        let projection: FusedQuantizedLinearProjection?
        do {
            projection = try fuseQuantizedLinearProjections(linears, materialize: materialize)
        } catch {
            throw FusedQuantizedLinearConstructionError(underlyingError: error)
        }
        guard let projection, projection.sourceViews.count == linears.count
        else {
            state = .ineligible
            return false
        }

        do {
            try withError { try installSourceModules(projection.sourceViews) }
        } catch let installationError {
            let rollbackError: (any Error)?
            do {
                try withError { try installSourceModules(linears) }
                rollbackError = nil
            } catch let error {
                rollbackError = error
            }
            fused = nil
            let error = FusedQuantizedLinearPreparationError(
                installationError: installationError,
                rollbackError: rollbackError)
            state = rollbackError == nil ? .ineligible : .failedRestoration(error)
            throw error
        }

        fused = projection.fused
        state = .ready
        return true
    }
}

/// Coalesce compatible quantized linears along their output dimension.
///
/// This is intentionally stricter than merely casting to `QuantizedLinear`:
/// custom subclasses may add behavior, adapters, or state that cannot be
/// represented by one stock quantized matmul. Incompatible inputs return
/// `nil` and callers keep their original projections.
package func fuseQuantizedLinearProjections(
    _ linears: [Linear],
    materialize: ([MLXArray]) -> Void = { eval($0) }
) throws -> FusedQuantizedLinearProjection? {
    guard linears.count > 1 else { return nil }

    let projections = linears.compactMap { $0 as? QuantizedLinear }
    guard projections.count == linears.count,
        zip(linears, projections).allSatisfy({ linear, projection in
            ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self)
                && linear === projection
        }),
        let first = projections.first,
        first.bias == nil,
        first.weight.ndim == 2,
        first.weight.dim(0) == first.shape.0,
        first.scales.ndim == 2,
        first.scales.dim(0) == first.shape.0,
        first.biases == nil || first.biases?.shape == first.scales.shape
    else {
        return nil
    }

    let hasQuantizationBiases = first.biases != nil
    guard
        projections.allSatisfy({ projection in
            projection.bias == nil
                && projection.bits == first.bits
                && projection.groupSize == first.groupSize
                && projection.mode == first.mode
                && projection.shape.1 == first.shape.1
                && projection.weight.ndim == 2
                && projection.weight.dim(0) == projection.shape.0
                && projection.weight.dim(1) == first.weight.dim(1)
                && projection.weight.dtype == first.weight.dtype
                && projection.scales.ndim == 2
                && projection.scales.dim(0) == projection.shape.0
                && projection.scales.dim(1) == first.scales.dim(1)
                && projection.scales.dtype == first.scales.dtype
                && (projection.biases != nil) == hasQuantizationBiases
                && (projection.biases == nil || projection.biases?.shape == projection.scales.shape)
                && projection.biases?.dtype == first.biases?.dtype
        })
    else {
        return nil
    }

    // Scope the handler here: loading may run on a dispatch worker rather than
    // the caller's Swift task. Never consume a tensor after an MLX error.
    return try withError { error in
        let fusedWeight = concatenated(projections.map(\.weight), axis: 0)
        try error.check()
        let fusedScales = concatenated(projections.map(\.scales), axis: 0)
        try error.check()
        let fusedBiases =
            hasQuantizationBiases
            ? concatenated(projections.compactMap(\.biases), axis: 0)
            : nil
        try error.check()

        materialize([fusedWeight, fusedScales] + (fusedBiases.map { [$0] } ?? []))
        try error.check()

        let fused = QuantizedLinear(
            weight: fusedWeight,
            bias: nil,
            scales: fusedScales,
            biases: fusedBiases,
            groupSize: first.groupSize,
            bits: first.bits,
            mode: first.mode)
        fused.freeze()

        var start = 0
        let sourceViews = try projections.map { projection in
            let end = start + projection.shape.0
            defer { start = end }

            let rows = start ..< end
            let weight = fusedWeight[rows]
            try error.check()
            let scales = fusedScales[rows]
            try error.check()
            let biases = fusedBiases.map { $0[rows] }
            try error.check()
            let view = QuantizedLinear(
                weight: weight,
                bias: nil,
                scales: scales,
                biases: biases,
                groupSize: first.groupSize,
                bits: first.bits,
                mode: first.mode)
            view.freeze()
            return view
        }

        // Realize the storage-sharing slices before replacing any originals.
        eval(sourceViews)
        try error.check()
        return FusedQuantizedLinearProjection(fused: fused, sourceViews: sourceViews)
    }
}
