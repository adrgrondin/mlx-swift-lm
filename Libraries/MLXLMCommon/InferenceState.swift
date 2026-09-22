// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

package struct InferenceStatePreparationFailure {
    package let modelType: String
    package let error: any Error
}

package struct InferenceStatePreparationReport {
    package let failures: [InferenceStatePreparationFailure]

    package var succeeded: Bool { failures.isEmpty }
}

private let inferenceStateLogger = Logger(
    subsystem: "mlx-swift-lm", category: "inference-state")

/// Prepare a language model after checkpoint loading or an explicit topology
/// update.
///
/// Preparation failures are logged and reported, not assumed recoverable.
/// `BaseLanguageModel` values outside the inference lifecycle, such as rerankers,
/// require no preparation.
@discardableResult
package func prepareInferenceState(
    in model: BaseLanguageModel
) -> InferenceStatePreparationReport {
    guard let languageModel = model as? any LanguageModel else {
        return InferenceStatePreparationReport(failures: [])
    }

    do {
        try languageModel.prepare()
        return InferenceStatePreparationReport(failures: [])
    } catch {
        let modelType = String(reflecting: type(of: languageModel))
        inferenceStateLogger.error(
            "Failed to prepare inference state for \(modelType): \(String(describing: error))")
        return InferenceStatePreparationReport(failures: [
            .init(modelType: modelType, error: error)
        ])
    }
}

/// Prepare derived state and realize a fully loaded model before publication.
///
/// All custom checkpoint loaders should finalize through this function so
/// inference-only optimizations are applied consistently. Ordinary materialization
/// and unrecognized preparation errors propagate. Only fusion failures that leave
/// the original projections intact permit unfused inference.
@discardableResult
package func materializeModelForInference(
    _ model: BaseLanguageModel
) throws -> InferenceStatePreparationReport {
    // Validate the original model before an optional evaluation can fail and
    // leave a lazy source array in an unusable state.
    try withError { eval(model) }
    let report = prepareInferenceState(in: model)
    for failure in report.failures {
        switch failure.error {
        case is FusedQuantizedLinearConstructionError:
            break
        case let error as FusedQuantizedLinearPreparationError where error.rollbackError == nil:
            break
        default:
            throw failure.error
        }
    }
    try withError { eval(model) }
    return report
}
