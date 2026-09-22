// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import Metal
import XCTest

/// Lazy graphs that fail at compilation or launch, not at graph construction.
/// This tests error transport, not the cause of any field Metal failure.
enum MLXEvaluationFailureFixture {
    static let marker = "intentional_materialization_failure"

    static func array(shape: [Int] = [1], dtype: DType = .float32) -> MLXArray {
        let kernel = MLXFast.metalKernel(
            name: "materialization_failure_fixture",
            inputNames: [String](), outputNames: ["output"],
            source: "\n#error \(marker)\n")
        return kernel(
            [], grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [shape], outputDTypes: [dtype], stream: .gpu)[0]
    }

    static func rejectedLaunchArray() throws -> MLXArray {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let threads = device.maxThreadsPerThreadgroup.width + 1
        let kernel = MLXFast.metalKernel(
            name: "rejected_launch_fixture",
            inputNames: [String](), outputNames: ["output"],
            source: "if (thread_position_in_grid.x == 0) { output[0] = 1; }")
        return kernel(
            [], grid: (threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[1]], outputDTypes: [.float32], stream: .gpu)[0]
    }

    static func assertRejectedLaunchError(
        _ error: any Error, file: StaticString = #filePath, line: UInt = #line
    ) {
        let error = (error as? FusedQuantizedLinearConstructionError)?.underlyingError ?? error
        XCTAssertTrue(error is MLXError, "\(error)", file: file, line: line)
        XCTAssertTrue(
            error.localizedDescription.contains("maximum allowed threads per threadgroup"),
            "\(error)", file: file, line: line)
    }

    static func assertExpectedError(
        _ error: any Error, file: StaticString = #filePath, line: UInt = #line
    ) {
        let error = (error as? FusedQuantizedLinearConstructionError)?.underlyingError ?? error
        XCTAssertTrue(error is MLXError, "\(error)", file: file, line: line)
        XCTAssertTrue(
            error.localizedDescription.contains(marker), "\(error)", file: file, line: line)
    }
}
