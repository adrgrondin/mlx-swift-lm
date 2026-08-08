// Cache-gated integration coverage for the released Maple checkpoint.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

private let mapleModelID = "deepgrove/maple-2bit-mlx"

@Suite(.serialized)
struct MapleLoadIntegrationTests {
    @Test(.enabled(if: hfSnapshotDir(modelId: mapleModelID) != nil))
    func testCachedCheckpointLoadsAndGenerates() async throws {
        let directory = try #require(hfSnapshotDir(modelId: mapleModelID))
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())
        #expect(context.model is MapleModel)

        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user("Why is the sky blue? Answer briefly.")]))
        let stream = try generate(
            input: input,
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            context: context)

        var output = ""
        for await event in stream {
            if case .chunk(let text) = event { output += text }
        }
        #expect(!output.isEmpty, "Maple loaded but produced no output")
    }
}
