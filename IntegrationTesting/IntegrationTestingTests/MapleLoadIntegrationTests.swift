// Cache-gated integration coverage for the released Maple checkpoint.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
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

    /// Greedy decode with the fused kernels must produce the same token ids as
    /// the portable path on the real checkpoint.
    @Test(.enabled(if: hfSnapshotDir(modelId: mapleModelID) != nil))
    func testFusedDecodeMatchesPortableTokenIds() async throws {
        let directory = try #require(hfSnapshotDir(modelId: mapleModelID))
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())
        let model = try #require(context.model as? MapleModel)

        func decodeTokenIds() -> [Int] {
            let cache = model.newCache(parameters: nil)
            let prompt = MLXArray([9707, 11, 1879, 330, 13339] as [Int32]).reshaped(1, 5)
            var token = model(prompt, cache: cache)
            var ids: [Int] = []
            for _ in 0 ..< 8 {
                let next = argMax(token[0..., token.dim(1) - 1, 0...], axis: -1)
                eval(next)
                ids.append(Int(next.item(Int32.self)))
                token = model(next.reshaped(1, 1), cache: cache)
            }
            return ids
        }

        let fusedIds = decodeTokenIds()
        // The first decode probed and latched every fast path; force them off.
        model.disableFusedDecodeKernels()
        let portableIds = decodeTokenIds()
        #expect(fusedIds == portableIds)
    }
}
