// Copyright © 2026 Apple Inc.

import CoreImage
import CoreMedia
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

private let lfm3Models = IntegrationTestModels(
    downloader: #hubDownloader(), tokenizerLoader: #huggingFaceTokenizerLoader())

/// Opt in to the released checkpoints with TEST_RUNNER_MLX_RUN_LFM3_INTEGRATION=1.
@Suite(
    .serialized, .timeLimit(.minutes(10)),
    .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_LFM3_INTEGRATION"] == "1"))
struct LFM3IntegrationTests {
    @Test func textAndReasoningTemplate() async throws {
        let container = try await lfm3Models.llmContainer(for: LLMRegistry.lfm3_1_7b_4bit)
        let input = try await container.prepare(input: UserInput(prompt: "Hello"))
        let ids = input.text.tokens.asArray(Int.self)
        let rendered = await container.perform { $0.tokenizer.decode(tokenIds: ids) }
        #expect(
            rendered.hasSuffix("<|im_start|>assistant\n<think>"),
            "Unexpected generation prompt: \(rendered)")
        let session = ChatSession(
            container, generateParameters: .init(maxTokens: 128, temperature: 0),
            additionalContext: ["enable_thinking": false])
        let answer = try await session.respond(
            to: "What is the capital of France? Answer with only the city name.")
        #expect(answer.lowercased().contains("paris"), "Unexpected text response: \(answer)")
        let followup = try await session.respond(
            to: "Which country is that city in? Answer with only the country name.")
        #expect(
            followup.lowercased().contains("france"), "Unexpected cached continuation: \(followup)")
    }

    @Test func imageAndVideo() async throws {
        let container = try await lfm3Models.vlmContainer(for: VLMRegistry.lfm3_vl_2b_4bit)
        let red = VisionTestImages.solidColor(.red, size: 256)
        let blue = VisionTestImages.solidColor(.blue, size: 256)
        let imageSession = ChatSession(
            container, generateParameters: .init(maxTokens: 128, temperature: 0),
            processing: .init(), additionalContext: ["enable_thinking": false])
        let imageAnswer = try await imageSession.respond(
            to: "What color is this image? Reply with just the color.", image: .ciImage(red))
        #expect(
            imageAnswer.lowercased().contains("red"), "Unexpected image response: \(imageAnswer)")
        let frames = (0 ..< 120).map { index in
            VideoFrame(
                image: .ciImage(index < 60 ? red : blue),
                timeStamp: CMTime(value: Int64(index), timescale: 30))
        }
        let videoSession = ChatSession(
            container, generateParameters: .init(maxTokens: 128, temperature: 0),
            processing: .init(maxPixels: 262144), additionalContext: ["enable_thinking": false])
        let videoAnswer = try await videoSession.respond(
            to: "Which color appears first, and which color appears last?", video: .frames(frames))
        let lower = videoAnswer.lowercased()
        let first = lower.range(of: "red")
        let last = lower.range(of: "blue")
        #expect(
            first != nil && last != nil && first!.lowerBound < last!.lowerBound,
            "Expected red then blue; got: \(videoAnswer)")
    }
}
