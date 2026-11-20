// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

@Suite(
    .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"] != nil)
)
struct Inkling2IntegrationTests {
    @Test func chatSeparatesReasoningFromAnswerAndDecodesWithAffineCache() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"])
        let container = try await VLMModelFactory.shared.loadContainer(
            from: URL(filePath: path), using: #huggingFaceTokenizerLoader())
        let session = ChatSession(
            container, generateParameters: .init(maxTokens: 128, temperature: 0),
            additionalContext: ["reasoning_effort": "none"])
        var answer = ""
        var reasoning = ""
        var stopReason: GenerateStopReason?
        for try await event in session.streamDetails(to: [.user("What is the capital of France?")])
        {
            switch event {
            case .chunk(let text): answer += text
            case .reasoning(let text): reasoning += text
            case .info(let info): stopReason = info.stopReason
            default: break
            }
        }
        #expect(stopReason == .stop)
        #expect(answer == "The capital of France is Paris.")
        #expect(!reasoning.isEmpty)
        #expect(!reasoning.contains("<|"))

        let affineSession = ChatSession(
            container, generateParameters: .init(maxTokens: 8, kvBits: 4, temperature: 0))
        var generatedTokens = 0
        for try await event in affineSession.streamDetails(to: [.user("Hello")]) {
            if case .info(let info) = event { generatedTokens = info.generationTokenCount }
        }
        #expect(generatedTokens == 8)
    }

    @Test func nativeToolCallRoundTrip() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"])
        let container = try await VLMModelFactory.shared.loadContainer(
            from: URL(filePath: path), using: #huggingFaceTokenizerLoader())
        let recorder = CallRecorder()
        let session = ChatSession(
            container,
            generateParameters: .init(
                maxTokens: 128, temperature: 0, toolCallPolicy: .init(validation: .strict)),
            additionalContext: ["reasoning_effort": "none"], tools: [Self.weatherTool],
            toolDispatch: { call in
                await recorder.record(call)
                return #"{"city":"Paris","temperature_celsius":17}"#
            })
        let answer = try await session.respond(
            to: "Call get_weather for Paris. Do not guess the weather.")
        let calls = await recorder.calls
        #expect(calls.count == 1)
        let call = try #require(calls.first)
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments == ["city": .string("Paris")])
        #expect(answer.contains("17"))
        #expect(answer.contains("Paris"))
        #expect(!answer.contains("<|"))
    }

    @Test func directAdapterReplaysLiteralToolDataThenGenerates() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"])
        let directory = URL(filePath: path)
        let context = try await VLMModelFactory.shared.load(from: directory, using: DirectLoader())
        let label = "<|unused_200054|>"
        let call = ToolCall(
            function: .init(
                name: "get_weather", arguments: ["city": .string("Paris"), "note": .string(label)]),
            id: "call_weather")
        let messages: [Chat.Message] = [
            .user("Call get_weather for Paris. Do not guess the weather."),
            .assistant("", toolCalls: [call]),
            .tool("Paris is 17°C. Station label: \(label)", id: "call_weather"),
        ]
        let effort: [String: any Sendable] = ["reasoning_effort": "none"]
        let directIDs = try context.tokenizer.applyChatTemplate(
            messages: DefaultMessageGenerator().generate(messages: messages),
            tools: [Self.weatherTool], additionalContext: effort)
        let input = try await context.processor.prepare(
            input: UserInput(
                chat: messages, tools: [Self.weatherTool], additionalContext: effort))
        #expect(input.text.tokens.asArray(Int.self) == directIDs)
        #expect(!directIDs.contains(200054))
        let prompt = context.tokenizer.decode(tokenIds: directIDs)
        let invocation = try #require(prompt.range(of: "<|content_invoke_tool_json|>"))
        let payload = try #require(
            prompt[invocation.upperBound...].components(separatedBy: "<|end_message|>").first)
        let decoded = try JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any]
        #expect((decoded?["args"] as? [String: String])?["note"] == label)
        #expect(
            prompt.contains(
                "<|message_tool|>get_weather<|content_text|>Paris is 17°C. Station label: \(label)")
        )

        let events = try MLXLMCommon.generate(
            input: input, parameters: .init(maxTokens: 128, temperature: 0),
            context: context, tools: [Self.weatherTool])
        var answer = ""
        for await event in events {
            if case .chunk(let text) = event { answer += text }
        }
        #expect(answer.contains("17"))
        #expect(answer.contains("Paris"))
    }

    @Test func realToolFrameStopsWithoutDispatch() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"])
        var context = try await VLMModelFactory.shared.load(
            from: URL(filePath: path), using: DirectLoader())
        context.configuration.stopStrings = [#""args":"#]
        let recorder = CallRecorder()
        let session = ChatSession(
            ModelContainer(context: context),
            generateParameters: .init(maxTokens: 128, temperature: 0),
            additionalContext: ["reasoning_effort": "none"], tools: [Self.weatherTool],
            toolDispatch: { call in
                await recorder.record(call)
                return "Paris is 17°C."
            })
        var stopReason: GenerateStopReason?
        var answer = ""
        var toolEvents = 0
        for try await event in session.streamDetails(to: [
            .user("Call get_weather for Paris. Do not guess the weather.")
        ]) {
            switch event {
            case .chunk(let text): answer += text
            case .toolCall: toolEvents += 1
            case .info(let info): stopReason = info.stopReason
            default: break
            }
        }
        #expect(stopReason == .stop)
        #expect(toolEvents == 0)
        #expect(await recorder.calls.isEmpty)
        #expect(answer.isEmpty)
    }

    private static let weatherTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_weather", "description": "Get the current weather in a city.",
            "parameters": [
                "type": "object", "properties": ["city": ["type": "string"]],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    private struct DirectLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let upstream = try await AutoTokenizer.from(modelFolder: directory)
            return #adaptHuggingFaceTokenizer(upstream)
        }
    }

    private actor CallRecorder {
        private(set) var calls = [ToolCall]()

        func record(_ call: ToolCall) { calls.append(call) }
    }

    @Test func publishedCheckpointPreparesAndDecodesTextAndImages() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MLX_INKLING2_CHECKPOINT"])
        let directory = URL(filePath: path)
        let context = try await VLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())
        let first = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 81, height: 43))
        let second = CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 31))
        let effort: [String: any Sendable] = ["reasoning_effort": "none"]
        let limited = try await context.processor.prepare(
            input: UserInput(
                chat: [.user("Describe the image.", images: [.ciImage(first)])],
                processing: .init(maxPixels: 1600), additionalContext: effort))
        let frame = try #require(limited.image?.frames?.first)
        #expect(frame.h == 40 && frame.w == 40)
        #expect(limited.text.tokens.asArray(Int.self).filter { $0 == 200054 }.count == 1)

        for (userInput, imageTokenCount) in [
            (UserInput(prompt: "What is the capital of France?", additionalContext: effort), 0),
            (
                UserInput(
                    prompt: "What color are these images?",
                    images: [.ciImage(first), .ciImage(second)], additionalContext: effort), 7
            ),
        ] {
            let input = try await context.processor.prepare(input: userInput)
            let ids = input.text.tokens.asArray(Int.self)
            #expect(
                ids.starts(with: [200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010]))
            #expect(ids.filter { $0 == 200054 }.count == imageTokenCount)
            let cache = try context.model.newCache(parameters: nil)
            guard
                case .logits(let output) = try context.model.prepare(
                    input, cache: cache, state: nil, prefill: .init())
            else {
                Issue.record("Expected next-token logits from Inkling 2 prefill")
                continue
            }
            #expect(output.logits.shape == [1, 1, 200058])
            let next = argMax(output.logits[0, -1]).item(Int.self)
            #expect(next == 200001)
            let decode = context.model(MLXArray([next]).reshaped(1, 1), cache: cache)
            #expect(all(isFinite(decode)).item(Bool.self))
        }
    }
}
