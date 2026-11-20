// Copyright © 2026 Apple Inc.

import MLXScriptedLM
import Testing

@testable import MLXLMCommon

struct Inkling2ToolCallTests {
    private let header = "<|message_model|>"
    private let invoke = "<|content_invoke_tool_json|>"
    private let end = "<|end_message|>"
    private let payload = #"{"name":"get_weather","args":{"city":"Paris"}}"#
    private let tools: [ToolSpec] = [
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object", "properties": ["city": ["type": "string"]],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    @Test func framesKeepToolCallsOutOfTextAndReasoning() {
        let frame = header + "get_weather" + invoke + payload
        let events = decode(
            header + "<|content_thinking|>Check. " + frame + end
                + header + "<|content_text|>Looking it up." + end
                + frame + end + frame + end,
            tools: tools)
        let calls = events.compactMap { if case .toolCall(let call) = $0 { call } else { nil } }
        let answer = events.compactMap { if case .response(let text) = $0 { text } else { nil } }
        let reasoning = events.compactMap {
            if case .reasoning(let text) = $0 { text } else { nil }
        }
        #expect(answer.joined() == "Looking it up.")
        #expect(reasoning.joined().hasPrefix("Check. "))
        #expect(calls.count == 2)
        for call in calls {
            #expect(call.function.name == "get_weather")
            #expect(call.function.arguments == ["city": .string("Paris")])
            #expect(call.id != nil)
        }
    }

    @Test func rejectsMismatchedUnauthorizedInvalidAndIncompleteFrames() {
        let mismatched = header + "other" + invoke + payload + end
        let undeclared = header + "other" + invoke + #"{"name":"other","args":{}}"# + end
        let invalid = header + "get_weather" + invoke + #"{"name":"get_weather","args":{}}"# + end
        let incomplete = header + "get_weather" + invoke + payload
        let events = decode(mismatched + undeclared + invalid + incomplete, tools: tools)
        let rejections = events.compactMap {
            if case .rejectedToolCall(let rejection) = $0 { rejection.reason } else { nil }
        }
        #expect(events.count == 4)
        #expect(
            rejections == [.malformedSyntax, .undeclaredTool, .invalidArguments, .incompleteOutput])
    }

    @Test func doesNotAuthorizeCallsWithoutToolDeclarations() {
        let events = decode(header + "get_weather" + invoke + payload + end, tools: nil)
        #expect(events.count == 1)
        if case .rejectedToolCall(let rejection) = events.first {
            #expect(rejection.reason == .undeclaredTool)
        } else {
            Issue.record("Expected an undeclared-tool rejection")
        }
    }

    @Test func stopStringsCancelCallsBeforeDispatch() {
        let frame = header + "get_weather" + invoke + payload + end
        for stop in ["Paris", "s\"}}" + end, invoke] {
            #expect(decode(frame, tools: tools, stopStrings: [stop]) == [.stop])
        }
    }

    private func decode(
        _ wire: String, tools: [ToolSpec]?, stopStrings: Set<String> = []
    ) -> [TokenStreamEvent] {
        let tokenizer = ScriptedTokenizer(
            corpus: [wire],
            specials: [header, invoke, end, "<|content_thinking|>", "<|content_text|>"],
            configuration: .init(maxPieceLength: 2))
        var decoder = ToolCallFormat.inkling2.makeTokenStreamDecoder(
            tokenizer: tokenizer, tools: tools, stopStrings: stopStrings,
            toolCallPolicy: .init(validation: .strict))
        var events = [TokenStreamEvent]()
        for token in tokenizer.encode(text: wire, addSpecialTokens: false) {
            if !decoder.push(
                token,
                emit: {
                    events.append($0)
                    return true
                })
            {
                break
            }
        }
        _ = decoder.finish {
            events.append($0)
            return true
        }
        return events
    }
}
