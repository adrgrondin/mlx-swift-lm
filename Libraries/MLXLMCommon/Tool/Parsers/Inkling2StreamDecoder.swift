// Copyright © 2026 Apple Inc.

import Foundation

struct Inkling2ToolCallParser: ToolCallParser {
    static let header = "<|message_model|>"
    static let invocation = "<|content_invoke_tool_json|>"
    static let end = "<|end_message|>"

    let startTag: String? = Self.header
    let endTag: String? = Self.end

    private struct Payload: Decodable {
        let name: String
        let args: [String: JSONValue]
    }

    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard content.hasPrefix(Self.header), content.hasSuffix(Self.end) else { return nil }
        let body = content.dropFirst(Self.header.count).dropLast(Self.end.count)
        guard let marker = body.range(of: Self.invocation) else { return nil }
        let name = body[..<marker.lowerBound]
        guard !name.isEmpty,
            let payload = try? JSONDecoder().decode(
                Payload.self, from: Data(body[marker.upperBound...].utf8)),
            payload.name == name
        else { return nil }
        return ToolCall(function: .init(name: payload.name, arguments: payload.args))
    }
}

struct Inkling2StreamDecoder: TokenStreamDecoder {
    private typealias Frame = Inkling2ToolCallParser

    private enum State {
        case text, header, invocation, discard, stopped
    }

    private let tokenizer: any Tokenizer
    private let headerID: Int
    private let invocationID: Int
    private let endID: Int
    private let textControls: Set<Int>
    private let toolProcessor: ToolCallProcessor
    private var text: StandardTokenStreamDecoder
    private var frameStops: StopStringFilter
    private var frameDetokenizer: (any StreamingDetokenizer)?
    private var state = State.text
    private var name = ""
    private var tokens = [Int]()

    init?(
        tokenizer: any Tokenizer, tools: [[String: any Sendable]]?, stopStrings: Set<String>,
        policy: ToolCallPolicy, reasoning: (config: ReasoningConfig, primedInside: Bool)?
    ) {
        guard let header = tokenizer.singleTokenID(for: Frame.header),
            let invocation = tokenizer.singleTokenID(for: Frame.invocation),
            let end = tokenizer.singleTokenID(for: Frame.end),
            Set([header, invocation, end]).count == 3
        else { return nil }
        self.tokenizer = tokenizer
        headerID = header
        invocationID = invocation
        endID = end
        let reasoning = reasoning ?? (.inkling2, false)
        textControls = Set(
            ([reasoning.config.startDelimiter] + Array(reasoning.config.suppressedControlTokens))
                .compactMap { tokenizer.singleTokenID(for: $0) })
        toolProcessor = ToolCallProcessor(
            format: .inkling2, tools: tools ?? [], toolCallPolicy: policy)
        frameStops = StopStringFilter(stopStrings: stopStrings)
        text = StandardTokenStreamDecoder(
            tokenizer: tokenizer, format: .inkling2, tools: nil, stopStrings: stopStrings,
            reasoning: reasoning, parsesToolCalls: false)
    }

    var isInsideReasoning: Bool { text.isInsideReasoning }
    var rejectedToolCallCount: Int { toolProcessor.rejectedToolCallCount }
    var canEndForSteering: Bool { state == .text && text.canEndForSteering }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        if state == .stopped { return false }
        if token == headerID, !isInsideReasoning {
            if state == .invocation {
                if frameBoundaryMatchesStop() { return stopFrame(emit: emit) }
                if !finishCall(.incompleteOutput, emit: emit) { return false }
            }
            tokens.removeAll(keepingCapacity: true)
            state = .header
            return text.push(token, emit: emit)
        }
        switch state {
        case .text:
            return text.push(token, emit: emit)
        case .header:
            if token == invocationID {
                name = tokenizer.decode(tokenIds: tokens)
                tokens.removeAll(keepingCapacity: true)
                state = .invocation
                if frameStops.isEnabled {
                    frameStops.buffer = ""
                    frameDetokenizer = tokenizer.makeStreamingDetokenizer()
                    if frameStops.process(Frame.header + name + Frame.invocation).stopped {
                        return stopFrame(emit: emit)
                    }
                }
            } else if token == endID || textControls.contains(token) {
                tokens.removeAll(keepingCapacity: true)
                state = .text
                return text.push(token, emit: emit)
            } else if tokens.count < 128 {
                tokens.append(token)
            } else {
                tokens.removeAll(keepingCapacity: true)
                state = .discard
            }
        case .invocation:
            if token == endID {
                if frameBoundaryMatchesStop(Frame.end) { return stopFrame(emit: emit) }
                state = .text
                return finishCall(nil, emit: emit) && text.push(token, emit: emit)
            }
            frameDetokenizer?.append(token: token)
            if let fragment = frameDetokenizer?.next(), frameStops.process(fragment).stopped {
                return stopFrame(emit: emit)
            }
            guard tokens.count < 32_768 else {
                state = .discard
                return finishCall(.resourceLimitExceeded, emit: emit)
            }
            tokens.append(token)
        case .discard:
            if token == endID {
                state = .text
                return text.push(token, emit: emit)
            }
        case .stopped:
            return false
        }
        return true
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if state == .stopped { return true }
        if state == .invocation {
            if frameBoundaryMatchesStop() { return stopFrame(emit: emit) }
            if !finishCall(.incompleteOutput, emit: emit) { return false }
        }
        state = .text
        return text.finish(emit: emit)
    }

    private mutating func frameBoundaryMatchesStop(_ marker: String = "") -> Bool {
        guard frameStops.isEnabled else { return false }
        let remaining = frameDetokenizer?.finish() ?? ""
        return frameStops.process(remaining + marker).stopped
    }

    private mutating func stopFrame(emit: (TokenStreamEvent) -> Bool) -> Bool {
        state = .stopped
        tokens.removeAll(keepingCapacity: false)
        frameDetokenizer = nil
        _ = emit(.stop)
        return false
    }

    private mutating func finishCall(
        _ rejection: RejectedToolCall.Reason?, emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        let body = tokenizer.decode(tokenIds: tokens)
        frameDetokenizer = nil
        tokens.removeAll(keepingCapacity: true)
        let raw =
            Frame.header + name + Frame.invocation + body + (rejection == nil ? Frame.end : "")
        return toolProcessor.processFramedCall(raw, rejection: rejection).allSatisfy { output in
            switch output {
            case .toolCall(let call): emit(.toolCall(call))
            case .rejectedToolCall(let rejected): emit(.rejectedToolCall(rejected))
            case .response(let text): emit(.response(text))
            }
        }
    }
}

extension ReasoningConfig {
    package static var inkling2: Self {
        var config = Self(
            startDelimiter: "<|content_thinking|>", endDelimiter: Inkling2ToolCallParser.end,
            promptStrategy: .none, isSpecialToken: true)
        config.suppressedControlTokens = [
            Inkling2ToolCallParser.header, "<|content_text|>", "<|content_tool_error|>",
            "<|content_invoke_tool_text|>", Inkling2ToolCallParser.end,
        ]
        return config
    }
}
