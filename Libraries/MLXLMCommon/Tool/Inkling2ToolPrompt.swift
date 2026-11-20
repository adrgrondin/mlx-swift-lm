// Copyright © 2026 Apple Inc.

import Foundation

/// The Inkling 2 tool prompt renderer used by the tokenizer bridge and model processor.
public enum _Inkling2ToolPrompt {
    private struct PromptError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    private typealias Frame = Inkling2ToolCallParser
    private static let image = "<|unused_200054|>"
    private static let endSampling = "<|content_model_end_sampling|>"

    /// Returns nil when the tokenizer is not Inkling 2 or the prompt has no tool data.
    public static func renderIfNeeded(
        tokenizer: any Tokenizer, messages: [[String: any Sendable]], tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]? {
        guard
            tools?.isEmpty == false
                || messages.contains(where: {
                    $0["role"] as? String == "tool"
                        || ($0["tool_calls"] as? [Message])?.isEmpty == false
                }), tokenizer.eosToken == endSampling,
            tokenizer.singleTokenID(for: image) == 200054,
            tokenizer.singleTokenID(for: Frame.invocation) != nil
        else { return nil }

        var prompt = Prompt(tokenizer: tokenizer)
        if let tools, !tools.isEmpty {
            let specs: [Message] = try tools.map { tool in
                let function = tool["function"] as? Message ?? tool
                return [
                    "description": function["description"] as? String ?? "",
                    "name": try name(function["name"]),
                    "parameters": function["parameters"] as? Message ?? [:],
                    "type": tool["type"] as? String ?? "function",
                ]
            }
            prompt.wire(
                "<|message_system|>tool_declare<|content_xml|>\(try json(specs))\(Frame.end)")
        }
        let effort = try effortBlock(additionalContext?["reasoning_effort"])
        var needsEffort = true
        var namesByID = [String: String]()
        for message in messages where message["role"] as? String == "assistant" {
            for call in message["tool_calls"] as? [Message] ?? [] {
                if let id = call["id"] as? String,
                    let function = call["function"] as? Message,
                    let functionName = function["name"] as? String
                {
                    namesByID[id] = functionName
                }
            }
        }
        for message in messages {
            let role = message["role"] as? String ?? ""
            let header: String
            switch role {
            case "system": header = "<|message_system|>"
            case "user": header = "<|message_user|>"
            case "assistant": header = Frame.header
            case "tool": header = "<|message_tool|>"
            default: throw PromptError("Unknown Inkling 2 message role: \(role)")
            }
            if needsEffort, role != "system" {
                prompt.wire(effort)
                needsEffort = false
            }
            let explicitName = message["name"] as? String ?? ""
            let correlatedName =
                (message["tool_call_id"] as? String).flatMap { namesByID[$0] } ?? ""
            let sender =
                switch role {
                case "tool": explicitName.isEmpty ? correlatedName : explicitName
                case "assistant": explicitName
                default: ""
                }
            let prefix = try header + (sender.isEmpty ? "" : name(sender))
            let thinking =
                role == "assistant" ? (message["reasoning_content"] as? String ?? "") : ""
            let calls = role == "assistant" ? (message["tool_calls"] as? [Message] ?? []) : []
            if !thinking.isEmpty { prompt.block(prefix + "<|content_thinking|>", thinking) }
            if let content = message["content"] as? String {
                if !content.isEmpty || (thinking.isEmpty && calls.isEmpty) {
                    prompt.block(prefix + "<|content_text|>", content)
                }
            } else if let parts = message["content"] as? [Any] {
                for part in parts {
                    if let text = part as? String {
                        prompt.block(prefix + "<|content_text|>", text)
                        continue
                    }
                    let part = part as? Message
                    switch part?["type"] as? String ?? "text" {
                    case "text", "input_text":
                        prompt.block(prefix + "<|content_text|>", part?["text"] as? String ?? "")
                    case "image", "input_image", "image_url":
                        guard role == "user" || role == "tool" else {
                            throw PromptError(
                                "Inkling 2 accepts images only in user and tool messages")
                        }
                        prompt.wire(prefix + "<|content_image|>" + image + Frame.end)
                    default:
                        throw PromptError("Unsupported content part in Inkling 2 \(role) message")
                    }
                }
            }
            for call in calls {
                let function = call["function"] as? Message
                let functionName = try name(function?["name"])
                guard sender.isEmpty || sender == functionName else {
                    throw PromptError("Inkling 2 assistant name does not match its tool call")
                }
                let args = function?["arguments"]
                if let args, !(args is Message) {
                    throw PromptError("Inkling 2 tool arguments must be a JSON object")
                }
                let payload = try
                    "{\"name\":\(json(functionName)),\"args\":\(json(args as? Message ?? [:]))}"
                prompt.wire(header + functionName + Frame.invocation + payload + Frame.end)
            }
            if role == "assistant" { prompt.wire(endSampling) }
        }
        if needsEffort { prompt.wire(effort) }
        prompt.flush()
        return prompt.ids
    }

    private static func name(_ value: Any?) throws -> String {
        guard let name = value as? String, !name.isEmpty, !name.contains("<|") else {
            throw PromptError("Invalid Inkling 2 tool name")
        }
        return name
    }

    private static func json(_ value: Any) throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: value,
            options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes])
        // Keep control spellings valid JSON data, not tokenizer special tokens.
        return String(decoding: data, as: UTF8.self).replacingOccurrences(
            of: "<|", with: #"\u003c|"#)
    }

    private static func effortBlock(_ effort: Any?) throws -> String {
        let level: Double
        if let label = effort as? String {
            let levels = [
                "none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99,
                "max": 0.99,
            ]
            guard let value = levels[label.trimmingCharacters(in: .whitespacesAndNewlines)] else {
                throw PromptError("Unknown Inkling 2 reasoning effort: \(label)")
            }
            level = value
        } else {
            level = (effort as? NSNumber)?.doubleValue ?? 0.9
        }
        guard (0 ... 0.99).contains(level) else {
            throw PromptError("Inkling 2 reasoning effort must be in [0, 0.99]")
        }
        return
            "<|message_system|><|content_text|>Thinking effort level: \(level == 0 ? "0" : String(level))\(Frame.end)"
    }

    private struct Prompt {
        let tokenizer: any Tokenizer
        var ids = [Int]()
        private var pending = ""

        init(tokenizer: any Tokenizer) { self.tokenizer = tokenizer }

        mutating func wire(_ text: String) { pending += text }

        mutating func block(_ prefix: String, _ text: String) {
            wire(prefix)
            // Break up literal control spellings without changing their decoded text.
            let parts = text.components(separatedBy: "<|")
            wire(parts[0])
            for part in parts.dropFirst() {
                wire("<")
                flush()
                wire("|" + part)
            }
            wire(Frame.end)
        }

        mutating func flush() {
            if !pending.isEmpty {
                ids += tokenizer.encode(text: pending, addSpecialTokens: false)
                pending = ""
            }
        }
    }
}
