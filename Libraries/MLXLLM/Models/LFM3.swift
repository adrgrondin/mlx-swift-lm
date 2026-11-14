// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// LiquidAI's LFM3 text model.
public final class LFM3Model: LFM3LanguageModel, LLMModel {
    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        LFM3MessageGenerator()
    }
}
