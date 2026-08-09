# MLXLLM

# Documentation

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations

# Contents

This is a port of several models from:

- https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/

Tokenization is provided via the `TokenizerLoader` protocol – see the main
[README](../../README.md) for available integration packages.

The [LLMModelFactory.swift](LLMModelFactory.swift) provides minor overrides and customization --
if you require overrides for the tokenizer or prompt customizations they can be
added there.

This is set up to load models from Hugging Face, e.g. https://huggingface.co/mlx-community

The following models have been tried:

- mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX
- mlx-community/Llama-3.2-1B-Instruct-4bit
- mlx-community/Llama-3.2-3B-Instruct-4bit
- deepgrove/maple-2bit-mlx
- mlx-community/Meta-Llama-3-8B-Instruct-4bit
- mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
- mlx-community/Mistral-7B-Instruct-v0.3-4bit
- mlx-community/Mistral-Nemo-Instruct-2407-4bit
- mlx-community/OpenELM-270M-Instruct
- mlx-community/Phi-3.5-MoE-instruct-4bit
- mlx-community/Phi-3.5-mini-instruct-4bit
- mlx-community/Qwen1.5-0.5B-Chat-4bit
- mlx-community/SmolLM-135M-Instruct-4bit
- mlx-community/gemma-2-2b-it-4bit
- mlx-community/gemma-2-9b-it-4bit
- mlx-community/phi-2-hf-4bit-mlx
- mlx-community/quantized-gemma-2b-it

Currently supported model types are:

- Cohere
- Gemma
- Gemma2
- InternLM2
- Llama / Mistral
- Maple
- MiniCPM (v1/v2/v4; v3 uses a different architecture)
- OpenELM
- Phi
- Phi3
- PhiMoE
- Qwen2
- Qwen3
- Qwen3-Next
- Starcoder2
- MiMo
- MiMo_v2_flash
- MiniMax
- GLM4
- GLM4MOE
- AceReason
- NemotronH

See [llm-tool](../../Tools/llm-tool)

# Quick Start

Using LLMs and VLMs from MLXLMCommon is as easy as:

```swift
import MLXLLM
import MLXLMCommon
import MLXLMHuggingFace
import MLXLMTokenizers

let model = try await loadModel(
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

## Maple

Maple has a native exact-head implementation. Load the released mixed-quantized checkpoint through the normal API; its bundled tokenizer and chat template are used automatically:

```swift
let model = try await loadModel(
    using: TokenizersLoader(),
    id: "deepgrove/maple-2bit-mlx"
)
let session = ChatSession(model)
print(try await session.respond(to: "Why is the sky blue?"))
```

The checkpoint's `model_file: "maple.py"` entry is ignored: `model_type: "maple"` selects registered Swift code and does not execute remote Python.

Single-token decode uses fused Metal kernels (residual-add + RMSNorm, per-head Q/K norm + RoPE, and the MoE router) that are probed once against the portable implementation on live weights and permanently fall back to it on any mismatch. Set `MLX_MAPLE_FUSED_KERNELS=0` to force the portable decode path.

Checkpoints carrying `flash_head` metadata also load the approximate FlashHead tensors. The exact vocabulary head remains the default; opt in to the approximate head for single-stream decode through the loaded model container:

```swift
let container = try await loadModelContainer(
    using: TokenizersLoader(),
    id: "deepgrove/maple-2bit-mlx"
)
// Opt in to the approximate FlashHead for single-stream decode.
await container.perform { context in
    (context.model as? MapleModel)?.headMode = .flash
}
```

In `.flash` mode, decode scores cluster centroids and then exactly scores only the top clusters' tokens plus forced control tokens; unscored vocabulary is `-inf`. Prefill, batched calls, and non-quantized heads always use the exact head. Greedy decoding is exact whenever the true argmax lies in the probed clusters.

The released model is a 20B-A1B sparse MoE with 2-bit transformer/expert weights and 4-bit embeddings/output head. A Mac with at least 16 GB of unified memory is recommended; available memory, prompt length, and other applications affect the practical limit. For now, producing compatible ternary checkpoints requires the Maple-specific Python converter from the [DeepGrove MLX LM fork](https://github.com/deepgrove-ai/mlx-lm); generic round-to-nearest conversion does not reproduce Maple's trained ternarization.

For more information see 
[Evaluation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/evaluation)
or [Using Models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using-model)
for more advanced API.

# Adding a Model

If the model follows the typical LLM pattern:

- `config.json`, `tokenizer.json`, and `tokenizer_config.json`
- `*.safetensors`

You can follow the pattern of the models in the [Models](Models) directory
and create a `.swift` file for your new model:

## Create a Configuration

Create a configuration struct to match the `config.json` (any parameters needed).

```swift
public struct YourModelConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    
    // use this pattern for values that need defaults
    public let _layerNormEps: Float?
    public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case _layerNormEps = "layer_norm_eps"
    }
}
```

## Create the Model Class

Create the model class. The top-level public class should have a
structure something like this:

```swift
public class YourModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {

    public let kvHeads: [Int]

    @ModuleInfo var model: YourModelInner

    public func loraLinearLayers() -> LoRALinearLayers {
        // TODO: modify as needed
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ args: YourModelConfiguration) {
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = YourModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // TODO: modify as needed
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }
}
```

## Register the Model

In [LLMModelFactory.swift](LLMModelFactory.swift) register the model type itself
(this is independent of the model id):

```swift
public class LLMTypeRegistry: @unchecked Sendable {
...
    private var creators: [String: @Sendable (URL) throws -> any LanguageModel] = [
        "yourModel": create(YourModelConfiguration.self, YourModel.init),
```

Add a constant for the model in the `LLMRegistry` (not strictly required but useful
for callers to refer to it in code):

```swift
public class LLMRegistry: @unchecked Sendable {
...
    static public let yourModel_4bit = ModelConfiguration(
        id: "mlx-community/YourModel-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?"
    )
```

and finally add it to the all list -- this will let users find the model
configuration by id:

```swift
    private static func all() -> [ModelConfiguration] {
        [
            codeLlama13b4bit,
...
            yourModel_4bit,
```

# Using a Model

See [MLXLMCommon/README.md](../MLXLMCommon/README.md#using-a-model).

# LoRA

[Lora.swift](Lora.swift) contains an implementation of LoRA based on this example:

- https://github.com/ml-explore/mlx-examples/tree/main/lora

See [llm-tool/LoraCommands.swift](../../Tools/llm-tool/LoraCommands.swift) for an example of a driver and
[llm-tool](../../Tools/llm-tool) for examples of how to run it.
