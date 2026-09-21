# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

> [!IMPORTANT]
> The `main` branch is a _new_ major version number: 3.x.  In order
> to decouple from tokenizer and downloader packages some breaking
> changes were introduced. See [upgrading documentation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/upgrade) for detailed instructions on upgrading.

> [!IMPORTANT]
> We use `swift-format` to keep the code formatting consistent.  CI has this pinned to `603.0.0` right now. 

Some key features include:

- Model loading with integrations for a variety of tokenizer and model downloading packages.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM, check out [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [Techniques for developing in mlx-swift-lm](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/developing)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations
- [MLXGuidedGeneration](Libraries/MLXGuidedGeneration/README.md): Grammar-constrained generation (JSON Schema or EBNF) for any MLX model.
- [MLXFoundationModels](Libraries/MLXFoundationModels/README.md): Bridge MLX models into Apple's `FoundationModels.LanguageModel` for use with `LanguageModelSession`. (Requires the macOS/iOS/visionOS 27.0 SDK.)

## Usage

This package integrates with a variety of tokenizer and downloader packages through protocol conformance. Users can pick from three ways to integrate with these packages, which offer different tradeoffs between freedom and convenience.

See documentation on [how to integrate mlx-swift-lm and downloaders/tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).

## Installation

Add the core package to your `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
```

Then chose an [integration package for downloaders and tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).


## Quick Start

See also [MLXLMCommon](Libraries/MLXLMCommon). The simplest way to get started is using the `MLXHuggingFace` macros, which provide a default Hugging Face downloader and tokenizer integration.

## Package.swift

```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
    .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
],
targets: [
    .target(
        name: "YourTargetName",
        dependencies: [
            .product(name: "MLXLLM", package: "mlx-swift-lm"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
            .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
]
```

## Usage

```swift
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

let model = try await #huggingFaceLoadModelContainer(
    configuration: LLMRegistry.gemma3_1B_qat_4bit
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

For alternative integration approaches (custom downloaders, alternative tokenizer packages, local-only weights), see the [using documentation](Libraries/MLXLMCommon/Documentation.docc/using.md).

## Bonsai 2 (Prism Hadamard checkpoints)

This branch registers `prism_hadamard_qwen35` in both model factories for
[`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit).
Use `VLMModelFactory` for text and images, or `LLMModelFactory` for text-only inference.
The text-only loader does not create a vision tower and skips its tensor data before loading
weights into memory; checkpoint headers are still read and downloads are unchanged.
`VLMModelFactory` still loads both towers, even for text-only prompts.
No Python runtime is imported or executed.

The Swift loader implements the model pack's
[`runtime.py`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/blob/main/runtime/runtime.py)
and [`vision_artifact.py`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/blob/main/runtime/vision_artifact.py)
contracts: 2-bit affine weights with group size 128, FP32 normalized blockwise Hadamard
transforms before language projections, and inverse transforms after embedding lookup.
The vision tower remains unrotated. Ordinary quantized-projection fusion is not applied to
these transformed layers.

Keep the original `config.json`, `model.safetensors`, tokenizer/chat-template files, and
`preprocessor_config.json` for vision. The layer manifest comes from `config.json`; sign
vectors are loaded from the safetensors parameters, not `hadamard.json`. Do not rename the
model type to `qwen3_5`: that would omit required transforms. Schema-1 text packs and
schema-2 `mlx-vlm-qwen3_5` packs with grouped GDN layout are supported; invalid metadata,
packed shapes, or sign vectors fail loading rather than falling back to ordinary Qwen.

Regression tests cover transform numerics, synthetic checkpoint loading, text/image
inference and continuation, and the published 27B tensor topology. These tests do not
measure full-checkpoint generation quality, peak memory, or device performance. The
published pack is about 8.6 GB, with additional runtime memory needed for activations and caches.

## `FoundationModels` integration

`MLXFoundationModels` is a bridge between MLX models and Apple's `FoundationModels` framework: build an `MLXLanguageModel`, pass it to `LanguageModelSession`, and generate through the standard `FoundationModels` API. Requires the macOS/iOS/visionOS 27.0 SDK.

```swift
import Foundation
import FoundationModels
import HuggingFace
import MLXFoundationModels
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable
struct Recommendation {
    let attraction: String
    let neighborhood: String
    let tip: String
}

if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
    let model = #huggingFaceLanguageModel(
        configuration: LLMRegistry.gemma3_1B_qat_4bit,
        capabilities: [.guidedGeneration])
    let session = LanguageModelSession(model: model)

    let recommendation = try await session.respond(
        to: "Recommend one thing to do in Chicago.",
        generating: Recommendation.self)
    print(recommendation.content)
    // Recommendation(
    //     attraction: "Art Institute of Chicago",
    //     neighborhood: "Loop",
    //     tip: "Admire Seurat's Sunday on La Grande Jatte up close.")
}
```

Here, we combine `MLXFoundationModels` with [`MLXGuidedGeneration`](Libraries/MLXGuidedGeneration/README.md) by requesting a `@Generable` type: the response is grammar-constrained to that type's schema. `MLXGuidedGeneration` is a standalone primitive that constrains any MLX model's output to a schema.

Other capabilities include `.vision`, `.toolCalling`, and `.reasoning`. See [Libraries/MLXFoundationModels](Libraries/MLXFoundationModels/README.md) for the full capability set, custom weights and loaders, and more information about using MLXFoundationModels.
