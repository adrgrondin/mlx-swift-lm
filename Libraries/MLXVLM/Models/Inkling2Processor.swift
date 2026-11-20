// Copyright © 2026 Apple Inc.
// Image preprocessing adapted from ml-explore/mlx-lm#2086 (Apache-2.0).

import CoreImage
import Foundation
import MLX
import MLXLMCommon

struct Inkling2ProcessorLoadingResolver: VLMProcessorLoadingResolver {
    func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        guard context.modelType == "inkling2_loop" else { return nil }
        return .init(data: context.configurationData, processorType: "Inkling2Processor")
    }
}

struct Inkling2Processor: UserInputProcessor {
    let configuration: Inkling2Configuration
    let tokenizer: any Tokenizer

    private struct Messages: MessageGenerator {
        func generate(message: Chat.Message) -> Message {
            var result = DefaultMessageGenerator().generate(message: message)
            if !message.images.isEmpty {
                result["content"] = contentParts(for: message, layout: .imagesThenText)
            }
            return result
        }
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty, input.audios.isEmpty else {
            throw VLMError.processing("Inkling 2 Mini accepts text and still images only")
        }
        let messages = Messages().generate(from: input.removingSpecialTokenLabels(using: tokenizer))
        let ids =
            try _Inkling2ToolPrompt.renderIfNeeded(
                tokenizer: tokenizer, messages: messages, tools: input.tools,
                additionalContext: input.additionalContext)
            ?? tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        let placeholder = configuration.imageToken
        guard ids.filter({ $0 == placeholder }).count == input.images.count else {
            throw VLMError.processing("Inkling 2 needs one image placeholder per input image")
        }
        guard !input.images.isEmpty else { return LMInput(tokens: MLXArray(ids)) }

        var frames = [THW]()
        var pixels = [MLXArray]()
        var counts = [Int]()
        let block = configuration.vision.blockSize
        for image in input.images {
            let ci = MediaProcessing.apply(try image.asCIImage(), processing: input.processing)
            let bounded = try Self.fitImage(ci, block: block, maxPixels: input.processing.maxPixels)
            let (values, frame) = try Self.preprocess(bounded, block: block)
            pixels.append(values.flattened())
            frames.append(frame)
            counts.append((frame.h / block) * (frame.w / block))
        }
        var expanded = [Int]()
        var countsIterator = counts.makeIterator()
        for id in ids {
            if id == placeholder {
                expanded.append(contentsOf: repeatElement(id, count: countsIterator.next()!))
            } else {
                expanded.append(id)
            }
        }
        return LMInput(
            text: .init(tokens: MLXArray(expanded)),
            image: .init(pixels: concatenated(pixels), frames: frames))
    }

    static func fitImage(_ image: CIImage, block: Int, maxPixels: Int?) throws -> CIImage {
        let bounds = image.extent.integral
        guard !bounds.isEmpty, !bounds.isInfinite, bounds.width.isFinite, bounds.height.isFinite
        else { throw VLMError.processing("Invalid Inkling 2 image dimensions") }
        let budget = maxPixels.map { $0 / block / block } ?? 4096
        guard budget >= 1 else {
            throw VLMError.processing("Inkling 2 image budget must fit one \(block)×\(block) block")
        }
        let block = CGFloat(block)
        let width = bounds.width / block
        let height = bounds.height / block
        guard ceil(width) * ceil(height) > CGFloat(budget) else { return image }

        // Budget complete merge blocks, including padding on the short edge.
        let ratio = min(1, sqrt(CGFloat(budget) / (width * height)))
        let cols = min(CGFloat(budget), max(1, floor(width * ratio)))
        let rows = min(floor(CGFloat(budget) / cols), max(1, floor(height * ratio)))
        let scale = min(cols / width, rows / height)
        let resized = image.transformed(
            by: CGAffineTransform(
                a: scale, b: 0, c: 0, d: scale, tx: -bounds.minX * scale, ty: -bounds.minY * scale))
        return resized.cropped(
            to: CGRect(
                x: 0, y: 0, width: min(cols * block, resized.extent.maxX),
                height: min(rows * block, resized.extent.maxY)))
    }

    static func preprocess(_ image: CIImage, block: Int) throws -> (MLXArray, THW) {
        let bounds = image.extent.integral
        guard !bounds.isEmpty, !bounds.isInfinite, bounds.width.isFinite, bounds.height.isFinite
        else {
            throw VLMError.processing("Invalid Inkling 2 image dimensions")
        }
        let (width, height) = (Int(bounds.width), Int(bounds.height))
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(options: [
            .workingColorSpace: colorSpace, .cacheIntermediates: false,
        ])
        let rowBytes = width * 4 * MemoryLayout<Float>.stride
        var data = Data(count: height * rowBytes)
        data.withUnsafeMutableBytes { bytes in
            context.render(
                image, toBitmap: bytes.baseAddress!, rowBytes: rowBytes, bounds: bounds,
                format: .RGBAf, colorSpace: colorSpace)
        }
        let rgba = MLXArray(data, [height, width, 4], type: Float.self)
        let alpha = rgba[.ellipsis, 3...]
        let rows = MLXArray((0 ..< height).map { $0 / 8 }).expandedDimensions(axis: 1)
        let cols = MLXArray((0 ..< width).map { $0 / 8 })
        let checker = which((rows + cols) % 2 .== 0, Float(204), Float(153)).expandedDimensions(
            axis: -1)
        let rgb = round(clip(rgba[.ellipsis, ..<3] * 255 + checker * (1 - alpha), min: 0, max: 255))
        let paddedHeight = (height + block - 1) / block * block
        let paddedWidth = (width + block - 1) / block * block
        let padded = padded(
            rgb, widths: [[0, paddedHeight - height], [0, paddedWidth - width], [0, 0]],
            value: MLXArray(-1.0))
        return ((padded / 255 - 0.5) / 0.5, THW(1, paddedHeight, paddedWidth))
    }
}
