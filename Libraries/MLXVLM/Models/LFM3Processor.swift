// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// Nested processor metadata used by the released BF16 and MLX LFM3-VL checkpoints.
public struct LFM3VLProcessorConfiguration: Codable, Sendable, ModelConfigurationValidating {
    struct Media: Codable, Sendable {
        struct Size: Codable, Sendable {
            let shortestEdge: Int
            let longestEdge: Int
            enum CodingKeys: String, CodingKey {
                case shortestEdge = "shortest_edge"
                case longestEdge = "longest_edge"
            }
        }
        let patchSize: Int
        let temporalPatchSize: Int
        let mergeSize: Int
        let imageMean: [CGFloat]
        let imageStd: [CGFloat]
        let size: Size
        let fps: Double
        let minFrames: Int
        let maxFrames: Int
        let capPixelsPerFrame: Bool
        let maxVideoTokens: Int

        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case mergeSize = "merge_size"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case size, fps
            case minFrames = "min_frames"
            case maxFrames = "max_frames"
            case capPixelsPerFrame = "cap_pixels_per_frame"
            case maxVideoTokens = "max_video_tokens"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
            imageMean =
                try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
            imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
            size = try c.decode(Size.self, forKey: .size)
            fps = try c.decodeIfPresent(Double.self, forKey: .fps) ?? 2
            minFrames = try c.decodeIfPresent(Int.self, forKey: .minFrames) ?? 4
            maxFrames = try c.decodeIfPresent(Int.self, forKey: .maxFrames) ?? 768
            capPixelsPerFrame =
                try c.decodeIfPresent(Bool.self, forKey: .capPixelsPerFrame) ?? false
            maxVideoTokens = try c.decodeIfPresent(Int.self, forKey: .maxVideoTokens) ?? 768
        }

        func validate() throws {
            guard patchSize > 0, patchSize <= 4096, temporalPatchSize > 0, temporalPatchSize <= 16,
                mergeSize > 0, mergeSize <= 64, imageMean.count == 3, imageStd.count == 3,
                imageMean.allSatisfy({ $0.isFinite }),
                imageStd.allSatisfy({ $0.isFinite && $0 > 0 }),
                size.shortestEdge > 0, size.longestEdge >= size.shortestEdge,
                fps.isFinite, fps > 0, minFrames > 0, maxFrames >= minFrames, maxVideoTokens > 0
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "Invalid LFM3 media processor configuration")
            }
        }
    }

    let imageProcessor: Media
    let videoProcessor: Media
    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case videoProcessor = "video_processor"
    }

    public func validateModelConfiguration() throws {
        try imageProcessor.validate()
        try videoProcessor.validate()
    }
}

/// Image and timestamped-video preparation for LFM3-VL.
public struct LFM3VLProcessor: UserInputProcessor {
    private let config: LFM3VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: LFM3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    static func targetSize(
        height: Int, width: Int, frames: Int? = nil, temporalFactor: Int = 2,
        factor: Int, minPixels: Int, maxPixels: Int
    ) throws -> (Int, Int) {
        guard height > 0, width > 0, factor > 0, minPixels > 0, maxPixels >= minPixels,
            Double(max(height, width)) / Double(min(height, width)) <= 200
        else { throw VLMError.processing("Invalid LFM3 resize dimensions or pixel budget") }
        var h = Double(height)
        var w = Double(width)
        let count = max(frames ?? 1, frames == nil ? 1 : temporalFactor)
        if frames != nil && min(h, w) < Double(factor) {
            let scale = max(Double(factor) / h, Double(factor) / w)
            h = floor(h * scale)
            w = floor(w * scale)
        }
        let f = Double(factor)
        var resizedH = (h / f).rounded(.toNearestOrEven) * f
        var resizedW = (w / f).rounded(.toNearestOrEven) * f
        let roundedFrames =
            frames == nil
            ? 1
            : Int((Double(count) / Double(temporalFactor)).rounded(.toNearestOrEven))
                * temporalFactor
        let area = Double(roundedFrames) * resizedH * resizedW
        if area > Double(maxPixels) {
            let beta = sqrt(Double(count) * h * w / Double(maxPixels))
            resizedH = max(f, floor(h / beta / f) * f)
            resizedW = max(f, floor(w / beta / f) * f)
        } else if area < Double(minPixels) {
            let beta = sqrt(Double(minPixels) / (Double(count) * h * w))
            resizedH = ceil(h * beta / f) * f
            resizedW = ceil(w * beta / f) * f
        }
        guard resizedH > 0, resizedW > 0, resizedH < Double(Int.max), resizedW < Double(Int.max)
        else {
            throw VLMError.processing("LFM3 resize produced invalid dimensions")
        }
        return (Int(resizedH), Int(resizedW))
    }

    static func sampleIndices(
        total: Int, sourceFPS: Double, processing: UserInput.VideoProcessing,
        fps: Double = 2, minFrames: Int = 4, maxFrames: Int = 768
    ) throws -> [Int] {
        guard total > 0, sourceFPS.isFinite, sourceFPS > 0 else {
            throw VLMError.processing("Invalid video metadata")
        }
        let requested: Int
        if let count = processing.targetFrames {
            guard count > 0 else { throw VLMError.processing("Video frame count must be positive") }
            requested = count
        } else {
            let targetFPS = processing.targetFramesPerSecond ?? fps
            guard targetFPS.isFinite, targetFPS > 0 else {
                throw VLMError.processing("Video FPS must be positive")
            }
            let count = Double(total) / sourceFPS * targetFPS
            requested = Int(min(count, Double(maxFrames)))
        }
        let count = min(
            max(requested, processing.targetFrames == nil ? minFrames : 1), maxFrames, total)
        guard count > 1 else { return [0] }
        return (0 ..< count).map {
            Int((Double($0) * Double(total - 1) / Double(count - 1)).rounded(.toNearestOrEven))
        }
    }

    private func pixels(_ image: CIImage, size: CGSize, media: LFM3VLProcessorConfiguration.Media)
        -> MLXArray
    {
        let image = MediaProcessing.resampleBicubic(
            MediaProcessing.inSRGBToneCurveSpace(image), to: size)
        return MediaProcessing.asMLXArray(
            MediaProcessing.normalize(
                image,
                mean: (media.imageMean[0], media.imageMean[1], media.imageMean[2]),
                std: (media.imageStd[0], media.imageStd[1], media.imageStd[2])))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let screened = input.removingSpecialTokenLabels(using: tokenizer)
        let messages = LFM3MessageGenerator(multimodal: true).generate(from: screened)
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)
        let imageConfig = config.imageProcessor
        var imagePatches: [MLXArray] = []
        var imageGrids: [THW] = []
        for image in input.images {
            let image = MediaProcessing.apply(try image.asCIImage(), processing: input.processing)
            let (h, w) = try Self.targetSize(
                height: Int(image.extent.height), width: Int(image.extent.width),
                factor: imageConfig.patchSize * imageConfig.mergeSize,
                minPixels: input.processing.minPixels ?? imageConfig.size.shortestEdge,
                maxPixels: input.processing.maxPixels ?? imageConfig.size.longestEdge)
            let (patches, grid) = try QwenVL.patchify(
                images: [pixels(image, size: CGSize(width: w, height: h), media: imageConfig)],
                mergeSize: imageConfig.mergeSize, patchSize: imageConfig.patchSize,
                temporalPatchSize: imageConfig.temporalPatchSize)
            imagePatches.append(patches)
            imageGrids.append(grid)
        }
        // Validate zero-media cases too, so a stray placeholder never becomes an ordinary embedding.
        tokens = try QwenVL.replacePaddingTokens(
            in: tokens, frames: imageGrids,
            paddingToken: "<|image|>", mergeSize: imageConfig.mergeSize, tokenizer: tokenizer)
        let videoConfig = config.videoProcessor
        var videoPatches: [MLXArray] = []
        var videoGrids: [THW] = []
        var videoReplacements: [[Int]] = []
        for video in input.videos {
            let sampled = try await MediaProcessing.indexedVideoFrames(video) { total, sourceFPS in
                try Self.sampleIndices(
                    total: total, sourceFPS: sourceFPS, processing: input.processing.video,
                    fps: videoConfig.fps, minFrames: videoConfig.minFrames,
                    maxFrames: videoConfig.maxFrames)
            }
            let images = sampled.images.map {
                MediaProcessing.apply($0, processing: input.processing)
            }
            let factor = videoConfig.patchSize * videoConfig.mergeSize
            var maxPixels = input.processing.maxPixels ?? videoConfig.size.longestEdge
            let minPixels = input.processing.minPixels ?? videoConfig.size.shortestEdge
            if videoConfig.capPixelsPerFrame {
                let frameCap = Double(videoConfig.maxVideoTokens) * Double(factor) * Double(factor)
                let perFrame = max(
                    min(frameCap, Double(maxPixels / images.count)), floor(Double(minPixels) * 1.05)
                )
                maxPixels = Int(min(Double(Int.max - 1), perFrame * Double(images.count)))
            }
            let first = images[0]
            let (h, w) = try Self.targetSize(
                height: Int(first.extent.height), width: Int(first.extent.width),
                frames: images.count, temporalFactor: videoConfig.temporalPatchSize, factor: factor,
                minPixels: minPixels, maxPixels: maxPixels)
            let (patches, grid) = try QwenVL.patchify(
                images: images.map {
                    pixels($0, size: CGSize(width: w, height: h), media: videoConfig)
                },
                mergeSize: videoConfig.mergeSize, patchSize: videoConfig.patchSize,
                temporalPatchSize: videoConfig.temporalPatchSize)
            videoPatches.append(patches)
            videoGrids.append(grid)
            let perFrame = grid.h * grid.w / (videoConfig.mergeSize * videoConfig.mergeSize)
            var replacement = ""
            for t in 0 ..< grid.t {
                let start = t * videoConfig.temporalPatchSize
                let end = min(start + videoConfig.temporalPatchSize - 1, sampled.seconds.count - 1)
                let seconds = (sampled.seconds[start] + sampled.seconds[end]) / 2
                let time = String(
                    format: "%.1f", locale: Locale(identifier: "en_US_POSIX"), seconds)
                replacement +=
                    "<\(time) seconds><|vision_start|>"
                    + String(repeating: "<|video|>", count: perFrame) + "<|vision_end|>"
            }
            videoReplacements.append(tokenizer.encode(text: replacement))
        }
        let marker = tokenizer.encode(text: "<|vision_start|><|video|><|vision_end|>")
        let ranges = tokens.ranges(of: marker)
        guard ranges.count == videoReplacements.count else {
            throw VLMError.processing("LFM3 video placeholders and media differ")
        }
        var expanded: [Int] = []
        var offset = 0
        for (range, replacement) in zip(ranges, videoReplacements) {
            expanded.append(contentsOf: tokens[offset ..< range.lowerBound])
            expanded.append(contentsOf: replacement)
            offset = range.upperBound
        }
        expanded.append(contentsOf: tokens[offset...])
        let prompt = MLXArray(expanded).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: prompt),
            image: imagePatches.isEmpty
                ? nil : .init(pixels: concatenated(imagePatches), frames: imageGrids),
            video: videoPatches.isEmpty
                ? nil : .init(pixels: concatenated(videoPatches), frames: videoGrids))
    }
}
