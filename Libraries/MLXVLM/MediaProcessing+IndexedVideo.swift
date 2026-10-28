// Copyright © 2026 Apple Inc.

import AVFoundation
import CoreImage
import MLXLMCommon

extension MediaProcessing {
    struct IndexedVideoFrames {
        let images: [CIImage]
        let seconds: [Double]
    }

    /// Samples source-frame indices rather than duration endpoints. Compressed samples are
    /// scanned for presentation times before decoding only the selected images.
    static func indexedVideoFrames(
        _ video: UserInput.Video,
        indices: (Int, Double) throws -> [Int]
    ) async throws -> IndexedVideoFrames {
        switch video.source {
        case .frames(let frames):
            guard !frames.isEmpty else { throw VLMError.processing("Video contains no frames") }
            let times = frames.map { $0.timeStamp.seconds }
            guard times.allSatisfy({ $0.isFinite }),
                zip(times, times.dropFirst()).allSatisfy({ $0 < $1 })
            else {
                throw VLMError.processing("Video timestamps must be finite and increasing")
            }
            let fps = frames.count > 1 ? Double(frames.count - 1) / (times.last! - times[0]) : 24
            let selected = try indices(frames.count, fps)
            return IndexedVideoFrames(
                images: try selected.map { try frames[$0].image.asCIImage() },
                seconds: selected.map { times[$0] })
        case .url(let url):
            return try await indexedAssetFrames(AVURLAsset(url: url), indices: indices)
        case .avAsset(let asset):
            return try await indexedAssetFrames(asset, indices: indices)
        }
    }

    private static func indexedAssetFrames(
        _ asset: AVAsset,
        indices: (Int, Double) throws -> [Int]
    ) async throws -> IndexedVideoFrames {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VLMError.noVideoTrackFound
        }
        let nominalFPS = try await track.load(.nominalFrameRate)
        let duration = try await asset.load(.duration)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw VLMError.videoNotDecodable }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? VLMError.videoNotDecodable }
        var times: [CMTime] = []
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            // Output times include edit-list offsets; zero-sample buffers are timeline markers.
            let time = CMSampleBufferGetOutputPresentationTimeStamp(sample)
            if CMSampleBufferGetNumSamples(sample) > 0, time.isNumeric { times.append(time) }
        }
        guard reader.status == .completed else { throw reader.error ?? VLMError.videoNotDecodable }
        times.sort { CMTimeCompare($0, $1) < 0 }
        guard !times.isEmpty else { throw VLMError.processing("Video contains no frames") }
        let sourceFPS = nominalFPS > 0 ? Double(nominalFPS) : Double(times.count) / duration.seconds
        guard sourceFPS.isFinite, sourceFPS > 0 else {
            throw VLMError.processing("Invalid source video FPS")
        }
        // Match the released processor's normalization of container FPS metadata.
        let fps = (sourceFPS * 1e6).rounded(.toNearestOrEven) / 1e6
        let selected = try indices(times.count, fps)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var images: [CIImage] = []
        for index in selected {
            try Task.checkCancellation()
            let result = try await generator.image(at: times[index])
            images.append(
                CIImage(
                    cgImage: result.image,
                    options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]))
        }
        return IndexedVideoFrames(images: images, seconds: selected.map { Double($0) / fps })
    }
}
