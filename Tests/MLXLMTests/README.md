## Media resources

The two movie resources test the MediaProcessing pipeline for correctness and validation.

The video file was created via FFMPEG via

```
ffmpeg \
-f lavfi \
-i smptebars=size=1920x1080:rate=30:duration=5.0 \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-vf "drawtext=text='Frame\\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:alpha=0.5:box=1:boxborderw=4,drawtext=text='TC':x=(w-tw)/2:y=(lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:fontcolor=white:timecode='01\\:00\\:00\\:00':timecode_rate=(30)" \
-c:v libx264 \
-c:a aac \
-crf 23 \
-preset medium \
-pix_fmt yuv420p \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y 1080p_30.mov
```

and the audio only file 

```
ffmpeg \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-c:a aac \
-crf 23 \
-preset medium \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y audio_only.mov
```

## Bonsai 2 reference fixture

`Resources/prism_hadamard_reference.json` records logits from Prism's Python model at
`38f27dc24b535928246b64b211b66d38b7a3e17f`, using MLX 0.32.3 on Metal. It contains a tiny
synthetic two-layer model, repeating tensor patterns, and expected outputs. No published
model weights or tokenizer are needed. Swift tests consume the stored patterns and logits;
they do not derive expected logits from another Swift model.

Coverage includes schema-1 and schema-2 loading, both Swift language paths, tied and untied
heads, block-0 and block-1024 projections, FP16 weights with FP32 signs/recurrent state,
uncached prefill, cached decode, and multi-token continuation. Separate unit tests cover
8192-element blocks, small-magnitude L2 normalization, gated-norm precision, and invalid
metadata. The fixture is not a full 27B quality or performance benchmark.

To regenerate, install `mlx==0.32.3` and the pinned Prism checkout in a temporary Python
virtual environment, then run from the repository root:

```sh
python scripts/generate-prism-hadamard-reference.py \
  --mlx-lm /path/to/pinned/prism-mlx-lm \
  --output Tests/MLXLMTests/Resources/prism_hadamard_reference.json
```

The generator checks the reference revision and MLX version. Normal unit tests require
neither Python nor network access.
