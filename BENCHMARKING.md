# Model performance benchmark

`mlx-lm-benchmark` is a non-interactive, release-mode benchmark for autonomous optimization loops. It loads an LLM or VLM from Hugging Face Hub or a local directory and records prefill, decode, and MLX memory metrics as versioned JSON.

## Run it

A quick run suitable for validating a change:

```bash
scripts/benchmark.sh \
  --model mlx-community/Qwen3-0.6B-4bit \
  --prompt-tokens 128 \
  --output-tokens 32 \
  --warmup-runs 1 \
  --runs 3 \
  --label baseline \
  --output /tmp/mlx-baseline.json
```

The default workload uses exact prompt lengths of 128, 512, and 2048 tokens, one warmup per length, five measured runs, and at most 256 output tokens:

```bash
scripts/benchmark.sh \
  --model mlx-community/gemma-4-E2B-it-qat-4bit \
  --output /tmp/gemma4.json
```

A local model directory works with the same command:

```bash
scripts/benchmark.sh \
  --model /path/to/model \
  --model-kind auto \
  --output /tmp/local-model.json
```

The wrapper always builds and runs the current checkout with Xcode's `Release` configuration. Xcode packaging is required so the command-line executable can locate MLX's default Metal library. Set `MLX_LM_BENCHMARK_DERIVED_DATA` to override its persistent build directory. Run `scripts/benchmark.sh --help` for every option.

## Agent contract

- The command is non-interactive and returns a nonzero status for invalid arguments, loading failures, and benchmark failures.
- Progress and concise per-run metrics go to stderr.
- The report goes to `--output`. If `--output` is omitted or is `-`, JSON goes to stdout.
- Prefer `--output` in automation because model implementations may write diagnostics to stdout.
- Existing files at `--output` are atomically replaced. Missing parent directories are created.
- `schema_version` is currently `1`. JSON keys are snake case and deterministically sorted.
- The report captures the Git commit, branch, dirty files, release/debug configuration, Swift version, hardware, OS, power/thermal state, relevant performance environment variables, model weights, workload settings, individual runs, and aggregate distributions.
- Use `--label` for a variant name and repeat `--tag key=value` for searchable experiment metadata.

A typical autonomous comparison loop is:

```bash
scripts/benchmark.sh --model "$MODEL" --label baseline \
  --prompt-tokens 128,512,2048 --runs 5 --output /tmp/baseline.json

# Edit the implementation.

scripts/benchmark.sh --model "$MODEL" --label candidate \
  --prompt-tokens 128,512,2048 --runs 5 --output /tmp/candidate.json

jq '.cases[] | {name, prefill: .summary.prefill_tokens_per_second.median,
    decode: .summary.decode_tokens_per_second.median,
    peak: .summary.peak_memory_bytes.median}' /tmp/baseline.json /tmp/candidate.json
```

Keep model, prompt lengths, output token count, cache settings, and environment identical when comparing variants. Check `system.low_power_mode_enabled`, `system.thermal_state`, `build.git_dirty`, and each run's `output_tokens` and `stop_reason` before accepting a result.

## Methodology

- Generation uses raw token events and greedy sampling (`temperature = 0`) so tokenizer decoding and random sampling do not obscure model/kernel performance.
- `--prompt-tokens 128,512,2048` creates exact token counts by tokenizing and repeating the prompt seed. Use `--prompt-tokens natural` to run the normal model input processor and chat template without resizing the prompt.
- Each prompt length receives its own excluded warmups. This primes shape-specific compiled graphs before measured runs.
- Model download, loading, native precompilation, and weight allocation are reported separately in `load` and excluded from generation measurements.
- Prompt construction/tokenization happens before the per-run MLX peak-memory reset and before generation timing. Prefill/decode timings come from `GenerateCompletionInfo` in the production generation loop, which synchronizes the MLX stream before completion.
- Models may emit EOS before `--output-tokens`. The report preserves actual output token counts and the `stop_reason`; compare decode throughput only when workloads are equivalent.
- Memory values are MLX allocator metrics. `active_bytes` is live MLX allocation, `cache_bytes` is reusable MLX buffer cache, and `peak_bytes` is peak active MLX memory after the per-run reset. They are not process resident-set size.
- By default, the MLX buffer cache is retained between runs to measure steady-state inference. Use `--clear-cache-between-runs` for cache-cold trials or `--cache-limit-mb` to test an application-specific cache budget.

## Performance controls

The command exposes generation controls commonly needed while optimizing kernels and caches:

```text
--prefill-step-size <tokens>
--max-kv-size <tokens>
--kv-bits <4|8>
--kv-group-size <tokens>
--quantized-kv-start <token-index>
--kv-scheme <affine4|affine8|turbo...>
--cache-limit-mb <MiB>
--clear-cache-between-runs
```

`--model-kind auto` inspects the resolved configuration and selects the LLM or VLM factory. Pass `--model-kind llm` or `--model-kind vlm` to force a path while debugging factory-specific implementations.
