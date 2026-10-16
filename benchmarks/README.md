# Preserved merge investigation

All measurements below were collected on 2026-10-15 on the local M5 Pro
(16 GPU cores, 24 GB unified memory). No experiment data was overwritten when
moving to the set-difference investigation. The production merge kernel remains
the original tiled implementation; neither direct stores nor cached run heads
showed a consistent improvement.

| File | Meaning |
|---|---|
| `m5-pro-2026-10-15.csv` | Initial MLX/Rust completed-call timings; includes submission/allocation/waits, not just GPU time. |
| `m5-pro-gpu-diagnosis-2026-10-15.txt` | Unchanged shader dispatched directly through Metal; GPU command-buffer timestamps and pipeline resource limits. |
| `limiter-investigation-2026-10-15.json` | Early mixed merge/copy counter recording. Do not attribute its aggregate stall signals to merge alone. |
| `isolated-limiters-2026-10-15.json` | Separate merge-only/copy-only recordings; middle-60%-span filtering, counter statistics and caveats. |
| `cached-heads-2026-10-15.txt` | Paired baseline/cached-head GPU timings. Candidate rejected; original restored. |
| `merge-profile-raw-2026-10-15.tar.gz` | Preserved source traces, capture, verified counter template, diagnostic harnesses and AIR. Kept outside `build/` so cleaning build output does not delete this evidence. |
| `SHA256SUMS` | Integrity checksums for those six artifacts. Run `shasum -a 256 -c benchmarks/SHA256SUMS` from the repository root. |

The raw archive includes the mixed counter trace, the isolated merge/copy traces,
the merge/copy GPU capture, `PerformanceLimiters.tracetemplate`, the paired timing
Swift harness, the isolated C++ workload, and baseline/cached optimized AIR. Its
paths retain their original `build/profile/` or `build/profiling/` prefix. Extract
into a separate directory when inspecting; do not overwrite active builds.

The `.trace` recordings reported a logging-stream error. Their GPU counter tables
were nevertheless exported successfully. Counter configuration was established
through the supported Instruments Performance Limiters menu, not guessed from
undocumented profile IDs. Counter influence metrics are not fractions of time
stalled. Samples remain device-wide even when selected by target GPU intervals.

The archive includes local process/device metadata; review it before public
sharing. It is retained locally and as an asset of the private GitHub release
`merge-checkpoint-2026-10-15` in `StarGazerM/mccl`, not in Git history.
The original artifacts under `build/` also remain intact. The archive passed
`gzip -t`; GitHub's uploaded SHA-256 digest matches the local checksum.

Retrieve it into a fresh checkout with:

```sh
gh release download merge-checkpoint-2026-10-15 --repo StarGazerM/mccl \
  --pattern merge-profile-raw-2026-10-15.tar.gz --dir benchmarks
shasum -a 256 -c benchmarks/SHA256SUMS
```
