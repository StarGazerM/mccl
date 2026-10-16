# MCCL: MLX-backed GPU operations

Stable ascending merge, multiset difference, and a blocked Bloom filter for `u32`
keys. Rust calls a C ABI backed by MLX C++ operations and offline-compiled Metal
kernels. There is no Python runtime dependency.

## Build and run

Requires Apple silicon, Xcode with the Metal Toolchain component, CMake >= 3.27,
and Rust. The first CMake configure downloads dependencies.

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 8
cargo test --release
cargo run --release -- check
cargo run --release -- bench --sizes 1K,64K,1M,4M,16M,32M --reps 50 --warmup 10
```

`MCCL_NATIVE_DIR` overrides the native library directory when building Rust.
Libraries and metallibs must remain together. The Rust executable has a build-time
rpath; the extension finds its metallib relative to its own dylib, not the cwd.

Dependencies are pinned in CMake: mlx-c `a341b4925024b88b2c593468f16e12f5e17315da`
and MLX v0.32.2, commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`.
Their upstream license notices remain in the fetched source trees.

## Optimized profiling build

Build profiling artifacts separately so the executable and the native libraries
all contain debugging information while retaining release optimization:

```sh
cmake -S . -B build/profile -DCMAKE_BUILD_TYPE=Release -DMCCL_PROFILE=ON
cmake --build build/profile --parallel 8
MCCL_NATIVE_DIR=build/profile cargo build --profile profiling
xcrun dsymutil target/profiling/mccl
xcrun dsymutil build/profile/libmccl_native.dylib
./target/profiling/mccl check
```

`MCCL_PROFILE` enables C/C++ `-g` and Metal
`-O3 -gline-tables-only -frecord-sources`. Cargo's `profiling` profile inherits
release optimization, retains full debug info, and disables stripping. Do not
accidentally profile `target/release/mccl` linked to the normal native build.

The profiling executable passed all 456 GPU correctness cases. Loader diagnostics
confirmed it loads `build/profile` libraries. Recording it with Metal System Trace
still reported the log-archive error; symbols do not by themselves enable GPU
counters. The resulting trace metadata still reported `Counter Set: (null)` and
`Shader Timeline: Disabled`. Source attribution, recording health, and counter
configuration must be checked separately.

## Sorted-array API and semantics

```rust
use mccl::Gpu;

let gpu = Gpu::new()?;
let a = gpu.upload(&[1, 3, 5])?;
let b = gpu.upload(&[2, 4, 6])?;
let merged = gpu.merge(&a, &b)?;
assert_eq!(merged.to_vec()?, [1, 2, 3, 4, 5, 6]);
# Ok::<(), mccl::Error>(())
```

The C entry points in `include/mccl.h` are `mccl_merge_sorted_u32`,
`mccl_difference_sorted_u32`, `mccl_copy_u32`, and `mccl_last_error`. They accept official mlx-c handles and
build lazy graphs. Evaluate the output and synchronize its stream before host
access. Native graph-construction errors return nonzero and preserve the previous
output; evaluation errors follow mlx-c's error-handler mechanism.

- Inputs must be sorted ascending; this is not checked by scanning the host.
- Merge is A-before-B for equal keys. Key-only results cannot expose equal-key
  provenance; stability follows the asymmetric co-rank and left-first merge rule.
- Empty inputs, unequal lengths, duplicate keys, and `UINT_MAX` are supported.
- Merge's combined length must fit `INT_MAX`; difference limits each input to `INT_MAX`. Only GPU streams and 1D u32 arrays are accepted.
- C inputs may be strided; MLX makes them contiguous, potentially adding a copy.
- Rust uploads produce initialized contiguous GPU buffers. Uploads wait; merge and
  copy remain lazy. Arrays retain input graphs and their thread-affine context.
- Unsorted inputs have unspecified results but bounded kernel accesses. They are
  not an alternative sorting operation.
- `Gpu`/`U32Array` are neither Send nor Sync. Rust installs a process-wide mlx-c
  error handler; callers sharing mlx-c must account for that handler ownership.
- CPU execution, other dtypes, payload merging, autodiff, and vectorized transforms
  are not implemented by this operation.

## Sorted multiset difference

`gpu.difference(&a, &b)?` subtracts multiplicities: each occurrence in B cancels
the earliest equal occurrence in A. For `[1,1,1,2,4] - [1,1,3]`, the result is
`[1,2,4]`. This is not a membership/anti-join filter.

The returned `Difference` owns private storage of capacity `a.len()` plus a
device-resident uint64 count. `capacity()` is metadata-only; `eval()` completes
the storage/count graphs; `count()` and `to_vec()` explicitly synchronize and
read back. `to_vec()` reads only the initialized survivor prefix, never the tail.
No host count read is needed to construct or execute the GPU graph.

Implementation: three MLX lower/upper-bound searches determine occurrence rank
and B multiplicity, followed by a binary keep mask, exclusive scan, count reduction,
and stable Metal scatter. Empty inputs are handled explicitly. This implementation
is a correct multi-stage baseline, not a claimed optimal tiled set algorithm.

```sh
cargo run --release -- difference-bench --sizes 64K,1M --reps 20 --warmup 5
```

Sizes denote combined A+B length. CSV reports each input length, survivor count,
and completed-call latency; no single-pass bandwidth formula is applied.
The initial release run passed 100 difference cases and all 456 merge cases with
Metal Shader Validation enabled. At 524,288 elements per input, the initial
20-sample median was approximately 0.74–0.93 ms across disjoint, interleaved, and
equal-multiplicity duplicate inputs. These are wall timings, not GPU-counter
measurements; no memory-boundness conclusion is established for difference.


## Blocked Bloom filter

```rust
let mut filter = gpu.bloom(65_536, 0)?; // block count, xxHash64 seed
let keys = gpu.upload(&[1, 2, 3, u32::MAX])?;
filter.add(&keys)?;
assert_eq!(filter.contains(&keys)?.to_vec()?, [1, 1, 1, 1]);
filter.clear()?;
assert_eq!(filter.contains(&keys)?.to_vec()?, [0, 0, 0, 0]);
```

This implements RCCL's eight-word blocked policy for **four-byte little-endian
u32 keys**: seeded xxHash64, multiply-shift block selection, and one salted bit
in each of eight uint32 words. Storage is `num_blocks * 32` bytes. All u32 keys
are valid, including zero and `UINT_MAX`; inputs need not be sorted.

`BloomFilter` provides `add`, `contains`, `clear`, `words`, `num_blocks`, and
`seed`. Construction, mutation, and queries complete before returning. Insert
uses device atomic OR in place, with no full-filter copy. Queries return independent
completed u32 flag arrays, unaffected by subsequent mutation or filter destruction.
`words()` is an explicit host snapshot for inspection. The filter retains its GPU
context and is neither Clone, Send, nor Sync. Cross-context keys are rejected.

Zero means absent; one means **possibly present**, never exact membership.
Completed insertion has no false negatives until clear, but false positives are
expected. There is no per-key erase, u64-key API, automatic FPR sizing, concurrent
insert/query contract, or hash map in this implementation. Block count must be
positive with `num_blocks * 8 <= INT_MAX`. GPU mutation errors are not transactional.

The C ABI exposes `mccl_bloom_create/free/clear/add/contains/read_words`; see
`include/mccl.h` for ownership and error requirements. Direct encoding uses
MLX's stream encoder and its commit/wait/error-check synchronization path.

```sh
cargo run --release -- bloom-check
cargo run --release -- bloom-bench --keys 1M --queries 2M \
  --blocks 65536 --seed 0 --reps 30 --warmup 5
```

Verification: 84 Bloom cases passed with Metal Shader Validation, including
exact CPU/GPU words, contention, duplicate inserts, incremental additions,
clear, empty/extreme keys, seeds, snapshot isolation and lifetime/context checks.
Six additional GPU bit-layout fixtures matched independent Python `xxhash==3.6.0`
results, including strided C inputs. Existing merge/difference regressions passed.

Initial data is in `benchmarks/bloom-filter-2026-10-15.csv`. Inserting 1,048,576
distinct keys into 65,536 blocks (2 MiB), then querying 1,048,576 members and
1,048,576 known nonmembers, produced 1,382 false positives (0.1318% of nonmembers)
and no false negatives. GPU bits and query flags matched the CPU oracle.
The observed rate is empirical, not a bound on arbitrary data.
Clear/insert/query medians were 0.604/0.732/0.762 ms. These include CPU submission
and completion; timing variability was substantial. No GPU bottleneck claim is
made from these wall timings.

## Apple execution design

`native/merge.metal` uses 256 threads and 1,024 output keys per tile. Two threads
compute global merge-path boundaries. The group stages contiguous A/B ranges,
then each thread computes a local co-rank and merges up to four keys. A second
threadgroup array transposes the outputs into coalesced stores. There are three
threadgroup-memory barriers and no cross-threadgroup spinning or global atomics.
Declared threadgroup storage is 8,200 bytes, below the measured M5 Pro limit of
32,768 bytes. This is a starting policy, not a claim of optimal Apple tuning.

`native/mccl.cpp` uses MLX allocation, active command encoding, and resource
tracking. Output storage never aliases the inputs. The copy baseline dispatches
an actual read/write kernel, rather than an identity operation MLX can eliminate.

## Verification

On 2026-10-15, the release build and the GPU integration test passed. The actual
CLI passed 456 merge cases and their forced copies, graph-lifetime checks, and
cross-context rejection with Metal Shader Validation enabled, launched from `/tmp`.
A separate direct C ABI smoke covered strided input, wrong dtype/shape/CPU stream,
output preservation on error, and an unsorted-input bounds exercise. That smoke
is not part of the permanent Rust test suite.

```sh
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 ./target/release/mccl check
```

Do not enable validation during performance measurement.

## Initial bandwidth investigation

Raw results: `benchmarks/m5-pro-2026-10-15.csv`.
Machine: M5 Pro, 16 GPU cores, 24 GB unified memory; macOS 27.0.1, Xcode 27.0.
The desktop was active on a 7680x2160 display; this was not a headless/isolated
hardware-counter experiment. Dependency build and validation finished before timing.

Sizes are total output keys; K/M suffixes mean powers of 1024. Inputs are equal
length in this benchmark. Each case has 10 warmups and 50 measured repetitions,
alternating merge/copy order. Every repetition constructs a fresh output graph
and waits for evaluation and stream completion. Upload, readback, and output
handle destruction are outside timing. Graph construction, scheduling, output
allocation from MLX's warmed allocator pool, and waits are included.

The CSV records median/p10/p90 CPU-wall completed-call latency. Effective bandwidth
is `8*N / elapsed`: N u32 keys read and N written. Merge also reads global partition
probes and uses threadgroup memory; these are not included in that logical-byte
count. These are NOT GPU timestamp durations or measured DRAM bandwidth.

At 33,554,432 keys:

| Distribution | Merge median (ms) | Copy median (ms) | Merge logical GB/s | Copy logical GB/s |
|---|---:|---:|---:|---:|
| Disjoint | 2.159 | 1.214 | 124.351 | 221.158 |
| Reverse disjoint | 1.835 | 1.170 | 146.255 | 229.510 |
| Interleaved | 1.973 | 1.189 | 136.024 | 225.774 |
| Duplicate-heavy | 1.955 | 1.167 | 137.310 | 230.001 |

Interpretation: large merges reach about 56–64% of the matched copy throughput.
The data do not establish DRAM-bandwidth saturation. They leave substantial
kernel-specific overhead (partition searches, dependent accesses, threadgroup
traffic/barriers, occupancy/resource limits) to distinguish through GPU profiling.
The ~0.15 ms small-input floor is dominated by completed-call overhead, not bulk
memory movement. Copy throughput is a measured comparison point, not a theoretical
hardware maximum. Earlier Instruments recording attempts had logging errors;
no clean hardware-counter capture is claimed here.
