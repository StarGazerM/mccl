#include <metal_stdlib>
using namespace metal;

constant uint kThreads = 256;
constant uint kPerThread = 4;
constant uint kTile = kThreads * kPerThread;

// A-before-B co-rank: B[j-1] >= A[i] moves the boundary toward more A.
// Bounds alone guarantee safe probes, even when sortedness is violated.
inline uint global_corank(
    device const uint* a, uint na,
    device const uint* b, uint nb, uint diagonal) {
  uint lo = diagonal > nb ? diagonal - nb : 0;
  uint hi = min(diagonal, na);
  while (lo < hi) {
    const uint i = lo + (hi - lo) / 2;
    const uint j = diagonal - i;
    if (i < na && j > 0 && b[j - 1] >= a[i]) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  return lo;
}

inline uint local_corank(
    threadgroup const uint* a, uint na,
    threadgroup const uint* b, uint nb, uint diagonal) {
  uint lo = diagonal > nb ? diagonal - nb : 0;
  uint hi = min(diagonal, na);
  while (lo < hi) {
    const uint i = lo + (hi - lo) / 2;
    const uint j = diagonal - i;
    if (i < na && j > 0 && b[j - 1] >= a[i]) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  return lo;
}

kernel void mccl_merge_u32(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& na [[buffer(3)]],
    constant uint& nb [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]) {
  // 2 * 1024 * 4 + 2 * 4 = 8200 static threadgroup bytes.
  threadgroup uint staged[kTile];
  threadgroup uint merged[kTile];
  threadgroup uint a_boundary[2];

  const uint begin = tile * kTile;
  const uint end = min(begin + kTile, na + nb);
  const uint count = end - begin;
  if (tid < 2) {
    a_boundary[tid] = global_corank(a, na, b, nb, tid == 0 ? begin : end);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint a_begin = a_boundary[0];
  const uint b_begin = begin - a_begin;
  // Sorted inputs already have monotonic partitions. Clamp for memory safety
  // on arbitrary input: staged A/B lengths must be nonnegative and sum to count.
  const uint a_end = clamp(
      a_boundary[1], max(a_begin, end > nb ? end - nb : 0u),
      min(na, a_begin + count));
  const uint a_count = a_end - a_begin;
  const uint b_count = count - a_count;
  for (uint i = tid; i < count; i += kThreads) {
    if (i < a_count) {
      staged[i] = a[a_begin + i];
    } else {
      staged[i] = b[b_begin + i - a_count];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint diagonal = min(tid * kPerThread, count);
  uint ai = local_corank(staged, a_count, staged + a_count, b_count, diagonal);
  uint bi = diagonal - ai;
  const uint stop = min(diagonal + kPerThread, count);
  for (uint i = diagonal; i < stop; ++i) {
    // No sentinel value (UINT_MAX is valid input), and no speculative tail load.
    if (ai == a_count) {
      merged[i] = staged[a_count + bi++];
    } else if (bi == b_count) {
      merged[i] = staged[ai++];
    } else if (staged[ai] <= staged[a_count + bi]) {
      merged[i] = staged[ai++];
    } else {
      merged[i] = staged[a_count + bi++];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Transpose each thread's serial results into contiguous SIMD-group stores.
  for (uint i = tid; i < count; i += kThreads) {
    out[begin + i] = merged[i];
  }
}

kernel void mccl_copy_u32_kernel(
    device const uint* input [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]) {
  const uint begin = tile * kTile;
  for (uint i = tid; i < kTile; i += kThreads) {
    const uint index = begin + i;
    if (index < count) {
      out[index] = input[index];
    }
  }
}
