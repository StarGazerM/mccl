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

kernel void mccl_difference_scatter_u32(
    device const uint* input [[buffer(0)]],
    device const uint* keep [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device uint* storage [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]) {
  const uint begin = tile * kTile;
  for (uint i = tid; i < kTile; i += kThreads) {
    const uint index = begin + i;
    if (index < capacity && keep[index] != 0) {
      const uint offset = offsets[index];
      // Exclusive scan of 0/1 flags gives unique, increasing destinations.
      // The guard also keeps the write bounded independently of sortedness.
      if (offset < capacity) {
        storage[offset] = input[index];
      }
    }
  }
}

constant uint kBloomSalts[8] = {
    0x47b6137bu, 0x44974d91u, 0x8824ad5bu, 0xa2b7289du,
    0x705495c7u, 0x2df1424bu, 0x9efc4947u, 0x05c6fb31u};

// xxHash64's four-byte little-endian short path, not a hash of widened keys.
inline ulong bloom_hash_u32(uint key, ulong seed) {
  constexpr ulong prime1 = 11400714785074694791ul;
  constexpr ulong prime2 = 14029467366897019727ul;
  constexpr ulong prime3 = 1609587929392839161ul;
  constexpr ulong prime5 = 2870177450012600261ul;
  ulong hash = seed + prime5 + 4ul;
  hash ^= ulong(key) * prime1;
  hash = ((hash << 23) | (hash >> 41)) * prime2 + prime3;
  hash ^= hash >> 33;
  hash *= prime2;
  hash ^= hash >> 29;
  hash *= prime3;
  hash ^= hash >> 32;
  return hash;
}

inline uint bloom_block_base(ulong hash, uint num_blocks) {
  return uint(((hash >> 32) * ulong(num_blocks)) >> 32) * 8u;
}

inline uint bloom_mask(uint hash, uint word) {
  return 1u << ((kBloomSalts[word] * hash) >> 27);
}

kernel void mccl_bloom_clear_kernel(
    device uint* words [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    words[index] = 0u;
  }
}

kernel void mccl_bloom_add_kernel(
    device const uint* keys [[buffer(0)]],
    device atomic_uint* words [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& num_blocks [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  const ulong hash = bloom_hash_u32(keys[index], seed);
  const uint base = bloom_block_base(hash, num_blocks);
  for (uint word = 0; word < 8; ++word) {
    atomic_fetch_or_explicit(
        &words[base + word], bloom_mask(uint(hash), word), memory_order_relaxed);
  }
}

kernel void mccl_bloom_contains_kernel(
    device const uint* keys [[buffer(0)]],
    device const uint* words [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& num_blocks [[buffer(4)]],
    constant ulong& seed [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  const ulong hash = bloom_hash_u32(keys[index], seed);
  const uint base = bloom_block_base(hash, num_blocks);
  uint present = 1u;
  for (uint word = 0; word < 8; ++word) {
    present &= uint((words[base + word] & bloom_mask(uint(hash), word)) != 0u);
  }
  out[index] = present;
}
