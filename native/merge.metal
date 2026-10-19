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

constant uint kMapEmpty = 0xffffffffu;
constant uint kMapTombstone = 0xfffffffeu;

// xxHash32's four-byte little-endian short path with a uint32 seed.
inline uint map_hash_u32(uint key, uint seed) {
  constexpr uint prime2 = 2246822519u;
  constexpr uint prime3 = 3266489917u;
  constexpr uint prime4 = 668265263u;
  constexpr uint prime5 = 374761393u;
  uint hash = seed + prime5 + 4u;
  hash += key * prime3;
  hash = ((hash << 17) | (hash >> 15)) * prime4;
  hash ^= hash >> 15;
  hash *= prime2;
  hash ^= hash >> 13;
  hash *= prime3;
  hash ^= hash >> 16;
  return hash;
}

inline uint map_next_slot(uint slot, uint capacity) {
  return slot + 1u == capacity ? 0u : slot + 1u;
}

// Empty/tombstone -> live transitions are monotonic within a claim dispatch.
// Tombstone reuse is safe only for keys absent from the unchanged prebatch
// table; prelookup routes existing keys directly to their original slots.
template <bool ReuseTombstones = false>
inline uint map_claim_slot(
    device atomic_uint* table,
    uint key,
    uint capacity,
    uint seed,
    thread bool& inserted,
    thread bool& reused) {
  uint slot = map_hash_u32(key, seed) % capacity;
  for (uint probe = 0; probe < capacity; ++probe) {
    uint observed = atomic_load_explicit(&table[slot], memory_order_relaxed);
    // Retry weak CAS at the same free slot so duplicates cannot pass it.
    while (observed == kMapEmpty ||
           (ReuseTombstones && observed == kMapTombstone)) {
      const bool was_tombstone = observed == kMapTombstone;
      if (atomic_compare_exchange_weak_explicit(
              &table[slot], &observed, key,
              memory_order_relaxed, memory_order_relaxed)) {
        inserted = true;
        reused = was_tombstone;
        return slot;
      }
    }
    if (observed == key) {
      return slot;
    }
    slot = map_next_slot(slot, capacity);
  }
  return kMapEmpty;
}

kernel void mccl_map_clear_kernel(
    device uint* keys [[buffer(0)]],
    device atomic_uint* winners [[buffer(1)]],
    constant uint& capacity [[buffer(2)]],
    constant uint& winner_count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < capacity) {
    keys[index] = kMapEmpty;
  }
  // Existing winner storage is already zero after completed insertions.
  if (index < winner_count) {
    atomic_store_explicit(&winners[index], 0u, memory_order_relaxed);
  }
}

kernel void mccl_map_reset_stats_kernel(
    device uint* stats [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  if (index < 4u) {
    stats[index] = 0u;
  }
}

kernel void mccl_map_insert_kernel(
    device const uint* input [[buffer(0)]],
    device atomic_uint* keys [[buffer(1)]],
    device atomic_uint* winners [[buffer(2)]],
    device atomic_uint* stats [[buffer(3)]],
    device uint* slots [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& seed [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  const uint key = input[index];
  if (key >= kMapTombstone) {
    if (count <= capacity / 8) slots[index] = kMapEmpty;
    atomic_fetch_add_explicit(&stats[2], 1u, memory_order_relaxed);
    return;
  }
  bool inserted = false;
  bool reused = false;
  const uint slot = map_claim_slot(keys, key, capacity, seed, inserted, reused);
  if (count <= capacity / 8) slots[index] = slot;
  if (slot == kMapEmpty) {
    atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
    return;
  }
  if (inserted) atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
  atomic_fetch_max_explicit(&winners[slot], index + 1u, memory_order_relaxed);
}

kernel void mccl_map_prelookup_kernel(
    device const uint* input [[buffer(0)]],
    device const uint* keys [[buffer(1)]],
    device uint* slots [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  slots[index] = kMapEmpty;
  const uint key = input[index];
  if (key >= kMapTombstone) return;
  uint slot = map_hash_u32(key, seed) % capacity;
  for (uint probe = 0; probe < capacity; ++probe) {
    const uint observed = keys[slot];
    if (observed == kMapEmpty) return;
    if (observed == key) {
      slots[index] = slot;
      return;
    }
    slot = map_next_slot(slot, capacity);
  }
}

kernel void mccl_map_insert_reuse_kernel(
    device const uint* input [[buffer(0)]],
    device atomic_uint* keys [[buffer(1)]],
    device atomic_uint* winners [[buffer(2)]],
    device atomic_uint* stats [[buffer(3)]],
    device uint* slots [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& seed [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  const uint key = input[index];
  if (key >= kMapTombstone) {
    atomic_fetch_add_explicit(&stats[2], 1u, memory_order_relaxed);
    return;
  }
  uint slot = slots[index];
  bool inserted = false;
  bool reused = false;
  if (slot == kMapEmpty) {
    slot = map_claim_slot<true>(keys, key, capacity, seed, inserted, reused);
    if (count <= capacity / 8) slots[index] = slot;
  }
  if (slot == kMapEmpty) {
    atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
    return;
  }
  if (inserted) {
    atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
    if (reused) atomic_fetch_add_explicit(&stats[3], 1u, memory_order_relaxed);
  }
  atomic_fetch_max_explicit(&winners[slot], index + 1u, memory_order_relaxed);
}

kernel void mccl_map_publish_kernel(
    device const uint* input_values [[buffer(0)]],
    device atomic_uint* winners [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    device uint* values [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (count <= capacity / 8) {
    if (index >= count) return;
    const uint slot = slots[index];
    // Losing duplicates may read while the winner clears its ordinal. Every
    // access must be atomic; observing zero cannot elect a second publisher.
    if (slot < capacity &&
        atomic_load_explicit(&winners[slot], memory_order_relaxed) == index + 1u) {
      values[slot] = input_values[index];
      atomic_store_explicit(&winners[slot], 0u, memory_order_relaxed);
    }
  } else if (index < capacity) {
    const uint winner = atomic_load_explicit(&winners[index], memory_order_relaxed);
    if (winner != 0u) {
      values[index] = input_values[winner - 1u];
      atomic_store_explicit(&winners[index], 0u, memory_order_relaxed);
    }
  }
}

constant uint kMapProbeWidth [[function_constant(0)]];

kernel void mccl_map_find_kernel(
    device const uint* input [[buffer(0)]],
    device const uint* keys [[buffer(1)]],
    device const uint* values [[buffer(2)]],
    device uint* out_values [[buffer(3)]],
    device uint* out_found [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& seed [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  out_values[index] = 0u;
  out_found[index] = 0u;
  const uint key = input[index];
  if (key >= kMapTombstone) {
    return;
  }
  uint slot = map_hash_u32(key, seed) % capacity;
  if (kMapProbeWidth > 1u) {
    // Ordered queries permit an aligned vector load without atomic access.
    // Mask the initial prefix and final wraparound suffix so the first
    // key-or-empty encountered is identical to scalar linear probing.
    uint base = slot & ~(kMapProbeWidth - 1u);
    uint skip = slot & (kMapProbeWidth - 1u);
    for (uint examined = 0; examined < capacity;) {
      uint hit = 0u;
      uint empty = 0u;
      for (uint vector = 0; vector < kMapProbeWidth / 4; ++vector) {
        const uint4 chunk =
            reinterpret_cast<device const uint4*>(keys)[base / 4 + vector];
        hit |= (uint(chunk.x == key) | (uint(chunk.y == key) << 1) |
                (uint(chunk.z == key) << 2) | (uint(chunk.w == key) << 3))
            << (4u * vector);
        empty |= (uint(chunk.x == kMapEmpty) |
                  (uint(chunk.y == kMapEmpty) << 1) |
                  (uint(chunk.z == kMapEmpty) << 2) |
                  (uint(chunk.w == kMapEmpty) << 3)) << (4u * vector);
      }
      const uint available = min(kMapProbeWidth - skip, capacity - examined);
      const uint active = ((1u << available) - 1u) << skip;
      const uint stop = (hit | empty) & active;
      if (stop != 0u) {
        const uint lane = ctz(stop);
        if ((hit & (1u << lane)) != 0u) {
          out_values[index] = values[base + lane];
          out_found[index] = 1u;
        }
        return;
      }
      examined += available;
      base += kMapProbeWidth;
      if (base == capacity) base = 0;
      skip = 0;
    }
    return;
  }
  for (uint probe = 0; probe < capacity; ++probe) {
    const uint observed = keys[slot];
    if (observed == kMapEmpty) {
      return;
    }
    if (observed == key) {
      out_values[index] = values[slot];
      out_found[index] = 1u;
      return;
    }
    slot = map_next_slot(slot, capacity);
  }
}

kernel void mccl_map_erase_kernel(
    device const uint* input [[buffer(0)]],
    device atomic_uint* keys [[buffer(1)]],
    device atomic_uint* stats [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  const uint key = input[index];
  if (key >= kMapTombstone) {
    return;
  }
  uint slot = map_hash_u32(key, seed) % capacity;
  for (uint probe = 0; probe < capacity; ++probe) {
    uint observed = atomic_load_explicit(&keys[slot], memory_order_relaxed);
    if (observed == kMapEmpty) {
      return;
    }
    if (observed == key) {
      do {
        if (atomic_compare_exchange_weak_explicit(
                &keys[slot], &observed, kMapTombstone,
                memory_order_relaxed, memory_order_relaxed)) {
          atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
          return;
        }
      } while (observed == key);
      // Another occurrence erased this unique key; only its CAS counts.
      return;
    }
    slot = map_next_slot(slot, capacity);
  }
}

kernel void mccl_map_rehash_kernel(
    device const uint* old_keys [[buffer(0)]],
    device const uint* old_values [[buffer(1)]],
    device atomic_uint* keys [[buffer(2)]],
    device uint* values [[buffer(3)]],
    constant uint& old_capacity [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    constant uint& seed [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= old_capacity) {
    return;
  }
  const uint key = old_keys[index];
  if (key >= kMapTombstone) {
    return;
  }
  bool inserted = false;
  bool reused = false;
  const uint slot = map_claim_slot(
      keys, key, capacity, seed, inserted, reused);
  // Old keys are unique and capacity >= live size: each claims one slot.
  // Keep the guard as a memory-safety bound independent of that invariant.
  if (slot != kMapEmpty) {
    values[slot] = old_values[index];
  }
}
