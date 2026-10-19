#ifndef MCCL_H
#define MCCL_H

#include <stddef.h>
#include <stdint.h>

#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#if defined(__GNUC__)
#define MCCL_API __attribute__((visibility("default")))
#else
#define MCCL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Build a lazy GPU graph for stable ascending u32 merge. Equal values from a
 * precede those from b. Both inputs must be one-dimensional uint32 arrays and
 * their combined length must be <= INT_MAX. Sortedness is a precondition, not
 * host-validated: unsorted inputs produce unspecified values, but remain
 * memory-safe. Noncontiguous inputs are made contiguous on the supplied stream.
 * Empty inputs are supported. CPU streams are rejected.
 *
 * out must point to an initialized official mlx_array (mlx_array_new() is
 * sufficient); its previous value is replaced only on success. Input handles
 * may then be freed: the lazy graph retains their underlying arrays. Evaluate
 * out and synchronize s via mlx-c to materialize the result. Later evaluation
 * failures are reported by mlx-c's error handler, not mccl_last_error().
 *
 * Return 0 on success, 1 on error. No C++ exception crosses this C ABI. On error,
 * out is unchanged and mccl_last_error() describes the graph-construction error.
 */
MCCL_API int mccl_merge_sorted_u32(
    mlx_array* out, mlx_array a, mlx_array b, mlx_stream s);

/* Same shape/type/stream/ownership/error contract, with one input. Always
 * allocates independent output storage and dispatches a GPU read/write copy for
 * nonempty input: this is not MLX's potentially aliasing logical copy operation.
 */
MCCL_API int mccl_copy_u32(mlx_array* out, mlx_array a, mlx_stream s);

/*
 * Build a lazy GPU graph for stable sorted multiset difference a minus b.
 * Each occurrence in b cancels the earliest remaining equal occurrence in a.
 * Both inputs must be one-dimensional uint32 arrays, individually <= INT_MAX,
 * sorted in ascending order. Sortedness is not host-validated: unsorted inputs
 * produce unspecified values but remain memory-safe. Noncontiguous inputs are
 * made contiguous on s. Empty inputs are supported; CPU streams are rejected.
 *
 * storage has shape {len(a)} and dtype uint32. count is a scalar uint64 array.
 * ONLY storage[0:count] is initialized: never read or operate on its tail.
 * Graph construction does not evaluate arrays or read count on the host.
 * Evaluate both outputs and synchronize s before accessing the prefix; obtain
 * the host count explicitly with mlx_array_item_uint64.
 *
 * storage and count must point to initialized official mlx_array handles
 * (mlx_array_new() is sufficient). Their pointers and any nonnull contexts
 * must be distinct. Either may alias an input handle. Both output graphs are
 * built before installing either handle, and retain all required input arrays,
 * so input handles may subsequently be freed. Output storage never aliases a.
 *
 * Return 0 on success, 1 on error. On graph-construction error both outputs
 * remain unchanged and mccl_last_error() describes the error. Later evaluation
 * failures are reported by mlx-c's error handler, not mccl_last_error().
 */
MCCL_API int mccl_difference_sorted_u32(
    mlx_array* storage, mlx_array* count, mlx_array a, mlx_array b, mlx_stream s);

/*
 * Mutable blocked Bloom filter for arbitrary u32 keys, including 0/UINT32_MAX.
 * Each block is eight uint32 words (256 bits), using seeded xxHash64 over the
 * key's four little-endian bytes and the RCCL split-block salts. Positives are
 * probabilistic; completed insertion has no false negatives until clear.
 *
 * Handles are thread-affine: all operations, including free, must run on the
 * creating thread. The supplied GPU stream is retained by value for the entire
 * lifetime; its mlx_stream wrapper may be freed after creation. No operation
 * builds a lazy mutation graph. Creation/clear/add/contains commit and wait for
 * GPU completion before returning, without copying the bitset for insertion.
 * Inputs must be one-dimensional uint32 arrays with length <= INT_MAX; strided
 * inputs are made contiguous and evaluated before accessing the private bits.
 *
 * All functions return 0 on success and 1 on error, captured by mccl_last_error.
 * Null filter handles are errors except for free(NULL), which succeeds.
 * Argument errors do not mutate the filter. A GPU execution failure may leave
 * partially mutated bits and a failed MLX stream; mutation is not transactional.
 */
typedef struct mccl_bloom mccl_bloom;

/* num_blocks > 0 and num_blocks * 8 <= INT_MAX. The fixed seed is used for
 * both insertion and lookup. Writes a new zeroed filter to *out only on success;
 * out must not be null. The caller owns the returned handle.
 */
MCCL_API int mccl_bloom_create(
    mccl_bloom** out, uint32_t num_blocks, uint64_t seed, mlx_stream s);
MCCL_API int mccl_bloom_free(mccl_bloom* filter);
MCCL_API int mccl_bloom_clear(mccl_bloom* filter);
MCCL_API int mccl_bloom_add(mccl_bloom* filter, mlx_array keys);

/* out must point to an initialized official mlx_array (mlx_array_new() is
 * sufficient). On success it is replaced by an already-completed, independent
 * one-dimensional uint32 array of 0/1 membership answers, shape {len(keys)}.
 * Later mutations/frees cannot affect these answers. out may alias keys; on
 * error the previous output handle is unchanged. Empty keys are supported.
 */
MCCL_API int mccl_bloom_contains(
    mlx_array* out, mccl_bloom* filter, mlx_array keys);

/* Explicit synchronized CPU readback in block-major, then word-major order.
 * out must not be null and words must equal num_blocks * 8 exactly.
 */
MCCL_API int mccl_bloom_read_words(
    mccl_bloom* filter, uint32_t* out, size_t words);

/*
 * Fixed-capacity GPU uint32 -> uint32 hash map. UINT32_MAX is the empty-key
 * sentinel and UINT32_MAX - 1 is the tombstone-key sentinel: neither is a valid
 * key. Every uint32 value is valid. Hashing is seeded xxHash32 of the key's four
 * little-endian bytes, reduced modulo the exact capacity, with linear probing
 * bounded by capacity. Capacity is never rounded or grown automatically.
 *
 * Handles are thread-affine, retain the supplied GPU stream by value, and must
 * not be used concurrently. All operations are ordered, commit and wait for GPU
 * completion before returning; there are no lazy mutations. Inputs must be
 * one-dimensional uint32 arrays of length <= INT_MAX. Noncontiguous inputs are
 * materialized on the retained stream before accessing private map storage.
 *
 * All functions return 0 on success and 1 on error via mccl_last_error().
 * free(NULL) succeeds; other null map handles are errors. Argument errors leave
 * the map unchanged. Insert/erase/clear are not transactional once GPU encoding
 * begins: encoding, commit, or execution failure may leave storage and host
 * counts inconsistent, and the stream may have failed; discard the map.
 * Rehash publishes replacement buffers only after successful GPU completion,
 * leaving the original entries intact on failure.
 */
typedef struct mccl_map mccl_map;

/* out must not be null; *out changes only on success. Capacity must be positive,
 * <= INT_MAX, and each capacity * sizeof(uint32_t) buffer must fit the Metal
 * device's maximum buffer length. Allocation can still fail. Private storage
 * is three capacity-sized uint32 buffers plus four private uint32 counters, with
 * a reusable uint32 slot buffer for sparse batches or any tombstone-reuse batch.
 */
MCCL_API int mccl_map_create(
    mccl_map** out, uint32_t capacity, uint32_t seed, mlx_stream s);
MCCL_API int mccl_map_free(mccl_map* map);
MCCL_API int mccl_map_clear(mccl_map* map);

/* keys and values must have equal lengths. Reports must be nonnull, distinct
 * pointers and change only on success. inserted counts unique newly occupied
 * slots; full counts valid input occurrences not admitted because the table is
 * full; invalid counts reserved-key input occurrences. Partial admission returns
 * success with nonzero full/invalid. Existing keys are updated even when full.
 * The last input occurrence wins for every admitted key. Which distinct new
 * keys win insufficient remaining space is scheduling-dependent.
 *
 * Each nonempty call resets four private counters, claims keys using uint32
 * atomics, then publishes values and atomically clears the winning ordinals.
 * The fourth counter tracks reused tombstones; the public reports remain three.
 * The winner buffer stays zero between completed insertions; there is no
 * capacity-sized per-call reset. Batches <= capacity/8 record input slots and
 * publish across the batch; larger batches scan winners across capacity.
 * With no tombstones, larger batches avoid slot scratch. With tombstones, a
 * separate read-only prelookup records every input's existing slot or absence
 * before any claim: existing keys update their slots, absent keys may claim
 * empty slots or tombstones. Slot scratch grows without copying and is retained.
 * Erase counts unique tombstones. Before a nonempty insert, maintenance rebuilds
 * at the same capacity if tombstones >= max(1,capacity/16) or live+tombstones >=
 * ceil(15*capacity/16). This conservative policy limits accumulated probe chains;
 * it neither grows capacity nor limits admission below full capacity. Rebuild
 * scans the old capacity, probes survivors, and allocates replacement key/value
 * buffers. A table with tombstones but no live entries is cleared in place
 * instead. Maintenance is included in the call.
 */
MCCL_API int mccl_map_insert(
    mccl_map* map, mlx_array keys, mlx_array values,
    uint64_t* inserted, uint64_t* full, uint64_t* invalid);

/* values and found must point to distinct initialized mlx_array handles with
 * distinct nonnull contexts (mlx_array_new() is sufficient). Both handles are
 * replaced only on success, and may alias input handles. Returned arrays are
 * independent completed uint32 snapshots of shape {len(keys)}: found is 0/1;
 * values is zero for absent keys. Reserved query keys are always absent.
 */
MCCL_API int mccl_map_find(
    mlx_array* values, mlx_array* found, mccl_map* map, mlx_array keys);

/* erased must not be null and changes only on success. Duplicate input keys
 * count once; missing and reserved keys do not count. No rebuilding occurs in
 * erase itself; lookup continues through tombstones.
 */
MCCL_API int mccl_map_erase(
    mccl_map* map, mlx_array keys, uint64_t* erased);

/* Preserves every entry and the seed. Capacity has the create constraints and
 * must be >= the live entry count; rejection leaves the map unchanged. Rebuild
 * scans the old capacity and probes each live key into new key/value buffers;
 * a changed capacity also allocates and zeroes a new winner buffer. Old and new
 * storage coexist until successful completion. Same-capacity clean maps are a no-op.
 */
MCCL_API int mccl_map_rehash(mccl_map* map, uint32_t capacity);

/* Thread-local message, valid until the next mccl operation on this thread.
 * Empty after a successful operation. The returned pointer must not be freed.
 */
MCCL_API const char* mccl_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
