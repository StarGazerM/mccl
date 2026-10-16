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

/* Thread-local message, valid until the next mccl operation on this thread.
 * Empty after a successful operation. The returned pointer must not be freed.
 */
MCCL_API const char* mccl_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
