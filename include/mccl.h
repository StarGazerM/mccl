#ifndef MCCL_H
#define MCCL_H

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

/* Thread-local message, valid until the next mccl operation on this thread.
 * Empty after a successful operation. The returned pointer must not be freed.
 */
MCCL_API const char* mccl_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
