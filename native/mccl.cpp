#include "mccl.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

struct mccl_bloom {
  mccl_bloom(mx::Stream stream, uint32_t num_blocks, uint64_t seed)
      : stream(stream),
        num_blocks(num_blocks),
        seed(seed),
        owner(std::this_thread::get_id()),
        bits({static_cast<int>(num_blocks) * 8}, mx::uint32, nullptr, {}) {
    bits.set_data(mx::allocator::malloc(bits.nbytes()));
  }

  const mx::Stream stream;
  const uint32_t num_blocks;
  const uint64_t seed;
  const std::thread::id owner;
  mx::array bits;
};

namespace {

constexpr size_t kThreads = 256;
constexpr size_t kTile = 1024;
thread_local char last_error[1024] = {};

enum class Operation { Merge, Copy };

// Resolve relative to this dylib, never to the process working directory.
const std::string& library_directory() {
  static const std::string directory = [] {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void*>(&mccl_last_error), &info) ||
        !info.dli_fname) {
      throw std::runtime_error("mccl: cannot locate the native shared library");
    }
    return std::filesystem::canonical(info.dli_fname).parent_path().string();
  }();
  return directory;
}

class U32Operation final : public mx::Primitive {
 public:
  U32Operation(mx::Stream stream, Operation operation)
      : mx::Primitive(stream), operation_(operation) {}

  const char* name() const override {
    return operation_ == Operation::Merge ? "McclMergeSortedU32" : "McclCopyU32";
  }

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("mccl operations require a GPU stream");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    auto& out = outputs[0];
    // Never donate or alias an input, including for the copy baseline.
    out.set_data(mx::allocator::malloc(out.nbytes()));
    if (out.size() == 0) {
      return;
    }

    auto& device = mx::metal::device(stream().device);
    auto* library = device.get_library("mccl", library_directory());
    const bool merge = operation_ == Operation::Merge;
    auto* kernel = device.get_kernel(
        merge ? "mccl_merge_u32" : "mccl_copy_u32_kernel", library);
    if (kernel->maxTotalThreadsPerThreadgroup() < kThreads) {
      throw std::runtime_error("mccl kernel requires 256 threads per threadgroup");
    }

    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    if (merge) {
      const auto na = static_cast<uint32_t>(inputs[0].size());
      const auto nb = static_cast<uint32_t>(inputs[1].size());
      // A zero-sized MLX array can have no Metal buffer. Bind the other input
      // instead; its corresponding zero count ensures that pointer is not read.
      encoder.set_input_array(na != 0 ? inputs[0] : inputs[1], 0);
      encoder.set_input_array(nb != 0 ? inputs[1] : inputs[0], 1);
      encoder.set_output_array(out, 2);
      encoder.set_bytes(na, 3);
      encoder.set_bytes(nb, 4);
    } else {
      const auto count = static_cast<uint32_t>(inputs[0].size());
      encoder.set_input_array(inputs[0], 0);
      encoder.set_output_array(out, 1);
      encoder.set_bytes(count, 2);
    }
    encoder.dispatch_threadgroups(
        MTL::Size((out.size() + kTile - 1) / kTile, 1, 1),
        MTL::Size(kThreads, 1, 1));
  }

 private:
  Operation operation_;
};

class DifferenceScatter final : public mx::Primitive {
 public:
  explicit DifferenceScatter(mx::Stream stream) : mx::Primitive(stream) {}

  const char* name() const override {
    return "McclDifferenceScatterU32";
  }

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("mccl operations require a GPU stream");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    auto& out = outputs[0];
    // Only the retained prefix will be initialized; never donate an input.
    out.set_data(mx::allocator::malloc(out.nbytes()));
    if (out.size() == 0) {
      return;
    }

    auto& device = mx::metal::device(stream().device);
    auto* library = device.get_library("mccl", library_directory());
    auto* kernel = device.get_kernel("mccl_difference_scatter_u32", library);
    if (kernel->maxTotalThreadsPerThreadgroup() < kThreads) {
      throw std::runtime_error("mccl kernel requires 256 threads per threadgroup");
    }

    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_output_array(out, 3);
    const auto capacity = static_cast<uint32_t>(out.size());
    encoder.set_bytes(capacity, 4);
    encoder.dispatch_threadgroups(
        MTL::Size((out.size() + kTile - 1) / kTile, 1, 1),
        MTL::Size(kThreads, 1, 1));
  }
};

void validate_input(const mx::array& input) {
  if (input.ndim() != 1 || input.dtype() != mx::uint32) {
    throw std::invalid_argument("mccl inputs must be one-dimensional uint32 arrays");
  }
  if (input.size() > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("mccl input length exceeds INT_MAX");
  }
}

mx::Stream validate_stream(mlx_stream stream) {
  auto result = mlx_stream_get_(stream);
  if (result.device.type != mx::Device::gpu) {
    throw std::invalid_argument("mccl operations require a GPU stream");
  }
  return result;
}

mccl_bloom& validate_bloom(mccl_bloom* filter) {
  if (!filter) {
    throw std::invalid_argument("mccl Bloom filter must not be null");
  }
  if (filter->owner != std::this_thread::get_id()) {
    throw std::invalid_argument("mccl Bloom filter must be used on its creating thread");
  }
  return *filter;
}

mx::array prepare_bloom_keys(const mccl_bloom& filter, mlx_array keys) {
  const auto& input = mlx_array_get_(keys);
  validate_input(input);
  auto contiguous = mx::contiguous(input, false, filter.stream);
  // array::eval waits for the entire dependency graph, including other streams.
  // Do this before encoding direct accesses outside MLX's lazy scheduler.
  contiguous.eval();
  return contiguous;
}

MTL::ComputePipelineState* bloom_kernel(
    const mccl_bloom& filter, const char* name) {
  auto& device = mx::metal::device(filter.stream.device);
  auto* library = device.get_library("mccl", library_directory());
  auto* kernel = device.get_kernel(name, library);
  if (kernel->maxTotalThreadsPerThreadgroup() < kThreads) {
    throw std::runtime_error("mccl kernel requires 256 threads per threadgroup");
  }
  return kernel;
}

void clear_bloom(mccl_bloom& filter) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto* kernel = bloom_kernel(filter, "mccl_bloom_clear_kernel");
  auto& encoder = mx::metal::get_command_encoder(filter.stream);
  // Command buffers use unretained references. Retain the array even on an
  // encoding/commit exception; end_encoding transfers it to completion cleanup.
  // Private bits and unpublished results never need cross-stream fence tracking.
  encoder.add_temporary(filter.bits);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_output_array(filter.bits, 0);
  const auto count = static_cast<uint32_t>(filter.bits.size());
  encoder.set_bytes(count, 1);
  encoder.dispatch_threadgroups(
      MTL::Size((count + kThreads - 1) / kThreads, 1, 1),
      MTL::Size(kThreads, 1, 1));
  // Pinned MLX synchronize ends encoding, commits THIS command buffer, waits
  // until completion, then throws any GPU error. A separate commit is needless.
  encoder.synchronize();
  filter.bits.set_status(mx::array::Status::available);
}

template <typename Function>
int capture_error(Function&& function) noexcept {
  try {
    function();
    last_error[0] = '\0';
    return 0;
  } catch (const std::exception& error) {
    std::snprintf(last_error, sizeof(last_error), "%s", error.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "%s", "mccl: unknown C++ exception");
  }
  return 1;
}

} // namespace

extern "C" int mccl_merge_sorted_u32(
    mlx_array* out, mlx_array a, mlx_array b, mlx_stream stream) {
  return capture_error([&] {
    if (!out) {
      throw std::invalid_argument("mccl output pointer must not be null");
    }
    const auto& left = mlx_array_get_(a);
    const auto& right = mlx_array_get_(b);
    validate_input(left);
    validate_input(right);
    const size_t count = left.size() + right.size();
    if (count > static_cast<size_t>(INT_MAX)) {
      throw std::invalid_argument("mccl combined input length exceeds INT_MAX");
    }
    const auto s = validate_stream(stream);
    mx::array result(
        {static_cast<int>(count)}, mx::uint32,
        std::make_shared<U32Operation>(s, Operation::Merge),
        {mx::contiguous(left, false, s), mx::contiguous(right, false, s)});
    mlx_array_set_(*out, std::move(result));
  });
}

extern "C" int mccl_copy_u32(mlx_array* out, mlx_array a, mlx_stream stream) {
  return capture_error([&] {
    if (!out) {
      throw std::invalid_argument("mccl output pointer must not be null");
    }
    const auto& input = mlx_array_get_(a);
    validate_input(input);
    const auto s = validate_stream(stream);
    mx::array result(
        input.shape(), mx::uint32,
        std::make_shared<U32Operation>(s, Operation::Copy),
        {mx::contiguous(input, false, s)});
    mlx_array_set_(*out, std::move(result));
  });
}

extern "C" int mccl_difference_sorted_u32(
    mlx_array* storage,
    mlx_array* count,
    mlx_array a,
    mlx_array b,
    mlx_stream stream) {
  return capture_error([&] {
    if (!storage || !count) {
      throw std::invalid_argument("mccl output pointers must not be null");
    }
    if (storage == count || (storage->ctx && storage->ctx == count->ctx)) {
      throw std::invalid_argument("mccl difference outputs must be distinct handles");
    }
    const auto& left = mlx_array_get_(a);
    const auto& right = mlx_array_get_(b);
    validate_input(left);
    validate_input(right);
    const auto s = validate_stream(stream);
    auto values = mx::contiguous(left, false, s);
    auto flags = [&] {
      if (left.size() == 0) {
        return mx::zeros(left.shape(), mx::uint32, s);
      }
      if (right.size() == 0) {
        return mx::ones(left.shape(), mx::uint32, s);
      }
      auto removals = mx::contiguous(right, false, s);
      auto ordinal = mx::arange(static_cast<double>(left.size()), mx::uint32, s);
      auto first = mx::searchsorted(values, values, "left", s);
      auto lower = mx::searchsorted(removals, values, "left", s);
      auto upper = mx::searchsorted(removals, values, "right", s);
      auto rank = mx::subtract(ordinal, first, s);
      auto multiplicity = mx::subtract(upper, lower, s);
      // On unsorted input the unsigned differences may wrap, but comparison
      // still produces only 0/1 flags. Their scan is bounded by len(A).
      return mx::astype(mx::greater_equal(rank, multiplicity, s), mx::uint32, s);
    }();
    flags = mx::contiguous(flags, false, s);
    auto offsets = left.size() == 0
        ? flags
        : mx::cumsum(flags, 0, false, false, mx::uint32, s);
    mx::array result(
        left.shape(), mx::uint32, std::make_shared<DifferenceScatter>(s),
        {values, flags, mx::contiguous(offsets, false, s)});
    auto result_count = left.size() == 0
        ? mx::zeros({}, mx::uint64, s)
        : mx::sum(mx::astype(flags, mx::uint64, s), s);

    // Build both graphs and allocate any missing handle contexts before
    // changing either output. Outputs may alias input handles, whose original
    // array descriptors have already been retained by the graphs above.
    auto storage_context = storage->ctx
        ? nullptr
        : std::make_unique<mx::array>(std::move(result));
    auto count_context = count->ctx
        ? nullptr
        : std::make_unique<mx::array>(std::move(result_count));
    if (storage_context) {
      storage->ctx = storage_context.release();
    } else {
      mlx_array_set_(*storage, std::move(result));
    }
    if (count_context) {
      count->ctx = count_context.release();
    } else {
      mlx_array_set_(*count, std::move(result_count));
    }
  });
}

extern "C" int mccl_bloom_create(
    mccl_bloom** out, uint32_t num_blocks, uint64_t seed, mlx_stream stream) {
  return capture_error([&] {
    if (!out) {
      throw std::invalid_argument("mccl Bloom output pointer must not be null");
    }
    if (num_blocks == 0 || num_blocks > static_cast<uint32_t>(INT_MAX / 8)) {
      throw std::invalid_argument(
          "mccl Bloom num_blocks must be positive and num_blocks * 8 <= INT_MAX");
    }
    auto filter = std::make_unique<mccl_bloom>(
        validate_stream(stream), num_blocks, seed);
    clear_bloom(*filter);
    *out = filter.release();
  });
}

extern "C" int mccl_bloom_free(mccl_bloom* filter) {
  return capture_error([&] {
    if (filter) {
      validate_bloom(filter);
      delete filter;
    }
  });
}

extern "C" int mccl_bloom_clear(mccl_bloom* filter) {
  return capture_error([&] { clear_bloom(validate_bloom(filter)); });
}

extern "C" int mccl_bloom_add(mccl_bloom* filter, mlx_array keys) {
  return capture_error([&] {
    auto& bloom = validate_bloom(filter);
    auto input = prepare_bloom_keys(bloom, keys);
    if (input.size() == 0) {
      mx::synchronize(bloom.stream);
      return;
    }
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* kernel = bloom_kernel(bloom, "mccl_bloom_add_kernel");
    auto& encoder = mx::metal::get_command_encoder(bloom.stream);
    encoder.add_temporary(input);
    encoder.add_temporary(bloom.bits);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(input, 0);
    // set_output_array also registers a read dependency for the in-place OR.
    encoder.set_output_array(bloom.bits, 1);
    const auto count = static_cast<uint32_t>(input.size());
    encoder.set_bytes(count, 2);
    encoder.set_bytes(bloom.num_blocks, 3);
    encoder.set_bytes(bloom.seed, 4);
    encoder.dispatch_threadgroups(
        MTL::Size((count + kThreads - 1) / kThreads, 1, 1),
        MTL::Size(kThreads, 1, 1));
    encoder.synchronize();
  });
}

extern "C" int mccl_bloom_contains(
    mlx_array* out, mccl_bloom* filter, mlx_array keys) {
  return capture_error([&] {
    if (!out) {
      throw std::invalid_argument("mccl output pointer must not be null");
    }
    auto& bloom = validate_bloom(filter);
    auto input = prepare_bloom_keys(bloom, keys);
    mx::array result(input.shape(), mx::uint32, nullptr, {});
    result.set_data(mx::allocator::malloc(result.nbytes()));
    if (input.size() != 0) {
      auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
      auto* kernel = bloom_kernel(bloom, "mccl_bloom_contains_kernel");
      auto& encoder = mx::metal::get_command_encoder(bloom.stream);
      encoder.add_temporary(input);
      encoder.add_temporary(bloom.bits);
      encoder.add_temporary(result);
      encoder.set_compute_pipeline_state(kernel);
      encoder.set_input_array(input, 0);
      encoder.set_input_array(bloom.bits, 1);
      encoder.set_output_array(result, 2);
      const auto count = static_cast<uint32_t>(input.size());
      encoder.set_bytes(count, 3);
      encoder.set_bytes(bloom.num_blocks, 4);
      encoder.set_bytes(bloom.seed, 5);
      encoder.dispatch_threadgroups(
          MTL::Size((count + kThreads - 1) / kThreads, 1, 1),
          MTL::Size(kThreads, 1, 1));
      encoder.synchronize();
    } else {
      mx::synchronize(bloom.stream);
    }
    result.set_status(mx::array::Status::available);
    mlx_array_set_(*out, std::move(result));
  });
}

extern "C" int mccl_bloom_read_words(
    mccl_bloom* filter, uint32_t* out, size_t words) {
  return capture_error([&] {
    auto& bloom = validate_bloom(filter);
    if (!out) {
      throw std::invalid_argument("mccl Bloom word output pointer must not be null");
    }
    if (words != bloom.bits.size()) {
      throw std::invalid_argument("mccl Bloom readback requires exactly num_blocks * 8 words");
    }
    mx::synchronize(bloom.stream);
    std::memcpy(out, bloom.bits.data<uint32_t>(), bloom.bits.nbytes());
  });
}

extern "C" const char* mccl_last_error(void) {
  return last_error;
}
