#include "mccl.h"

#include <algorithm>
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

struct mccl_map {
  mccl_map(mx::Stream stream, uint32_t capacity, uint32_t seed)
      : stream(stream),
        capacity(capacity),
        seed(seed),
        owner(std::this_thread::get_id()),
        keys({static_cast<int>(capacity)}, mx::uint32, nullptr, {}),
        values({static_cast<int>(capacity)}, mx::uint32, nullptr, {}),
        winners({static_cast<int>(capacity)}, mx::uint32, nullptr, {}),
        stats({4}, mx::uint32, nullptr, {}),
        slots({0}, mx::uint32, nullptr, {}) {
    keys.set_data(mx::allocator::malloc(keys.nbytes()));
    values.set_data(mx::allocator::malloc(values.nbytes()));
    winners.set_data(mx::allocator::malloc(winners.nbytes()));
    stats.set_data(mx::allocator::malloc(stats.nbytes()));
  }

  const mx::Stream stream;
  uint32_t capacity;
  const uint32_t seed;
  const std::thread::id owner;
  uint32_t size = 0;
  uint32_t tombstones = 0;
  mx::array keys;
  mx::array values;
  // Zero between completed insertions; publication clears every winning slot.
  mx::array winners;
  mx::array stats;
  // Reused across insert batches; only the current input prefix is accessed.
  mx::array slots;
  // Device owns these cached pipelines; avoid specialization-vector/string
  // allocation on every completed lookup after each width has been resolved.
  MTL::ComputePipelineState* lookup_kernels[4] = {};
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

mccl_map& validate_map(mccl_map* map) {
  if (!map) {
    throw std::invalid_argument("mccl map must not be null");
  }
  if (map->owner != std::this_thread::get_id()) {
    throw std::invalid_argument("mccl map must be used on its creating thread");
  }
  return *map;
}

void validate_map_capacity(uint32_t capacity, mx::Stream stream) {
  if (capacity == 0 || capacity > static_cast<uint32_t>(INT_MAX)) {
    throw std::invalid_argument("mccl map capacity must be positive and <= INT_MAX");
  }
  const auto max_bytes =
      mx::metal::device(stream.device).mtl_device()->maxBufferLength();
  if (static_cast<size_t>(capacity) * sizeof(uint32_t) > max_bytes) {
    throw std::invalid_argument("mccl map capacity exceeds the Metal buffer limit");
  }
}

mx::array allocate_map_array(uint32_t count) {
  mx::array result({static_cast<int>(count)}, mx::uint32, nullptr, {});
  result.set_data(mx::allocator::malloc(result.nbytes()));
  return result;
}

mx::array prepare_map_keys(const mccl_map& map, mlx_array keys) {
  const auto& input = mlx_array_get_(keys);
  validate_input(input);
  auto contiguous = mx::contiguous(input, false, map.stream);
  contiguous.eval();
  return contiguous;
}

MTL::ComputePipelineState* map_kernel(const mccl_map& map, const char* name) {
  auto& device = mx::metal::device(map.stream.device);
  auto* library = device.get_library("mccl", library_directory());
  auto* kernel = device.get_kernel(name, library);
  if (kernel->maxTotalThreadsPerThreadgroup() < kThreads) {
    throw std::runtime_error("mccl kernel requires 256 threads per threadgroup");
  }
  return kernel;
}

void dispatch_map(mx::metal::CommandEncoder& encoder, uint32_t count) {
  encoder.dispatch_threadgroups(
      MTL::Size((static_cast<size_t>(count) + kThreads - 1) / kThreads, 1, 1),
      MTL::Size(kThreads, 1, 1));
}

void clear_map(mccl_map& map, bool initialize_winners = false) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto* kernel = map_kernel(map, "mccl_map_clear_kernel");
  auto& encoder = mx::metal::get_command_encoder(map.stream);
  encoder.add_temporary(map.keys);
  encoder.add_temporary(map.winners);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_output_array(map.keys, 0);
  encoder.set_output_array(map.winners, 1);
  encoder.set_bytes(map.capacity, 2);
  const uint32_t winner_count = initialize_winners ? map.capacity : 0;
  encoder.set_bytes(winner_count, 3);
  dispatch_map(encoder, map.capacity);
  encoder.synchronize();
  map.keys.set_status(mx::array::Status::available);
  if (initialize_winners) {
    map.winners.set_status(mx::array::Status::available);
  }
  map.size = 0;
  map.tombstones = 0;
}

void rehash_map(mccl_map& map, uint32_t capacity) {
  validate_map_capacity(capacity, map.stream);
  if (capacity < map.size) {
    throw std::invalid_argument("mccl map rehash capacity is smaller than its size");
  }
  if (capacity == map.capacity && !map.tombstones) {
    mx::synchronize(map.stream);
    return;
  }

  // Allocate everything and resolve kernels before encoding. The old table is
  // read-only throughout the rebuild and remains installed until completion.
  auto keys = allocate_map_array(capacity);
  auto values = allocate_map_array(capacity);
  auto winners = capacity == map.capacity
      ? map.winners
      : allocate_map_array(capacity);
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto* clear = map_kernel(map, "mccl_map_clear_kernel");
  auto* rehash = map_kernel(map, "mccl_map_rehash_kernel");
  auto& encoder = mx::metal::get_command_encoder(map.stream);
  encoder.add_temporary(map.keys);
  encoder.add_temporary(map.values);
  encoder.add_temporary(keys);
  encoder.add_temporary(values);
  encoder.add_temporary(winners);
  encoder.set_compute_pipeline_state(clear);
  encoder.set_output_array(keys, 0);
  encoder.set_output_array(winners, 1);
  encoder.set_bytes(capacity, 2);
  const uint32_t winner_count = capacity == map.capacity ? 0 : capacity;
  encoder.set_bytes(winner_count, 3);
  dispatch_map(encoder, capacity);
  if (map.size != 0) {
    encoder.set_compute_pipeline_state(rehash);
    encoder.set_input_array(map.keys, 0);
    encoder.set_input_array(map.values, 1);
    encoder.set_output_array(keys, 2);
    encoder.set_output_array(values, 3);
    encoder.set_bytes(map.capacity, 4);
    encoder.set_bytes(capacity, 5);
    encoder.set_bytes(map.seed, 6);
    dispatch_map(encoder, map.capacity);
  }
  encoder.synchronize();
  keys.set_status(mx::array::Status::available);
  values.set_status(mx::array::Status::available);
  winners.set_status(mx::array::Status::available);
  map.keys = std::move(keys);
  map.values = std::move(values);
  map.winners = std::move(winners);
  map.capacity = capacity;
  map.tombstones = 0;
}

void reset_map_stats(
    mccl_map& map,
    mx::metal::CommandEncoder& encoder,
    MTL::ComputePipelineState* kernel) {
  encoder.add_temporary(map.stats);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_output_array(map.stats, 0);
  dispatch_map(encoder, 4);
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

extern "C" int mccl_map_create(
    mccl_map** out, uint32_t capacity, uint32_t seed, mlx_stream stream) {
  return capture_error([&] {
    if (!out) {
      throw std::invalid_argument("mccl map output pointer must not be null");
    }
    const auto s = validate_stream(stream);
    validate_map_capacity(capacity, s);
    auto map = std::make_unique<mccl_map>(s, capacity, seed);
    clear_map(*map, true);
    *out = map.release();
  });
}

extern "C" int mccl_map_free(mccl_map* map) {
  return capture_error([&] {
    if (map) {
      validate_map(map);
      delete map;
    }
  });
}

extern "C" int mccl_map_clear(mccl_map* map) {
  return capture_error([&] { clear_map(validate_map(map)); });
}

extern "C" int mccl_map_insert(
    mccl_map* map,
    mlx_array keys,
    mlx_array values,
    uint64_t* inserted,
    uint64_t* full,
    uint64_t* invalid) {
  return capture_error([&] {
    auto& table = validate_map(map);
    if (!inserted || !full || !invalid) {
      throw std::invalid_argument("mccl map report pointers must not be null");
    }
    if (inserted == full || inserted == invalid || full == invalid) {
      throw std::invalid_argument("mccl map report pointers must be distinct");
    }
    const auto& key_input = mlx_array_get_(keys);
    const auto& value_input = mlx_array_get_(values);
    validate_input(key_input);
    validate_input(value_input);
    if (key_input.size() != value_input.size()) {
      throw std::invalid_argument("mccl map keys and values must have equal lengths");
    }
    const auto max_bytes =
        mx::metal::device(table.stream.device).mtl_device()->maxBufferLength();
    if (key_input.size() * sizeof(uint32_t) > max_bytes) {
      throw std::invalid_argument("mccl map input exceeds the Metal buffer limit");
    }
    auto input_keys = mx::contiguous(key_input, false, table.stream);
    auto input_values = mx::contiguous(value_input, false, table.stream);
    // Evaluate both graphs before directly mutating storage on the map stream.
    input_keys.eval();
    input_values.eval();
    if (input_keys.size() == 0) {
      mx::synchronize(table.stream);
      *inserted = *full = *invalid = 0;
      return;
    }

    const auto count = static_cast<uint32_t>(input_keys.size());
    const bool sparse = count <= table.capacity / 8;
    const uint64_t capacity = table.capacity;
    // Conservative maintenance policy bounds accumulated probe-chain debris;
    // small erases otherwise reuse holes without allocating a replacement table.
    const bool maintenance = table.tombstones != 0 &&
        (table.size == 0 ||
         table.tombstones >= std::max<uint64_t>(1, capacity / 16) ||
         uint64_t(table.size) + table.tombstones >= (15 * capacity + 15) / 16);
    const bool reuse = table.tombstones != 0 && !maintenance;
    const bool needs_slots = sparse || reuse;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* reset = map_kernel(table, "mccl_map_reset_stats_kernel");
    auto* prelookup = reuse
        ? map_kernel(table, "mccl_map_prelookup_kernel")
        : nullptr;
    auto* claim = map_kernel(table, reuse
        ? "mccl_map_insert_reuse_kernel"
        : "mccl_map_insert_kernel");
    auto* publish = map_kernel(table, "mccl_map_publish_kernel");
    if (needs_slots && table.slots.size() < count) {
      table.slots = allocate_map_array(count);
    }
    if (maintenance) {
      if (table.size == 0) {
        clear_map(table);
      } else {
        rehash_map(table, table.capacity);
      }
    }
    auto& encoder = mx::metal::get_command_encoder(table.stream);
    encoder.add_temporary(input_keys);
    encoder.add_temporary(input_values);
    encoder.add_temporary(table.keys);
    encoder.add_temporary(table.values);
    encoder.add_temporary(table.winners);
    if (needs_slots) encoder.add_temporary(table.slots);
    reset_map_stats(table, encoder, reset);
    if (reuse) {
      encoder.set_compute_pipeline_state(prelookup);
      encoder.set_input_array(input_keys, 0);
      encoder.set_input_array(table.keys, 1);
      encoder.set_output_array(table.slots, 2);
      encoder.set_bytes(count, 3);
      encoder.set_bytes(table.capacity, 4);
      encoder.set_bytes(table.seed, 5);
      dispatch_map(encoder, count);
    }
    encoder.set_compute_pipeline_state(claim);
    encoder.set_input_array(input_keys, 0);
    encoder.set_output_array(table.keys, 1);
    encoder.set_output_array(table.winners, 2);
    encoder.set_output_array(table.stats, 3);
    // This read/write binding depends on prelookup's slot output even for dense
    // batches: MLX inserts a dispatch barrier before any key can be claimed.
    encoder.set_output_array(needs_slots ? table.slots : table.winners, 4);
    encoder.set_bytes(count, 5);
    encoder.set_bytes(table.capacity, 6);
    encoder.set_bytes(table.seed, 7);
    dispatch_map(encoder, count);

    // set_output_array also inserts the write/read barrier after the atomic
    // claim pass. Publication restores winners to zero before synchronization;
    // no query or later insertion can observe an unfinished publication.
    encoder.set_compute_pipeline_state(publish);
    encoder.set_input_array(input_values, 0);
    encoder.set_output_array(table.winners, 1);
    encoder.set_input_array(sparse ? table.slots : table.winners, 2);
    encoder.set_output_array(table.values, 3);
    encoder.set_bytes(count, 4);
    encoder.set_bytes(table.capacity, 5);
    dispatch_map(encoder, sparse ? count : table.capacity);
    encoder.synchronize();
    const auto* stats = table.stats.data<uint32_t>();
    table.size += stats[0];
    table.tombstones -= stats[3];
    *inserted = stats[0];
    *full = stats[1];
    *invalid = stats[2];
  });
}

extern "C" int mccl_map_find(
    mlx_array* values, mlx_array* found, mccl_map* map, mlx_array keys) {
  return capture_error([&] {
    if (!values || !found) {
      throw std::invalid_argument("mccl map lookup outputs must not be null");
    }
    if (values == found || (values->ctx && values->ctx == found->ctx)) {
      throw std::invalid_argument("mccl map lookup outputs must be distinct handles");
    }
    auto& table = validate_map(map);
    auto input = prepare_map_keys(table, keys);
    const auto count = static_cast<uint32_t>(input.size());
    auto result_values = allocate_map_array(count);
    auto result_found = allocate_map_array(count);
    if (count != 0) {
      auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
      const uint32_t width =
          table.capacity % 16 == 0 &&
              uint64_t(table.size + table.tombstones) * 4 >= uint64_t(table.capacity) * 3
          ? 16 : (table.capacity % 8 == 0 ? 8 : (table.capacity % 4 == 0 ? 4 : 1));
      const uint32_t variant = width == 1 ? 0 : (width == 4 ? 1 : (width == 8 ? 2 : 3));
      auto*& kernel = table.lookup_kernels[variant];
      if (!kernel) {
        auto& device = mx::metal::device(table.stream.device);
        auto* library = device.get_library("mccl", library_directory());
        auto* compiled = device.get_kernel(
            "mccl_map_find_kernel", library,
            "mccl_map_find_kernel_w" + std::to_string(width),
            {{&width, MTL::DataTypeUInt, 0}});
        if (compiled->maxTotalThreadsPerThreadgroup() < kThreads) {
          throw std::runtime_error("mccl kernel requires 256 threads per threadgroup");
        }
        kernel = compiled;
      }
      auto& encoder = mx::metal::get_command_encoder(table.stream);
      encoder.add_temporary(input);
      encoder.add_temporary(table.keys);
      encoder.add_temporary(table.values);
      encoder.add_temporary(result_values);
      encoder.add_temporary(result_found);
      encoder.set_compute_pipeline_state(kernel);
      encoder.set_input_array(input, 0);
      encoder.set_input_array(table.keys, 1);
      encoder.set_input_array(table.values, 2);
      encoder.set_output_array(result_values, 3);
      encoder.set_output_array(result_found, 4);
      encoder.set_bytes(count, 5);
      encoder.set_bytes(table.capacity, 6);
      encoder.set_bytes(table.seed, 7);
      dispatch_map(encoder, count);
      encoder.synchronize();
    } else {
      mx::synchronize(table.stream);
    }
    result_values.set_status(mx::array::Status::available);
    result_found.set_status(mx::array::Status::available);

    // As with difference, allocate missing contexts before publishing either
    // result. Input handles may alias an output: input retains its own storage.
    auto values_context = values->ctx
        ? nullptr
        : std::make_unique<mx::array>(std::move(result_values));
    auto found_context = found->ctx
        ? nullptr
        : std::make_unique<mx::array>(std::move(result_found));
    if (values_context) {
      values->ctx = values_context.release();
    } else {
      mlx_array_set_(*values, std::move(result_values));
    }
    if (found_context) {
      found->ctx = found_context.release();
    } else {
      mlx_array_set_(*found, std::move(result_found));
    }
  });
}

extern "C" int mccl_map_erase(
    mccl_map* map, mlx_array keys, uint64_t* erased) {
  return capture_error([&] {
    auto& table = validate_map(map);
    if (!erased) {
      throw std::invalid_argument("mccl map erased output pointer must not be null");
    }
    auto input = prepare_map_keys(table, keys);
    if (input.size() == 0) {
      mx::synchronize(table.stream);
      *erased = 0;
      return;
    }
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto* reset = map_kernel(table, "mccl_map_reset_stats_kernel");
    auto* kernel = map_kernel(table, "mccl_map_erase_kernel");
    auto& encoder = mx::metal::get_command_encoder(table.stream);
    encoder.add_temporary(input);
    encoder.add_temporary(table.keys);
    reset_map_stats(table, encoder, reset);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(input, 0);
    encoder.set_output_array(table.keys, 1);
    encoder.set_output_array(table.stats, 2);
    const auto count = static_cast<uint32_t>(input.size());
    encoder.set_bytes(count, 3);
    encoder.set_bytes(table.capacity, 4);
    encoder.set_bytes(table.seed, 5);
    dispatch_map(encoder, count);
    encoder.synchronize();
    const auto removed = table.stats.data<uint32_t>()[0];
    table.size -= removed;
    table.tombstones += removed;
    *erased = removed;
  });
}

extern "C" int mccl_map_rehash(mccl_map* map, uint32_t capacity) {
  return capture_error([&] { rehash_map(validate_map(map), capacity); });
}

extern "C" const char* mccl_last_error(void) {
  return last_error;
}
