#include "mccl.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

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

extern "C" const char* mccl_last_error(void) {
  return last_error;
}
