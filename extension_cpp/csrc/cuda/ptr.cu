#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE (256)

namespace extension_cpp
{

  __global__ void ptr_kernel(int numel, const long *n, long *out)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
      out[idx] = n[idx] + 1;
  }

  void ptr_out_cuda(const at::Tensor &n, at::Tensor &out)
  {
    TORCH_CHECK(n.sizes() == out.sizes());
    TORCH_CHECK(n.dtype() == at::kLong);
    TORCH_CHECK(out.dtype() == at::kLong);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(n.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    at::Tensor n_contig = n.contiguous();
    const long *n_ptr = n_contig.data_ptr<long>();
    long *result_ptr = out.data_ptr<long>();
    int numel = n_contig.numel();

    ptr_kernel<<<(numel + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numel, n_ptr, result_ptr);
  }

  // Registers CUDA implementations for mymuladd, mymul, myadd_out
  TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m)
  {
    m.impl("ptr_out", &ptr_out_cuda);
  }

}
