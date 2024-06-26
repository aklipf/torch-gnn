#include <torch/extension.h>

#include <vector>

namespace extension_cpp {

// An example of an operator that mutates one of its inputs.
void ptr_out_cpu(const at::Tensor &n, at::Tensor &out) {
  TORCH_CHECK(n.sizes() == out.sizes());
  TORCH_CHECK(n.dtype() == at::kLong);
  TORCH_CHECK(out.dtype() == at::kLong);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(n.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
  at::Tensor n_contig = n.contiguous();
  const long *n_ptr = n_contig.data_ptr<long>();
  long *result_ptr = out.data_ptr<long>();
  int numel = n_contig.numel();

  for (int64_t i = 0; i < numel; i++) {
    result_ptr[i] = n_ptr[i] + 1;
  }
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
  m.def("ptr_out(Tensor n, Tensor(n!) out) -> ()");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) { m.impl("ptr_out", &ptr_out_cpu); }

} // namespace extension_cpp
