#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/device_memory.h"


#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations (cutlass_tensorop_h16816gemm_128x128_32x4_nn_align8)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::bfloat16_t;                            // 元素类型
using         LayoutA     = cutlass::layout::RowMajor;                      // 存储布局
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // 内存对齐方式

// B matrix configuration
using         ElementB    = cutlass::bfloat16_t;                           
using         LayoutB     = cutlass::layout::ColumnMajor;                   
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    

// C/D matrix configuration
using         ElementC    = cutlass::bfloat16_t;                         
using         LayoutC     = cutlass::layout::RowMajor;               
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;   

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator  = float;                                      // 内部累加器的元素类型为float
using ArchTag             = cutlass::arch::Sm80;                        // 架构为SM80（NVIDIA Ampere）
using OperatorClass       = cutlass::arch::OpClassTensorOp;             // 使用张量运算
using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 32>;     // Threadblock-level tile size 
using WarpShape           = cutlass::gemm::GemmShape<64, 64, 32>;       // Warp-level tile size 
using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 16>;        // Instruction-level tile size 
constexpr int NumStages   = 3;                                          // Number of global->shared pipeline stages used in the GEMM mainloop

// Epilogue output operator
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<    // 定义后处理操作符
    ElementC,               // Element type for C and D matrix operands
    AlignmentC,             // Memory access granularity of C and D matrix in units of elements
    ElementAccumulator,     // Element type from internal accumaccumulation
    ElementAccumulator>;    // Data type used to compute linear combination


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Reference device GEMM implementation type
/////////////////////////////////////////////////////////////////////////////////////////////////

using DeviceGemmReference = cutlass::reference::device::Gemm<       // 参考实现
  ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 
  ElementAccumulator,
  ElementAccumulator>;

// Classic data-parallel device GEMM implementation type
using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<       // 数据并行实现
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape, WarpShape, InstructionShape, 
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;

// StreamK device GEMM implementation type
using DeviceGemmStreamK = cutlass::gemm::device::GemmUniversal<     // Stream K实现
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape, WarpShape, InstructionShape, 
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, // <-- Only difference
    NumStages,
    AlignmentA,
    AlignmentB>;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Populates a DeviceGemmBasic::Arguments structure from the given commandline options
typename DeviceGemmBasic::Arguments args_from_options(
    cutlass::gemm::GemmCoord  problem_size, // {m, n, k}
    int split_k_factor,
    float alpha,
    float beta,
    const at::Tensor &tensor_a,
    const at::Tensor &tensor_b,
    at::Tensor &tensor_c,
    at::Tensor &tensor_d)
{
  return typename DeviceGemmBasic::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,    // universal mode
    problem_size,                               // problem_size
    split_k_factor,                             // batch count / splitk slices
    {                                           // epilogue parameters
      ElementAccumulator(alpha),
      ElementAccumulator(beta)
    },
    tensor_a.data_ptr(),                        // ptr_A
    tensor_b.data_ptr(),                        // ptr_B
    tensor_c.data_ptr(),                        // ptr_C
    tensor_d.data_ptr(),                        // ptr_D
    problem_size.mk().product(),                // batch_stride_A
    problem_size.nk().product(),                // batch_stride_B
    problem_size.mn().product(),                // batch_stride_C
    problem_size.mn().product(),                // batch_stride_D
    problem_size.k(),                           // stride_a
    problem_size.k(),                           // stride_b
    problem_size.n(),                           // stride_c
    problem_size.n());                          // stride_d
}

/// Populates a DeviceGemmStreamK::Arguments structure from the given commandline options
typename DeviceGemmStreamK::Arguments args_from_options(
    cutlass::gemm::GemmCoord  problem_size, // {m, n, k}
    int split_k_factor,
    float alpha,
    float beta,
    const at::Tensor &tensor_a,
    const at::Tensor &tensor_b,
    at::Tensor &tensor_c,
    at::Tensor &tensor_d,
    int avail_sms)
{
  return typename DeviceGemmStreamK::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,    // universal mode
    problem_size,                               // problem_size
    split_k_factor,                             // batch count / splitk slices
    {                                           // epilogue parameters
      ElementAccumulator(alpha),
      ElementAccumulator(beta)
    },
    tensor_a.data_ptr(),                        // ptr_A
    tensor_b.data_ptr(),                        // ptr_B
    tensor_c.data_ptr(),                        // ptr_C
    tensor_d.data_ptr(),                        // ptr_D
    problem_size.mk().product(),                // batch_stride_A
    problem_size.nk().product(),                // batch_stride_B
    problem_size.mn().product(),                // batch_stride_C
    problem_size.mn().product(),                // batch_stride_D
    problem_size.k(),                           // stride_a
    problem_size.k(),                           // stride_b
    problem_size.n(),                           // stride_c
    problem_size.n(),                           // stride_d
    avail_sms);                                 // avail_sms
}


at::Tensor GemmBasic(const at::Tensor &input, const at::Tensor &w0, const at::Tensor &b0, c10::optional<int> &split_k_factor_, c10::optional<at::Tensor> &out_) {
    // 确保输入类型为float16或bfloat16
    auto input_dtype = input.dtype();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kBFloat16,
                "GemmBasic only support bf16 data type");

    // 确保权重和偏差的类型与输入类型一致
    TORCH_CHECK(w0.dtype() == input_dtype, "input and weight 0 must have the same dtype");
    TORCH_CHECK(b0.dtype() == input_dtype, "input and bias 1 must have the same dtype");

    // 计算GEMM维度M、N、K
    const int M = input.numel() /  input.size(-1);
    const int N = w0.size(0);
    const int K = w0.size(1);

    // 初始化输出张量
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == input_dtype, "Output must have the same dtype as inputs");
    } else {
        out = torch::empty({M, N}, input.options());
    }

    // 获取split_k_factor
    int split_k_factor = 1;
    if (split_k_factor_.has_value()) {
        split_k_factor = split_k_factor_.value();    
    } 

    cutlass::gemm::GemmCoord problem_size({M, N, K});

    // 实例化CUTLASS GEMM内核
    DeviceGemmBasic device_gemm;

    float alpha = 1;
    float beta = 0;
    
    // 创建并初始化GEMM参数
    auto arguments = args_from_options(problem_size, split_k_factor, alpha, beta, input, w0, out, out);
    size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(device_gemm.can_implement(arguments));
    CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(device_gemm());

    return out;
}
 
at::Tensor GemmStreamK(const at::Tensor &input, const at::Tensor &w0, const at::Tensor &b0, c10::optional<int> &split_k_factor_, c10::optional<int> &avail_sms_, c10::optional<at::Tensor> &out_) {
    // 确保输入类型为float16或bfloat16
    auto input_dtype = input.dtype();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kBFloat16,
                "GemmBasic only support bf16 data type");

    // 确保权重和偏差的类型与输入类型一致
    TORCH_CHECK(w0.dtype() == input_dtype, "input and weight 0 must have the same dtype");
    TORCH_CHECK(b0.dtype() == input_dtype, "input and bias 1 must have the same dtype");

    // 计算GEMM维度M、N、K
    const int M = input.numel() /  input.size(-1);
    const int N = w0.size(0);
    const int K = w0.size(1);

    // 初始化输出张量
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == input_dtype, "Output must have the same dtype as inputs");
    } else {
        out = torch::empty({M, N}, input.options());
    }

    // 获取split_k_factor
    int split_k_factor = 1;
    if (split_k_factor_.has_value()) {
        split_k_factor = split_k_factor_.value();    
    }

    // 获取avail_sms
    int avail_sms = -1;
    if (avail_sms_.has_value()) {
        avail_sms = avail_sms_.value();    
    }

    cutlass::gemm::GemmCoord problem_size({M, N, K});

    // 实例化CUTLASS GEMM内核
    DeviceGemmStreamK device_gemm;

    float alpha = 1;
    float beta = 0;
    
    // 创建并初始化GEMM参数
    auto arguments = args_from_options(problem_size, split_k_factor, alpha, beta, input, w0, out, out, avail_sms);
    size_t workspace_size = DeviceGemmStreamK::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(device_gemm.can_implement(arguments));
    CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(device_gemm());

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cutlass";
    m.def("GemmBasic", &GemmBasic, "GemmBasic");
    m.def("GemmStreamK", &GemmStreamK, "GemmStreamK");
}