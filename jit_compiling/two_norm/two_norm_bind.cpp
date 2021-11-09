#include <torch/extension.h>
#include "two_norm_kernel.h"

void torch_launch_two_norm(const torch::Tensor &a_tensor,
                           const torch::Tensor &b_tensor, torch::Tensor &c_tensor, int n, int m) {

    launch_two_norm((const float *)a_tensor.data_ptr(),
                    (const float *)b_tensor.data_ptr(),
                    (float *)c_tensor.data_ptr(),
                    n,m);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // 宏TORCH_EXTENSION_NAME的值在setup.py中定义，决定了什么
    m.def("two_norm",  // 函数在python中调用的名字
          &torch_launch_two_norm,  // 函数指针，需绑定的C++函数引用
          "compute 2-norm of two matrix"  // 函数说明
          );
}
