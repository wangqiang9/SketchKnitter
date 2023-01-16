#include <torch/extension.h>
#include <vector>


void rasterize_cuda_forward(
    at::Tensor lines,
    at::Tensor intensities,
    at::Tensor line_map,
    at::Tensor line_index_map,
    at::Tensor line_weight_map,
    at::Tensor locks,
    int   img_size,
    float thickness,
    float eps);

void rasterize_cuda_backward(
    at::Tensor grad_intensities,
    at::Tensor grad_line_map,
    at::Tensor line_map,
    at::Tensor line_index_map,
    at::Tensor line_weight_map,
    at::Tensor lines,
    at::Tensor intensities,
    int   img_size,
    float thickness,
    float eps);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void rasterize_forward(
    at::Tensor lines,
    at::Tensor intensities,
    at::Tensor line_map,
    at::Tensor line_index_map,
    at::Tensor line_weight_map,
    at::Tensor locks,
    int   img_size,
    float thickness,
    float eps) {
    CHECK_INPUT(lines);
    CHECK_INPUT(intensities);
    CHECK_INPUT(line_map);
    CHECK_INPUT(line_index_map);
    CHECK_INPUT(line_weight_map);
    CHECK_INPUT(locks);

    return rasterize_cuda_forward(
        lines,
        intensities,
        line_map,
        line_index_map,
        line_weight_map,
        locks,
        img_size,
        thickness,
        eps);
}

void rasterize_backward(
    at::Tensor grad_intensities,
    at::Tensor grad_line_map,
    at::Tensor line_map,
    at::Tensor line_index_map,
    at::Tensor line_weight_map,
    at::Tensor lines,
    at::Tensor intensities,
    int   img_size,
    float thickness,
    float eps) {
    CHECK_INPUT(grad_intensities);
    CHECK_INPUT(grad_line_map);
    CHECK_INPUT(line_map);
    CHECK_INPUT(line_index_map);
    CHECK_INPUT(line_weight_map);
    CHECK_INPUT(lines);
    CHECK_INPUT(intensities);
    
    return rasterize_cuda_backward(
        grad_intensities,
        grad_line_map,
        line_map,
        line_index_map,
        line_weight_map,
        lines,
        intensities,
        img_size,
        thickness,
        eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward, "Rasterize forward (CUDA)");
    m.def("rasterize_backward", &rasterize_backward, "Rasterize backward (CUDA)");
}
