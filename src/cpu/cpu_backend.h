#pragma once
#include "../device_abstraction.h"
#include <torch/torch.h>

namespace device_abstraction {

// CPU implementation using tensor operations on torch::kCPU (mirrors MPS backend logic)
template <typename Scalar, typename IndexType>
class CPUMarchingCubesBackend : public MarchingCubesBackend<Scalar, IndexType> {
public:
    CPUMarchingCubesBackend();

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor grid,
        torch::Tensor deform,
        Scalar iso
    ) override;

    void backward(
        torch::Tensor grid,
        torch::Tensor deform,
        Scalar iso,
        torch::Tensor adj_verts,
        torch::Tensor adj_grid,
        torch::Tensor adj_deform
    ) override;

private:
    torch::Tensor compute_cube_codes(const torch::Tensor& grid, Scalar iso);
    std::tuple<torch::Tensor, torch::Tensor> build_mesh(
        const torch::Tensor& grid,
        const torch::Tensor& deform,
        const torch::Tensor& cube_codes,
        Scalar iso
    );

    torch::Device device_;
    torch::Tensor edge_table_;
    torch::Tensor tri_table_;
    torch::Tensor edge_connection_table_;
    torch::Tensor vertex_offset_table_;
};

// Minimal CPU dual marching cubes placeholder
template <typename Scalar, typename IndexType>
class CPUDualMarchingCubesBackend : public MarchingCubesBackend<Scalar, IndexType> {
public:
    CPUDualMarchingCubesBackend() = default;

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor grid,
        torch::Tensor deform,
        Scalar iso
    ) override;

    void backward(
        torch::Tensor grid,
        torch::Tensor deform,
        Scalar iso,
        torch::Tensor adj_verts,
        torch::Tensor adj_grid,
        torch::Tensor adj_deform
    ) override;
};

// Explicit template instantiation declarations
extern template class CPUMarchingCubesBackend<float, int>;
extern template class CPUMarchingCubesBackend<double, int>;
extern template class CPUDualMarchingCubesBackend<float, int>;
extern template class CPUDualMarchingCubesBackend<double, int>;

} // namespace device_abstraction
