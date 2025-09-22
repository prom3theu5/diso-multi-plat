#pragma once
#include "../device_abstraction.h"
#include <torch/torch.h>

namespace device_abstraction {

// MPS implementation using PyTorch native operations
template <typename Scalar, typename IndexType>
class MPSMarchingCubesBackend : public MarchingCubesBackend<Scalar, IndexType> {
private:
    torch::Device device_;
    
    // Lookup tables as tensors (will be moved to MPS device)
    torch::Tensor edge_table_;
    torch::Tensor tri_table_;
    torch::Tensor edge_connection_table_;
    torch::Tensor vertex_offset_table_;
    torch::Tensor edge_location_table_;

    torch::Tensor last_inverse_;
    int64_t last_unique_size_ = -1;

    torch::Tensor first_cell_used_;
    torch::Tensor used_to_first_vert_;
    torch::Tensor axis_slot_;
    torch::Tensor used_indices_;

    // Helper functions
    torch::Tensor compute_cube_codes(const torch::Tensor& grid, Scalar iso);
    std::tuple<torch::Tensor, torch::Tensor> build_mesh(
        const torch::Tensor& grid,
        const torch::Tensor& deform,
        const torch::Tensor& cube_codes,
        Scalar iso
    );
    
public:
    explicit MPSMarchingCubesBackend(torch::Device device);
    
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

// Dual Marching Cubes MPS implementation
template <typename Scalar, typename IndexType>
class MPSDualMarchingCubesBackend : public MarchingCubesBackend<Scalar, IndexType> {
private:
    torch::Device device_;
    torch::Tensor edge_table_;
    torch::Tensor quad_table_;
    
public:
    explicit MPSDualMarchingCubesBackend(torch::Device device);
    
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

} // namespace device_abstraction
