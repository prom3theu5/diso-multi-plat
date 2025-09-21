#pragma once
#include "../device_abstraction.h"
#include <torch/torch.h>

#ifdef WITH_CUDA
#include "cumc.h"
#include "cudualmc.h"
#endif

namespace device_abstraction {

#ifdef WITH_CUDA
// CUDA wrapper that uses existing CUDA implementations
template <typename Scalar, typename IndexType>
class CUDAMarchingCubesBackend : public MarchingCubesBackend<Scalar, IndexType> {
private:
    cumc::CuMC<Scalar, IndexType> mc_;
    torch::Device device_;
    
public:
    explicit CUDAMarchingCubesBackend(torch::Device device);
    
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

// CUDA Dual Marching Cubes wrapper
template <typename Scalar, typename IndexType>
class CUDADualMarchingCubesBackend : public MarchingCubesBackend<Scalar, IndexType> {
private:
    cudualmc::CUDualMC<Scalar, IndexType> dmc_;
    torch::Device device_;
    
public:
    explicit CUDADualMarchingCubesBackend(torch::Device device);
    
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
#endif // WITH_CUDA

} // namespace device_abstraction