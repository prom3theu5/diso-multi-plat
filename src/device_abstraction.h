#pragma once
#include <torch/torch.h>

namespace device_abstraction {

// Device-agnostic vertex structure
template <typename T>
struct Vertex {
    T x, y, z;
    
    Vertex() = default;
    Vertex(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    
    Vertex<T> operator+(const Vertex<T>& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    
    Vertex<T> operator-(const Vertex<T>& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    
    Vertex<T> operator*(T scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
    
    T dot(const Vertex<T>& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
};

// Abstract base class for device-specific implementations
template <typename Scalar, typename IndexType>
class MarchingCubesBackend {
public:
    virtual ~MarchingCubesBackend() = default;
    
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor grid, 
        torch::Tensor deform, 
        Scalar iso
    ) = 0;
    
    virtual void backward(
        torch::Tensor grid,
        torch::Tensor deform,
        Scalar iso,
        torch::Tensor adj_verts,
        torch::Tensor adj_grid,
        torch::Tensor adj_deform
    ) = 0;
};

// Factory function to create appropriate backend
template <typename Scalar, typename IndexType>
std::unique_ptr<MarchingCubesBackend<Scalar, IndexType>> create_backend(torch::Device device);

} // namespace device_abstraction