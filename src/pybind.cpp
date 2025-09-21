#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "device_abstraction.h"

// Conditionally include CUDA headers only if CUDA is available
#ifdef WITH_CUDA
#include "cuda/cumc.h"
#include "cuda/cudualmc.h"
#endif

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda() || x.is_cpu() || (x.device().type() == torch::kMPS), #x " must be a CUDA, CPU, or MPS tensor")

namespace py = pybind11;

// Device-agnostic wrapper class for Marching Cubes
template <typename Scalar, typename IndexType>
class UniversalMarchingCubes {
private:
    std::unique_ptr<device_abstraction::MarchingCubesBackend<Scalar, IndexType>> backend_;
    torch::Device current_device_;

public:
    UniversalMarchingCubes() : current_device_(torch::kCPU) {
        backend_ = device_abstraction::create_backend<Scalar, IndexType>(current_device_);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid, torch::Tensor deform, Scalar iso) {
        CHECK_INPUT(grid);
        if (deform.defined()) {
            CHECK_INPUT(deform);
        }

        // Switch backend if device changed
        if (grid.device() != current_device_) {
            std::cout << "Device switch detected: " << current_device_ << " -> " << grid.device() << std::endl;
            current_device_ = grid.device();
            try {
                std::cout << "Attempting to create backend for device: " << current_device_ << std::endl;
                backend_ = device_abstraction::create_backend<Scalar, IndexType>(current_device_);
                std::cout << "Successfully created backend for device: " << current_device_ << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to create backend for device " << current_device_ 
                         << " (" << e.what() << "), falling back to CPU" << std::endl;
                current_device_ = torch::kCPU;
                backend_ = device_abstraction::create_backend<Scalar, IndexType>(current_device_);
                
                // Move tensor to CPU if backend creation failed
                grid = grid.to(torch::kCPU);
                if (deform.defined()) {
                    deform = deform.to(torch::kCPU);
                }
            }
        }

        return backend_->forward(grid, deform, iso);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid, Scalar iso) {
        return forward(grid, torch::Tensor{}, iso);
    }

    void backward(torch::Tensor grid, torch::Tensor deform, Scalar iso, 
                  torch::Tensor adj_verts, torch::Tensor adj_grid, torch::Tensor adj_deform) {
        CHECK_INPUT(grid);
        if (deform.defined()) {
            CHECK_INPUT(deform);
        }
        CHECK_INPUT(adj_verts);
        CHECK_INPUT(adj_grid);
        if (adj_deform.defined()) {
            CHECK_INPUT(adj_deform);
        }

        backend_->backward(grid, deform, iso, adj_verts, adj_grid, adj_deform);
    }

    void backward(torch::Tensor grid, Scalar iso, torch::Tensor adj_verts, torch::Tensor adj_grid) {
        backward(grid, torch::Tensor{}, iso, adj_verts, adj_grid, torch::Tensor{});
    }
};

// Legacy CUDA-only classes for backward compatibility
#ifdef WITH_CUDA
namespace cumc {
    template <typename Scalar, typename IndexType>
    class CUMC {
    private:
        cumc::CuMC<Scalar, IndexType> mc;

    public:
        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid, torch::Tensor deform, Scalar iso) {
            TORCH_CHECK(grid.is_cuda(), "grid must be a CUDA tensor for CUMC");
            if (deform.defined()) {
                TORCH_CHECK(deform.is_cuda(), "deform must be a CUDA tensor for CUMC");
            }

            torch::ScalarType scalarType = grid.dtype();
            torch::ScalarType indexType = torch::kInt;

            IndexType dimX = grid.size(0);
            IndexType dimY = grid.size(1);
            IndexType dimZ = grid.size(2);

            mc.forward(grid.data_ptr<Scalar>(), 
                      deform.defined() ? reinterpret_cast<cumc::Vertex<Scalar>*>(deform.data_ptr<Scalar>()) : nullptr,
                      dimX, dimY, dimZ, iso, grid.device().index());

            auto verts = torch::from_blob(mc.verts, torch::IntArrayRef{mc.n_verts, 3}, grid.options().dtype(scalarType)).clone();
            auto tris = torch::from_blob(mc.tris, torch::IntArrayRef{mc.n_tris, 3}, grid.options().dtype(indexType)).clone();

            return {verts, tris};
        }

        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid, Scalar iso) {
            return forward(grid, torch::Tensor{}, iso);
        }

        void backward(torch::Tensor grid, torch::Tensor deform, Scalar iso, torch::Tensor adj_verts,
                      torch::Tensor adj_grid, torch::Tensor adj_deform) {
            TORCH_CHECK(grid.is_cuda(), "grid must be a CUDA tensor for CUMC");
            
            IndexType dimX = grid.size(0);
            IndexType dimY = grid.size(1);
            IndexType dimZ = grid.size(2);

            mc.backward(grid.data_ptr<Scalar>(),
                       deform.defined() ? reinterpret_cast<cumc::Vertex<Scalar>*>(deform.data_ptr<Scalar>()) : nullptr,
                       dimX, dimY, dimZ, iso,
                       adj_grid.data_ptr<Scalar>(),
                       adj_deform.defined() ? reinterpret_cast<cumc::Vertex<Scalar>*>(adj_deform.data_ptr<Scalar>()) : nullptr,
                       reinterpret_cast<cumc::Vertex<Scalar>*>(adj_verts.data_ptr<Scalar>()),
                       grid.device().index());
        }

        void backward(torch::Tensor grid, Scalar iso, torch::Tensor adj_verts, torch::Tensor adj_grid) {
            backward(grid, torch::Tensor{}, iso, adj_verts, adj_grid, torch::Tensor{});
        }
    };
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Universal (multi-device) classes
    py::class_<UniversalMarchingCubes<double, int>>(m, "UMCDouble")
        .def(py::init<>())
        .def("forward", py::overload_cast<torch::Tensor, torch::Tensor, double>(&UniversalMarchingCubes<double, int>::forward))
        .def("backward", py::overload_cast<torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor>(&UniversalMarchingCubes<double, int>::backward))
        .def("forward", py::overload_cast<torch::Tensor, double>(&UniversalMarchingCubes<double, int>::forward))
        .def("backward", py::overload_cast<torch::Tensor, double, torch::Tensor, torch::Tensor>(&UniversalMarchingCubes<double, int>::backward));

    py::class_<UniversalMarchingCubes<float, int>>(m, "UMCFloat")
        .def(py::init<>())
        .def("forward", py::overload_cast<torch::Tensor, torch::Tensor, float>(&UniversalMarchingCubes<float, int>::forward))
        .def("backward", py::overload_cast<torch::Tensor, torch::Tensor, float, torch::Tensor, torch::Tensor, torch::Tensor>(&UniversalMarchingCubes<float, int>::backward))
        .def("forward", py::overload_cast<torch::Tensor, float>(&UniversalMarchingCubes<float, int>::forward))
        .def("backward", py::overload_cast<torch::Tensor, float, torch::Tensor, torch::Tensor>(&UniversalMarchingCubes<float, int>::backward));

#ifdef WITH_CUDA
    // Legacy CUDA-only classes for backward compatibility
    py::class_<cumc::CUMC<double, int>>(m, "CUMCDouble")
        .def(py::init<>())
        .def("forward", py::overload_cast<torch::Tensor, torch::Tensor, double>(&cumc::CUMC<double, int>::forward))
        .def("backward", py::overload_cast<torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor>(&cumc::CUMC<double, int>::backward))
        .def("forward", py::overload_cast<torch::Tensor, double>(&cumc::CUMC<double, int>::forward))
        .def("backward", py::overload_cast<torch::Tensor, double, torch::Tensor, torch::Tensor>(&cumc::CUMC<double, int>::backward));

    py::class_<cumc::CUMC<float, int>>(m, "CUMCFloat")
        .def(py::init<>())
        .def("forward", py::overload_cast<torch::Tensor, torch::Tensor, float>(&cumc::CUMC<float, int>::forward))
        .def("backward", py::overload_cast<torch::Tensor, torch::Tensor, float, torch::Tensor, torch::Tensor, torch::Tensor>(&cumc::CUMC<float, int>::backward))
        .def("forward", py::overload_cast<torch::Tensor, float>(&cumc::CUMC<float, int>::forward))
        .def("backward", py::overload_cast<torch::Tensor, float, torch::Tensor, torch::Tensor>(&cumc::CUMC<float, int>::backward));
#endif
}