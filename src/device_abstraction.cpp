#include "device_abstraction.h"
#include "cpu/cpu_backend.h"

#ifdef WITH_MPS
#include "mps/mps_backend.h"
#endif

#ifdef WITH_CUDA
#include "cpu/cuda_backend.h"
#endif

namespace device_abstraction {

template <typename Scalar, typename IndexType>
std::unique_ptr<MarchingCubesBackend<Scalar, IndexType>> create_backend(torch::Device device) {
    std::cout << "create_backend called for device: " << device << std::endl;
    
    if (device.type() == torch::kMPS) {
        std::cout << "MPS device requested" << std::endl;
#ifdef WITH_MPS
        std::cout << "MPS support compiled in" << std::endl;
        try {
            // Check if MPS is actually available at runtime
            if (torch::mps::is_available()) {
                std::cout << "MPS is available, creating MPS backend" << std::endl;
                return std::make_unique<MPSMarchingCubesBackend<Scalar, IndexType>>(device);
            } else {
                std::cerr << "Warning: MPS requested but not available, falling back to CPU" << std::endl;
                return std::make_unique<CPUMarchingCubesBackend<Scalar, IndexType>>();
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: MPS backend creation failed (" << e.what() << "), falling back to CPU" << std::endl;
            return std::make_unique<CPUMarchingCubesBackend<Scalar, IndexType>>();
        }
#else
        std::cerr << "Warning: MPS support was not compiled in, falling back to CPU" << std::endl;
        return std::make_unique<CPUMarchingCubesBackend<Scalar, IndexType>>();
#endif
    }
#ifdef WITH_CUDA
    else if (device.type() == torch::kCUDA) {
        if (torch::cuda::is_available()) {
            return std::make_unique<CUDAMarchingCubesBackend<Scalar, IndexType>>(device);
        } else {
            throw std::runtime_error("CUDA is not available on this system");
        }
    }
#endif
    else if (device.type() == torch::kCPU) {
        std::cout << "CPU device requested, creating CPU backend" << std::endl;
        return std::make_unique<CPUMarchingCubesBackend<Scalar, IndexType>>();
    } else {
        throw std::runtime_error("Unsupported device type: " + device.str());
    }
}

// Explicit template instantiations
template std::unique_ptr<MarchingCubesBackend<float, int>> create_backend<float, int>(torch::Device);
template std::unique_ptr<MarchingCubesBackend<double, int>> create_backend<double, int>(torch::Device);

} // namespace device_abstraction