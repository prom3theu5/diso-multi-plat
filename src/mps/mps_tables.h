#pragma once
#include <torch/torch.h>

namespace device_abstraction {

// Create PyTorch tensors from the lookup tables
torch::Tensor create_edge_table(torch::Device device);
torch::Tensor create_tri_table(torch::Device device);
torch::Tensor create_edge_connection_table(torch::Device device);
torch::Tensor create_vertex_offset_table(torch::Device device);
torch::Tensor create_edge_location_table(torch::Device device);

} // namespace device_abstraction
