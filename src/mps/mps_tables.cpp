#include "mps_tables.h"
#include "../marching_cubes_tables.h"

namespace device_abstraction {

torch::Tensor create_edge_table(torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto tensor = torch::from_blob(
        const_cast<int*>(marching_cubes_tables::kEdgeTable.data()),
        {static_cast<long>(marching_cubes_tables::kEdgeTable.size())},
        options
    ).clone();

    if (device.type() != torch::kCPU) {
        tensor = tensor.to(device);
    }
    return tensor;
}

torch::Tensor create_tri_table(torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto tensor = torch::empty({256, 16}, options);
    auto accessor = tensor.accessor<int32_t, 2>();

    for (int case_idx = 0; case_idx < 256; ++case_idx) {
        int start = marching_cubes_tables::kCaseStart[case_idx];
        int end = marching_cubes_tables::kCaseStart[case_idx + 1];
        int write_idx = 0;
        for (int offset = start; offset < end && write_idx < 16; ++offset, ++write_idx) {
            accessor[case_idx][write_idx] = marching_cubes_tables::kCaseTriEdges[offset];
        }
        for (; write_idx < 16; ++write_idx) {
            accessor[case_idx][write_idx] = -1;
        }
    }

    if (device.type() != torch::kCPU) {
        tensor = tensor.to(device);
    }
    return tensor;
}

torch::Tensor create_edge_connection_table(torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto tensor = torch::from_blob(
        const_cast<int*>(&marching_cubes_tables::kEdgeConnection[0][0]),
        {12, 2},
        options
    ).clone();

    if (device.type() != torch::kCPU) {
        tensor = tensor.to(device);
    }
    return tensor;
}

torch::Tensor create_vertex_offset_table(torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto tensor = torch::from_blob(
        const_cast<int*>(&marching_cubes_tables::kVertexOffset[0][0]),
        {8, 3},
        options
    ).clone();

    if (device.type() != torch::kCPU) {
        tensor = tensor.to(device);
    }
    return tensor;
}

torch::Tensor create_edge_location_table(torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto tensor = torch::from_blob(
        const_cast<int*>(&marching_cubes_tables::kEdgeCanonical[0][0]),
        {12, 4},
        options
    ).clone();

    if (device.type() != torch::kCPU) {
        tensor = tensor.to(device);
    }
    return tensor;
}

} // namespace device_abstraction
