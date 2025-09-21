#include "cpu_backend.h"
#include "../mps/mps_tables.h"
#include <array>
#include <iostream>

namespace device_abstraction {
namespace {
std::array<torch::Tensor, 8> gather_vertex_values(
    const torch::Tensor& grid,
    int64_t nx,
    int64_t ny,
    int64_t nz
) {
    auto slice_tensor = [&](int dx, int dy, int dz) {
        return grid.slice(0, dx, dx + nx)
                   .slice(1, dy, dy + ny)
                   .slice(2, dz, dz + nz);
    };

    return {
        slice_tensor(0, 0, 0),
        slice_tensor(1, 0, 0),
        slice_tensor(1, 1, 0),
        slice_tensor(0, 1, 0),
        slice_tensor(0, 0, 1),
        slice_tensor(1, 0, 1),
        slice_tensor(1, 1, 1),
        slice_tensor(0, 1, 1)
    };
}
} // namespace

template <typename Scalar, typename IndexType>
CPUMarchingCubesBackend<Scalar, IndexType>::CPUMarchingCubesBackend()
    : device_(torch::kCPU) {
    edge_table_ = create_edge_table(device_);
    tri_table_ = create_tri_table(device_);
    edge_connection_table_ = create_edge_connection_table(device_);
    vertex_offset_table_ = create_vertex_offset_table(device_);
    std::cout << "Initialized CPU Marching Cubes backend" << std::endl;
}

template <typename Scalar, typename IndexType>
torch::Tensor CPUMarchingCubesBackend<Scalar, IndexType>::compute_cube_codes(
    const torch::Tensor& grid,
    Scalar iso
) {
    auto dims = grid.sizes();
    TORCH_CHECK(dims.size() == 3, "Grid tensor must be 3D");
    int64_t nx = dims[0] - 1;
    int64_t ny = dims[1] - 1;
    int64_t nz = dims[2] - 1;

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    auto cube_codes = torch::zeros({nx, ny, nz}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    auto vertex_values = gather_vertex_values(grid, nx, ny, nz);

    for (int i = 0; i < 8; ++i) {
        cube_codes += (vertex_values[i] >= iso).to(torch::kInt32) * (1 << i);
    }

    return cube_codes.reshape({-1}).contiguous();
}

template <typename Scalar, typename IndexType>
std::tuple<torch::Tensor, torch::Tensor> CPUMarchingCubesBackend<Scalar, IndexType>::build_mesh(
    const torch::Tensor& grid,
    const torch::Tensor& deform,
    const torch::Tensor& cube_codes,
    Scalar iso
) {
    auto dims = grid.sizes();
    TORCH_CHECK(dims.size() == 3, "Grid tensor must be 3D");
    int64_t nx = dims[0] - 1;
    int64_t ny = dims[1] - 1;
    int64_t nz = dims[2] - 1;

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        auto empty_verts = torch::empty({0, 3}, grid.options().dtype(grid.dtype()).device(device_));
        auto empty_tris = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
        return {empty_verts, empty_tris};
    }

    const int64_t num_cubes = nx * ny * nz;
    auto grid_dtype = grid.dtype();

    auto vertex_values = gather_vertex_values(grid, nx, ny, nz);
    std::vector<torch::Tensor> vertex_values_flat(8);
    for (int i = 0; i < 8; ++i) {
        vertex_values_flat[i] = vertex_values[i].reshape({num_cubes});
    }
    auto vertex_values_tensor = torch::stack(vertex_values_flat, 0); // [8, num_cubes]

    auto edge_connections = edge_connection_table_.to(torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto v0_idx = edge_connections.select(1, 0);
    auto v1_idx = edge_connections.select(1, 1);

    auto vertex_offsets_tensor = vertex_offset_table_.to(grid.options().dtype(grid_dtype));
    auto vertex_offsets_v0 = vertex_offsets_tensor.index_select(0, v0_idx);
    auto vertex_offsets_v1 = vertex_offsets_tensor.index_select(0, v1_idx);
    auto edge_diff_tensor = (vertex_offsets_v1 - vertex_offsets_v0).to(grid.options().dtype(grid_dtype));

    auto val0 = vertex_values_tensor.index_select(0, v0_idx).transpose(0, 1); // [num_cubes, 12]
    auto val1 = vertex_values_tensor.index_select(0, v1_idx).transpose(0, 1);

    auto denom = val1 - val0;
    auto abs_denom = torch::abs(denom);
    auto eps_tensor = torch::full_like(denom, static_cast<Scalar>(1e-6));
    auto sign = torch::sign(denom);
    auto safe_sign = torch::where(sign == 0, torch::ones_like(sign), sign);
    auto safe_denom = torch::where(abs_denom < eps_tensor, safe_sign * eps_tensor, denom);

    auto raw_t = (static_cast<Scalar>(iso) - val0) / safe_denom;
    auto t = torch::clamp(raw_t, static_cast<Scalar>(0), static_cast<Scalar>(1));

    auto xs = torch::arange(0, nx, grid.options()).view({nx, 1, 1}).expand({nx, ny, nz}).reshape({num_cubes});
    auto ys = torch::arange(0, ny, grid.options()).view({1, ny, 1}).expand({nx, ny, nz}).reshape({num_cubes});
    auto zs = torch::arange(0, nz, grid.options()).view({1, 1, nz}).expand({nx, ny, nz}).reshape({num_cubes});
    auto base_coords = torch::stack({xs, ys, zs}, 1);

    auto edge_pos_tensor = base_coords.unsqueeze(1) +
                           vertex_offsets_v0.unsqueeze(0) +
                           t.unsqueeze(-1) * edge_diff_tensor.unsqueeze(0); // [num_cubes, 12, 3]

    auto cube_codes_long = cube_codes.to(torch::kLong);
    auto tri_edges = tri_table_.index_select(0, cube_codes_long); // [num_cubes, 16]
    auto valid_mask = tri_edges >= 0;
    auto vertex_counts = valid_mask.sum(1, true);
    auto total_vertices = vertex_counts.sum().template item<int64_t>();

    if (total_vertices == 0) {
        auto empty_verts = torch::empty({0, 3}, grid.options().dtype(grid_dtype).device(device_));
        auto empty_tris = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
        return {empty_verts, empty_tris};
    }

    auto tri_edges_clamped = tri_edges.clamp_min(0).to(torch::kLong);
    auto one_hot = torch::one_hot(tri_edges_clamped, 12).to(grid.options().dtype(grid_dtype));
    auto vertex_positions = torch::bmm(one_hot, edge_pos_tensor);
    vertex_positions *= valid_mask.unsqueeze(-1).to(grid.options().dtype(grid_dtype));

    auto vertex_positions_flat = vertex_positions.reshape({-1, 3});
    auto valid_mask_flat = valid_mask.reshape({-1});
    auto valid_indices = torch::nonzero(valid_mask_flat).squeeze(1);

    auto verts = vertex_positions_flat.index_select(0, valid_indices).contiguous();

    if (deform.defined() && verts.size(0) > 0) {
        int64_t dim0 = dims[0];
        int64_t dim1 = dims[1];
        int64_t dim2 = dims[2];

        auto max_coords = torch::tensor({static_cast<Scalar>(dim0 - 1),
                                         static_cast<Scalar>(dim1 - 1),
                                         static_cast<Scalar>(dim2 - 1)},
                                        verts.options());
        auto zeros = torch::zeros({3}, verts.options());
        auto clamped = torch::minimum(torch::maximum(verts, zeros.unsqueeze(0)), max_coords.unsqueeze(0));
        auto indices = clamped.floor().to(torch::kLong);
        auto ix = indices.select(1, 0);
        auto iy = indices.select(1, 1);
        auto iz = indices.select(1, 2);
        auto stride_x = dim1 * dim2;
        auto stride_y = dim2;
        auto linear_indices = ix * stride_x + iy * stride_y + iz;
        auto deform_flat = deform.reshape({dim0 * dim1 * dim2, deform.size(3)});
        auto deform_offsets = deform_flat.index_select(0, linear_indices).to(verts.dtype());
        verts = verts + deform_offsets;
    }

    auto vertex_counts_flat = vertex_counts.reshape({num_cubes});
    auto row_prefix = valid_mask.to(torch::kLong).cumsum(1);
    auto prefix_total = vertex_counts_flat.cumsum(0);
    auto vertex_offsets = torch::zeros_like(vertex_counts_flat);
    if (num_cubes > 1) {
        vertex_offsets.slice(0, 1, num_cubes).copy_(prefix_total.slice(0, 0, num_cubes - 1));
    }
    auto global_indices = row_prefix - 1 + vertex_offsets.view({num_cubes, 1});
    global_indices.masked_fill_(~valid_mask, -1);

    auto global_indices_flat = global_indices.reshape({-1});
    auto used_indices = global_indices_flat.index_select(0, valid_indices);
    auto tris = used_indices.view({-1, 3}).to(torch::kInt32).contiguous();

    return {verts, tris};
}

template <typename Scalar, typename IndexType>
std::tuple<torch::Tensor, torch::Tensor> CPUMarchingCubesBackend<Scalar, IndexType>::forward(
    torch::Tensor grid,
    torch::Tensor deform,
    Scalar iso
) {
    grid = grid.to(device_);
    if (deform.defined()) {
        deform = deform.to(device_);
    }

    auto cube_codes = compute_cube_codes(grid, iso);
    return build_mesh(grid, deform, cube_codes, iso);
}

template <typename Scalar, typename IndexType>
void CPUMarchingCubesBackend<Scalar, IndexType>::backward(
    torch::Tensor grid,
    torch::Tensor deform,
    Scalar iso,
    torch::Tensor adj_verts,
    torch::Tensor adj_grid,
    torch::Tensor adj_deform
) {
    grid = grid.to(device_);
    adj_verts = adj_verts.to(device_);
    adj_grid = adj_grid.to(device_);
    if (deform.defined()) {
        deform = deform.to(device_);
    }
    if (adj_deform.defined()) {
        adj_deform = adj_deform.to(device_);
    }

    adj_grid.zero_();
    if (adj_deform.defined()) {
        adj_deform.zero_();
    }

    if (adj_verts.numel() == 0) {
        return;
    }

    auto dims = grid.sizes();
    TORCH_CHECK(dims.size() == 3, "Grid tensor must be 3D");

    int64_t dim0 = dims[0];
    int64_t dim1 = dims[1];
    int64_t dim2 = dims[2];

    if (dim0 < 2 || dim1 < 2 || dim2 < 2) {
        return;
    }

    int64_t nx = dim0 - 1;
    int64_t ny = dim1 - 1;
    int64_t nz = dim2 - 1;

    auto cube_codes = compute_cube_codes(grid, iso);
    auto num_cubes = cube_codes.size(0);

    if (num_cubes == 0) {
        return;
    }

    auto grid_dtype = grid.dtype();
    auto options = grid.options();

    auto cube_codes_long = cube_codes.to(torch::kLong);
    auto tri_edges = tri_table_.index_select(0, cube_codes_long); // [num_cubes, 16]
    auto valid_mask = tri_edges >= 0;
    auto vertex_counts = valid_mask.sum(1, true);
    auto total_vertices = vertex_counts.sum().template item<int64_t>();

    if (total_vertices == 0) {
        return;
    }

    auto valid_mask_flat = valid_mask.reshape({-1});
    auto tri_edges_clamped = tri_edges.clamp_min(0).to(torch::kLong);

    auto vertex_values = gather_vertex_values(grid, nx, ny, nz);
    std::vector<torch::Tensor> vertex_values_flat(8);
    for (int i = 0; i < 8; ++i) {
        vertex_values_flat[i] = vertex_values[i].reshape({num_cubes});
    }
    auto vertex_values_tensor = torch::stack(vertex_values_flat, 0); // [8, num_cubes]

    auto edge_connections = edge_connection_table_.to(torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto v0_idx = edge_connections.select(1, 0);
    auto v1_idx = edge_connections.select(1, 1);

    auto vertex_offsets_tensor = vertex_offset_table_.to(options.dtype(grid_dtype));
    auto vertex_offsets_v0 = vertex_offsets_tensor.index_select(0, v0_idx);
    auto vertex_offsets_v1 = vertex_offsets_tensor.index_select(0, v1_idx);
    auto edge_diff_tensor = (vertex_offsets_v1 - vertex_offsets_v0).to(options.dtype(grid_dtype));

    auto val0 = vertex_values_tensor.index_select(0, v0_idx).transpose(0, 1);
    auto val1 = vertex_values_tensor.index_select(0, v1_idx).transpose(0, 1);

    auto denom = val1 - val0;
    auto abs_denom = torch::abs(denom);
    auto eps_tensor = torch::full_like(denom, static_cast<Scalar>(1e-6));
    auto sign = torch::sign(denom);
    auto safe_sign = torch::where(sign == 0, torch::ones_like(sign), sign);
    auto safe_denom = torch::where(abs_denom < eps_tensor, safe_sign * eps_tensor, denom);

    auto raw_t = (static_cast<Scalar>(iso) - val0) / safe_denom;
    auto t = torch::clamp(raw_t, static_cast<Scalar>(0), static_cast<Scalar>(1));
    auto interior_mask = (raw_t > static_cast<Scalar>(0)) & (raw_t < static_cast<Scalar>(1));

    auto xs = torch::arange(0, nx, options).view({nx, 1, 1}).expand({nx, ny, nz}).reshape({num_cubes});
    auto ys = torch::arange(0, ny, options).view({1, ny, 1}).expand({nx, ny, nz}).reshape({num_cubes});
    auto zs = torch::arange(0, nz, options).view({1, 1, nz}).expand({nx, ny, nz}).reshape({num_cubes});
    auto base_coords = torch::stack({xs, ys, zs}, 1);

    auto edge_pos_tensor = base_coords.unsqueeze(1) +
                           vertex_offsets_v0.unsqueeze(0) +
                           t.unsqueeze(-1) * edge_diff_tensor.unsqueeze(0);

    auto cube_index_grid = torch::arange(num_cubes, torch::TensorOptions().dtype(torch::kLong).device(device_))
                               .view({num_cubes, 1})
                               .expand_as(tri_edges_clamped);
    auto cube_indices_valid = cube_index_grid.masked_select(valid_mask);
    auto edge_indices_valid = tri_edges_clamped.masked_select(valid_mask);
    auto edge_linear_indices = cube_indices_valid * 12 + edge_indices_valid;

    auto edge_pos_flat = edge_pos_tensor.reshape({num_cubes * 12, 3});
    auto verts_pre_deform = edge_pos_flat.index_select(0, edge_linear_indices);

    auto grad_edge_pos_flat = torch::zeros({num_cubes * 12, 3}, options.dtype(grid_dtype));
    grad_edge_pos_flat.index_add_(0, edge_linear_indices, adj_verts.to(grad_edge_pos_flat.dtype()));
    auto grad_edge_pos = grad_edge_pos_flat.view({num_cubes, 12, 3});

    auto grad_t = (grad_edge_pos * edge_diff_tensor.unsqueeze(0)).sum(-1);
    grad_t = grad_t * interior_mask.to(options.dtype(grid_dtype));

    auto denom_sq = safe_denom * safe_denom;
    auto grad_val0 = grad_t * ((static_cast<Scalar>(iso) - val1) / denom_sq);
    auto grad_val1 = grad_t * (-(static_cast<Scalar>(iso) - val0) / denom_sq);

    auto grad_vertex_values = torch::zeros({8, num_cubes}, options.dtype(grid_dtype));
    auto v0_idx_cpu = v0_idx.to(torch::kCPU);
    auto v1_idx_cpu = v1_idx.to(torch::kCPU);
    auto v0_ptr = v0_idx_cpu.template data_ptr<int64_t>();
    auto v1_ptr = v1_idx_cpu.template data_ptr<int64_t>();

    for (int edge = 0; edge < 12; ++edge) {
        int64_t idx0 = v0_ptr[edge];
        int64_t idx1 = v1_ptr[edge];
        grad_vertex_values.select(0, idx0).add_(grad_val0.select(1, edge));
        grad_vertex_values.select(0, idx1).add_(grad_val1.select(1, edge));
    }

    const int vertex_offsets_arr[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };

    for (int i = 0; i < 8; ++i) {
        auto grad_vals = grad_vertex_values.select(0, i).view({nx, ny, nz});
        adj_grid.slice(0, vertex_offsets_arr[i][0], vertex_offsets_arr[i][0] + nx)
                .slice(1, vertex_offsets_arr[i][1], vertex_offsets_arr[i][1] + ny)
                .slice(2, vertex_offsets_arr[i][2], vertex_offsets_arr[i][2] + nz)
                .add_(grad_vals);
    }

    if (deform.defined() && adj_deform.defined() && adj_verts.size(0) > 0) {
        auto deform_dims = deform.sizes();
        TORCH_CHECK(deform_dims.size() == 4, "Deform tensor must be 4D");

        auto deform_size0 = deform_dims[0];
        auto deform_size1 = deform_dims[1];
        auto deform_size2 = deform_dims[2];

        auto zeros = torch::zeros({3}, verts_pre_deform.options());
        auto max_coords = torch::tensor({static_cast<Scalar>(deform_size0 - 1),
                                         static_cast<Scalar>(deform_size1 - 1),
                                         static_cast<Scalar>(deform_size2 - 1)},
                                        verts_pre_deform.options());

        auto clamped = torch::minimum(torch::maximum(verts_pre_deform, zeros.unsqueeze(0)), max_coords.unsqueeze(0));
        auto indices = clamped.floor().to(torch::kLong);
        auto ix = indices.select(1, 0);
        auto iy = indices.select(1, 1);
        auto iz = indices.select(1, 2);

        auto stride_x = deform_size1 * deform_size2;
        auto stride_y = deform_size2;
        auto linear_indices = ix * stride_x + iy * stride_y + iz;

        auto adj_deform_flat = adj_deform.view({deform_size0 * deform_size1 * deform_size2, deform.size(3)});
        adj_deform_flat.index_add_(0, linear_indices, adj_verts.to(adj_deform.dtype()));
    }
}

template <typename Scalar, typename IndexType>
std::tuple<torch::Tensor, torch::Tensor> CPUDualMarchingCubesBackend<Scalar, IndexType>::forward(
    torch::Tensor grid,
    torch::Tensor deform,
    Scalar iso
) {
    auto device = grid.device();
    auto verts = torch::empty({0, 3}, grid.options().dtype(grid.dtype()).device(device));
    auto quads = torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    return {verts, quads};
}

template <typename Scalar, typename IndexType>
void CPUDualMarchingCubesBackend<Scalar, IndexType>::backward(
    torch::Tensor,
    torch::Tensor,
    Scalar,
    torch::Tensor,
    torch::Tensor adj_grid,
    torch::Tensor adj_deform
) {
    if (adj_grid.defined()) {
        adj_grid.zero_();
    }
    if (adj_deform.defined()) {
        adj_deform.zero_();
    }
}

// Explicit template instantiations
template class CPUMarchingCubesBackend<float, int>;
template class CPUMarchingCubesBackend<double, int>;
template class CPUDualMarchingCubesBackend<float, int>;
template class CPUDualMarchingCubesBackend<double, int>;

} // namespace device_abstraction
