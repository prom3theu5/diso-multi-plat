#include "cpu_backend.h"
#include "../mps/mps_tables.h"
#include <array>
#include <iostream>

namespace device_abstraction {
namespace {
torch::Tensor stable_nonzero_1d(const torch::Tensor& mask, const torch::Device& target_device) {
    TORCH_CHECK(mask.dim() == 1, "stable_nonzero_1d expects a 1D tensor");

    torch::Tensor mask_cpu = mask.device().type() == torch::kCPU
        ? mask
        : mask.to(torch::kCPU);

    auto indices_cpu = torch::nonzero(mask_cpu).squeeze(1);

    if (target_device.type() == torch::kCPU) {
        return indices_cpu;
    }
    return indices_cpu.to(target_device);
}

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
    edge_location_table_ = create_edge_location_table(device_);
    last_unique_size_ = 0;
    std::cout << "Initialized CPU Marching Cubes backend" << std::endl;
}

template <typename Scalar, typename IndexType>
torch::Tensor CPUMarchingCubesBackend<Scalar, IndexType>::compute_cube_codes(
    const torch::Tensor& grid,
    Scalar iso
) {
    auto dims = grid.sizes();
    TORCH_CHECK(dims.size() == 3, "Grid tensor must be 3D");
    int64_t grid_dim0 = dims[0];
    int64_t grid_dim1 = dims[1];
    int64_t grid_dim2 = dims[2];
    int64_t nx = grid_dim0 - 1;
    int64_t ny = grid_dim1 - 1;
    int64_t nz = grid_dim2 - 1;

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
    int64_t grid_dim0 = dims[0];
    int64_t grid_dim1 = dims[1];
    int64_t grid_dim2 = dims[2];
    int64_t nx = grid_dim0 - 1;
    int64_t ny = grid_dim1 - 1;
    int64_t nz = grid_dim2 - 1;

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        auto empty_verts = torch::empty({0, 3}, grid.options().dtype(grid.dtype()).device(device_));
        auto empty_tris = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
        first_cell_used_ = torch::Tensor();
        used_to_first_vert_ = torch::Tensor();
        axis_slot_ = torch::Tensor();
        used_indices_ = torch::Tensor();
        last_inverse_ = torch::Tensor();
        last_unique_size_ = 0;
        return {empty_verts, empty_tris};
    }

    const int64_t num_cubes = nx * ny * nz;
    auto options_float = grid.options();
    auto options_long = torch::TensorOptions().dtype(torch::kLong).device(device_);

    auto base = grid.narrow(0, 0, nx).narrow(1, 0, ny).narrow(2, 0, nz);
    auto neighbor_x = grid.narrow(0, 1, nx).narrow(1, 0, ny).narrow(2, 0, nz);
    auto neighbor_y = grid.narrow(0, 0, nx).narrow(1, 1, ny).narrow(2, 0, nz);
    auto neighbor_z = grid.narrow(0, 0, nx).narrow(1, 0, ny).narrow(2, 1, nz);

    auto iso_tensor = torch::full_like(base, iso);

    auto edge_x_mask = torch::logical_or(
        torch::logical_and(base.lt(iso_tensor), neighbor_x.ge(iso_tensor)),
        torch::logical_and(neighbor_x.lt(iso_tensor), base.ge(iso_tensor))
    );
    auto edge_y_mask = torch::logical_or(
        torch::logical_and(base.lt(iso_tensor), neighbor_y.ge(iso_tensor)),
        torch::logical_and(neighbor_y.lt(iso_tensor), base.ge(iso_tensor))
    );
    auto edge_z_mask = torch::logical_or(
        torch::logical_and(base.lt(iso_tensor), neighbor_z.ge(iso_tensor)),
        torch::logical_and(neighbor_z.lt(iso_tensor), base.ge(iso_tensor))
    );

    auto denom_x = neighbor_x - base;
    auto denom_y = neighbor_y - base;
    auto denom_z = neighbor_z - base;

    auto eps = torch::full_like(denom_x, static_cast<Scalar>(1e-6));

    auto safe_denom_x = torch::where(
        torch::abs(denom_x) < eps,
        torch::where(denom_x.eq(0), torch::ones_like(denom_x), torch::sign(denom_x)) * eps,
        denom_x
    );
    auto safe_denom_y = torch::where(
        torch::abs(denom_y) < eps,
        torch::where(denom_y.eq(0), torch::ones_like(denom_y), torch::sign(denom_y)) * eps,
        denom_y
    );
    auto safe_denom_z = torch::where(
        torch::abs(denom_z) < eps,
        torch::where(denom_z.eq(0), torch::ones_like(denom_z), torch::sign(denom_z)) * eps,
        denom_z
    );

    auto t_x = torch::clamp((iso_tensor - base) / safe_denom_x, static_cast<Scalar>(0), static_cast<Scalar>(1));
    auto t_y = torch::clamp((iso_tensor - base) / safe_denom_y, static_cast<Scalar>(0), static_cast<Scalar>(1));
    auto t_z = torch::clamp((iso_tensor - base) / safe_denom_z, static_cast<Scalar>(0), static_cast<Scalar>(1));

    auto edge_x_mask_flat = edge_x_mask.reshape({-1});
    auto edge_y_mask_flat = edge_y_mask.reshape({-1});
    auto edge_z_mask_flat = edge_z_mask.reshape({-1});

    auto t_x_flat = t_x.reshape({-1});
    auto t_y_flat = t_y.reshape({-1});
    auto t_z_flat = t_z.reshape({-1});

    auto d0_flat = base.reshape({-1});
    auto dx_flat = neighbor_x.reshape({-1});
    auto dy_flat = neighbor_y.reshape({-1});
    auto dz_flat = neighbor_z.reshape({-1});

    auto safe_denom_x_flat = safe_denom_x.reshape({-1});
    auto safe_denom_y_flat = safe_denom_y.reshape({-1});
    auto safe_denom_z_flat = safe_denom_z.reshape({-1});

    auto cube_codes_flat = cube_codes.reshape({-1});
    auto active_mask = cube_codes_flat.ne(0) & cube_codes_flat.ne(255);
    int64_t num_used = active_mask.sum().template item<int64_t>();

    if (num_used == 0) {
        auto empty_verts = torch::empty({0, 3}, options_float);
        auto empty_tris = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
        first_cell_used_ = torch::Tensor();
        used_to_first_vert_ = torch::Tensor();
        axis_slot_ = torch::Tensor();
        used_indices_ = torch::Tensor();
        last_inverse_ = torch::Tensor();
        last_unique_size_ = 0;
        return {empty_verts, empty_tris};
    }

    auto used_indices = stable_nonzero_1d(active_mask, device_);

    auto edge_counts = edge_x_mask_flat.to(torch::kLong) +
                       edge_y_mask_flat.to(torch::kLong) +
                       edge_z_mask_flat.to(torch::kLong);
    auto used_counts = edge_counts.index_select(0, used_indices);

    auto used_to_first = torch::zeros({num_used + 1}, options_long);
    if (num_used > 0) {
        used_to_first.slice(0, 1, num_used + 1).copy_(used_counts.cumsum(0));
    }
    int64_t total_vertices = used_to_first[-1].template item<int64_t>();

    auto first_cell_used = torch::zeros({num_cubes + 1}, options_long);
    if (num_cubes > 0) {
        first_cell_used.slice(0, 1, num_cubes + 1).copy_(active_mask.to(torch::kLong).cumsum(0));
    }
    auto first_cell_used_flat = first_cell_used.slice(0, 0, num_cubes);

    auto verts = torch::empty({total_vertices, 3}, options_float);
    auto axis_slot = torch::full({num_used, 3}, -1, options_long);
    auto slot_start = used_to_first.slice(0, 0, num_used);
    auto current_slot = slot_start.clone();

    auto xs = torch::arange(0, nx, options_long).view({nx, 1, 1}).expand({nx, ny, nz});
    auto ys = torch::arange(0, ny, options_long).view({1, ny, 1}).expand({nx, ny, nz});
    auto zs = torch::arange(0, nz, options_long).view({1, 1, nz}).expand({nx, ny, nz});
    auto base_coords_long = torch::stack({xs, ys, zs}, 3).reshape({num_cubes, 3});
    auto base_coords_used_long = base_coords_long.index_select(0, used_indices);
    auto base_coords_used_float = base_coords_used_long.to(options_float);

    auto cross_x_used = edge_x_mask_flat.index_select(0, used_indices);
    auto cross_y_used = edge_y_mask_flat.index_select(0, used_indices);
    auto cross_z_used = edge_z_mask_flat.index_select(0, used_indices);

    auto t_x_used = t_x_flat.index_select(0, used_indices);
    auto t_y_used = t_y_flat.index_select(0, used_indices);
    auto t_z_used = t_z_flat.index_select(0, used_indices);

    torch::Tensor deform0_used;
    torch::Tensor deform_x1_used;
    torch::Tensor deform_y1_used;
    torch::Tensor deform_z1_used;
    bool has_deform = deform.defined() && deform.numel() > 0;
    if (has_deform) {
        auto deform_cast = deform.to(options_float);
        auto deform0 = deform_cast.narrow(0, 0, nx).narrow(1, 0, ny).narrow(2, 0, nz);
        auto deform_x1 = deform_cast.narrow(0, 1, nx).narrow(1, 0, ny).narrow(2, 0, nz);
        auto deform_y1 = deform_cast.narrow(0, 0, nx).narrow(1, 1, ny).narrow(2, 0, nz);
        auto deform_z1 = deform_cast.narrow(0, 0, nx).narrow(1, 0, ny).narrow(2, 1, nz);

        deform0_used = deform0.reshape({num_cubes, deform.size(3)}).index_select(0, used_indices);
        deform_x1_used = deform_x1.reshape({num_cubes, deform.size(3)}).index_select(0, used_indices);
        deform_y1_used = deform_y1.reshape({num_cubes, deform.size(3)}).index_select(0, used_indices);
        deform_z1_used = deform_z1.reshape({num_cubes, deform.size(3)}).index_select(0, used_indices);
    }

    auto process_axis = [&](const torch::Tensor& axis_mask,
                            const torch::Tensor& t_all,
                            const torch::Tensor& deform0_all,
                            const torch::Tensor& deform1_all,
                            int axis) {
        auto idx_cells = stable_nonzero_1d(axis_mask, device_);
        if (idx_cells.numel() == 0) {
            return;
        }

        auto slots = current_slot.index_select(0, idx_cells);
        axis_slot.index_put_({idx_cells, axis}, slots);

        auto base_coords_l = base_coords_used_long.index_select(0, idx_cells);
        auto base_coords_f = base_coords_used_float.index_select(0, idx_cells);
        auto t_vals = t_all.index_select(0, idx_cells);

        torch::Tensor deform0_vals;
        torch::Tensor deform1_vals;
        if (has_deform) {
            deform0_vals = deform0_all.index_select(0, idx_cells);
            deform1_vals = deform1_all.index_select(0, idx_cells);
        }

        auto zeros = torch::zeros_like(t_vals, options_float.dtype());
        auto ones = torch::ones_like(t_vals, options_float.dtype());

        auto offset_unit = torch::stack({
            axis == 0 ? ones : zeros,
            axis == 1 ? ones : zeros,
            axis == 2 ? ones : zeros
        }, 1);

        auto p0 = base_coords_f;
        if (has_deform) {
            p0 = p0 + deform0_vals;
        }

        auto p1 = base_coords_f + offset_unit;
        if (has_deform) {
            p1 = p1 + deform1_vals;
        }

        auto vertex_pos = p0 + (p1 - p0) * t_vals.unsqueeze(1);
        verts.index_copy_(0, slots, vertex_pos);

        current_slot += axis_mask.to(torch::kLong);
    };

    process_axis(cross_x_used, t_x_used, deform0_used, deform_x1_used, 0);
    process_axis(cross_y_used, t_y_used, deform0_used, deform_y1_used, 1);
    process_axis(cross_z_used, t_z_used, deform0_used, deform_z1_used, 2);

    auto cube_codes_used = cube_codes_flat.index_select(0, used_indices).to(torch::kLong);
    auto tri_edges = tri_table_.index_select(0, cube_codes_used);
    auto valid_mask = tri_edges >= 0;
    auto tri_edges_clamped = tri_edges.clamp_min(0).to(torch::kLong);

    auto stride_x = ny * nz;
    auto stride_y = nz;

    auto edge_offsets_table = edge_location_table_.to(options_long);
    auto edge_offsets = edge_offsets_table.index_select(0, tri_edges_clamped.reshape({-1})).view({num_used, 16, 4});

    auto canonical_x = base_coords_used_long.select(1, 0).unsqueeze(1) + edge_offsets.select(2, 0);
    auto canonical_y = base_coords_used_long.select(1, 1).unsqueeze(1) + edge_offsets.select(2, 1);
    auto canonical_z = base_coords_used_long.select(1, 2).unsqueeze(1) + edge_offsets.select(2, 2);

    auto canonical_index = canonical_x * stride_x + canonical_y * stride_y + canonical_z;
    auto used_index_for_edges = first_cell_used_flat.index_select(0, canonical_index.reshape({-1})).view({num_used, 16});

    auto axis_index = edge_offsets.select(2, 3).to(torch::kLong);
    auto axis_slot_map = axis_slot.index_select(0, used_index_for_edges.reshape({-1})).view({num_used, 16, 3});
    auto axis_slot_safe = axis_slot_map.clamp_min(int64_t{0});
    auto vertex_idx = axis_slot_safe.gather(2, axis_index.unsqueeze(-1)).squeeze(-1);
    vertex_idx = vertex_idx.masked_fill(~valid_mask, 0);

    auto tris = vertex_idx.masked_select(valid_mask).view({-1, 3}).to(torch::kInt32);

    first_cell_used_ = first_cell_used;
    used_to_first_vert_ = used_to_first;
    axis_slot_ = axis_slot;
    used_indices_ = used_indices;

    last_inverse_ = torch::arange(total_vertices, options_long);
    last_unique_size_ = total_vertices;

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

    if (last_inverse_.defined() && last_unique_size_ == adj_verts.size(0)) {
        adj_verts = adj_verts.index_select(0, last_inverse_);
    }

    adj_grid.zero_();
    if (adj_deform.defined()) {
        adj_deform.zero_();
    }

    if (adj_verts.numel() == 0) {
        return;
    }

    TORCH_CHECK(used_to_first_vert_.defined(), "Cached edge slots are missing; run forward before backward");
    TORCH_CHECK(axis_slot_.defined(), "Cached axis mapping is missing; run forward before backward");
    TORCH_CHECK(used_indices_.defined(), "Cached used indices are missing; run forward before backward");

    auto used_to_first = used_to_first_vert_.to(torch::kLong);
    int64_t total_vertices = used_to_first[-1].template item<int64_t>();
    TORCH_CHECK(adj_verts.size(0) == total_vertices,
                "Adjoint vertex count does not match cached canonical slots");

    auto used_indices = used_indices_.to(torch::kLong);
    int64_t num_used = used_indices.size(0);
    if (num_used == 0) {
        return;
    }

    auto axis_slot = axis_slot_.to(torch::kLong);
    TORCH_CHECK(axis_slot.size(0) == num_used && axis_slot.size(1) == 3,
                "Axis slot cache has unexpected shape");

    auto dims = grid.sizes();
    TORCH_CHECK(dims.size() == 3, "Grid tensor must be 3D");

    int64_t grid_dim0 = dims[0];
    int64_t grid_dim1 = dims[1];
    int64_t grid_dim2 = dims[2];

    if (grid_dim0 < 2 || grid_dim1 < 2 || grid_dim2 < 2) {
        return;
    }

    int64_t nx = grid_dim0 - 1;
    int64_t ny = grid_dim1 - 1;
    int64_t nz = grid_dim2 - 1;
    int64_t num_cubes = nx * ny * nz;

    auto options_float = grid.options();
    auto options_long = torch::TensorOptions().dtype(torch::kLong).device(device_);

    auto base = grid.narrow(0, 0, nx).narrow(1, 0, ny).narrow(2, 0, nz);
    auto neighbor_x = grid.narrow(0, 1, nx).narrow(1, 0, ny).narrow(2, 0, nz);
    auto neighbor_y = grid.narrow(0, 0, nx).narrow(1, 1, ny).narrow(2, 0, nz);
    auto neighbor_z = grid.narrow(0, 0, nx).narrow(1, 0, ny).narrow(2, 1, nz);

    auto iso_tensor = torch::full_like(base, iso);

    auto edge_x_mask = torch::logical_or(
        torch::logical_and(base.lt(iso_tensor), neighbor_x.ge(iso_tensor)),
        torch::logical_and(neighbor_x.lt(iso_tensor), base.ge(iso_tensor))
    );
    auto edge_y_mask = torch::logical_or(
        torch::logical_and(base.lt(iso_tensor), neighbor_y.ge(iso_tensor)),
        torch::logical_and(neighbor_y.lt(iso_tensor), base.ge(iso_tensor))
    );
    auto edge_z_mask = torch::logical_or(
        torch::logical_and(base.lt(iso_tensor), neighbor_z.ge(iso_tensor)),
        torch::logical_and(neighbor_z.lt(iso_tensor), base.ge(iso_tensor))
    );

    auto denom_x = neighbor_x - base;
    auto denom_y = neighbor_y - base;
    auto denom_z = neighbor_z - base;

    auto eps = torch::full_like(denom_x, static_cast<Scalar>(1e-6));

    auto safe_denom_x = torch::where(
        torch::abs(denom_x) < eps,
        torch::where(denom_x.eq(0), torch::ones_like(denom_x), torch::sign(denom_x)) * eps,
        denom_x
    );
    auto safe_denom_y = torch::where(
        torch::abs(denom_y) < eps,
        torch::where(denom_y.eq(0), torch::ones_like(denom_y), torch::sign(denom_y)) * eps,
        denom_y
    );
    auto safe_denom_z = torch::where(
        torch::abs(denom_z) < eps,
        torch::where(denom_z.eq(0), torch::ones_like(denom_z), torch::sign(denom_z)) * eps,
        denom_z
    );

    auto t_x = torch::clamp((iso_tensor - base) / safe_denom_x, static_cast<Scalar>(0), static_cast<Scalar>(1));
    auto t_y = torch::clamp((iso_tensor - base) / safe_denom_y, static_cast<Scalar>(0), static_cast<Scalar>(1));
    auto t_z = torch::clamp((iso_tensor - base) / safe_denom_z, static_cast<Scalar>(0), static_cast<Scalar>(1));

    auto edge_x_mask_flat = edge_x_mask.reshape({-1});
    auto edge_y_mask_flat = edge_y_mask.reshape({-1});
    auto edge_z_mask_flat = edge_z_mask.reshape({-1});

    auto t_x_flat = t_x.reshape({-1});
    auto t_y_flat = t_y.reshape({-1});
    auto t_z_flat = t_z.reshape({-1});

    auto d0_flat = base.reshape({-1});
    auto dx_flat = neighbor_x.reshape({-1});
    auto dy_flat = neighbor_y.reshape({-1});
    auto dz_flat = neighbor_z.reshape({-1});

    auto safe_denom_x_flat = safe_denom_x.reshape({-1});
    auto safe_denom_y_flat = safe_denom_y.reshape({-1});
    auto safe_denom_z_flat = safe_denom_z.reshape({-1});

    auto base_coords_long_full = torch::stack({
        torch::arange(0, nx, options_long).view({nx, 1, 1}).expand({nx, ny, nz}),
        torch::arange(0, ny, options_long).view({1, ny, 1}).expand({nx, ny, nz}),
        torch::arange(0, nz, options_long).view({1, 1, nz}).expand({nx, ny, nz})
    }, 3).reshape({num_cubes, 3});

    auto base_coords_used_long = base_coords_long_full.index_select(0, used_indices);
    auto base_coords_used_float = base_coords_used_long.to(options_float);

    auto t_x_used = t_x_flat.index_select(0, used_indices);
    auto t_y_used = t_y_flat.index_select(0, used_indices);
    auto t_z_used = t_z_flat.index_select(0, used_indices);

    auto d0_used = d0_flat.index_select(0, used_indices);
    auto dx_used = dx_flat.index_select(0, used_indices);
    auto dy_used = dy_flat.index_select(0, used_indices);
    auto dz_used = dz_flat.index_select(0, used_indices);

    auto safe_denom_x_used = safe_denom_x_flat.index_select(0, used_indices);
    auto safe_denom_y_used = safe_denom_y_flat.index_select(0, used_indices);
    auto safe_denom_z_used = safe_denom_z_flat.index_select(0, used_indices);

    bool has_deform_offsets = deform.defined() && deform.numel() > 0;
    bool track_deform_grad = has_deform_offsets && adj_deform.defined() && adj_deform.numel() > 0;

    torch::Tensor deform0_used;
    torch::Tensor deform_x1_used;
    torch::Tensor deform_y1_used;
    torch::Tensor deform_z1_used;
    if (has_deform_offsets) {
        auto deform_cast = deform.to(options_float);
        auto deform0 = deform_cast.narrow(0, 0, nx).narrow(1, 0, ny).narrow(2, 0, nz);
        auto deform_x1 = deform_cast.narrow(0, 1, nx).narrow(1, 0, ny).narrow(2, 0, nz);
        auto deform_y1 = deform_cast.narrow(0, 0, nx).narrow(1, 1, ny).narrow(2, 0, nz);
        auto deform_z1 = deform_cast.narrow(0, 0, nx).narrow(1, 0, ny).narrow(2, 1, nz);

        deform0_used = deform0.reshape({num_cubes, deform.size(3)}).index_select(0, used_indices);
        deform_x1_used = deform_x1.reshape({num_cubes, deform.size(3)}).index_select(0, used_indices);
        deform_y1_used = deform_y1.reshape({num_cubes, deform.size(3)}).index_select(0, used_indices);
        deform_z1_used = deform_z1.reshape({num_cubes, deform.size(3)}).index_select(0, used_indices);
    }

    auto adj_grid_flat = adj_grid.view({grid_dim0 * grid_dim1 * grid_dim2});
    torch::Tensor adj_deform_flat;
    int64_t deform_stride_x = 0;
    int64_t deform_stride_y = 0;
    if (track_deform_grad) {
        auto deform_dims = deform.sizes();
        deform_stride_x = deform_dims[1] * deform_dims[2];
        deform_stride_y = deform_dims[2];
        adj_deform_flat = adj_deform.view({deform_dims[0] * deform_dims[1] * deform_dims[2], deform_dims[3]});
    }

    const int64_t grid_stride_x = grid_dim1 * grid_dim2;
    const int64_t grid_stride_y = grid_dim2;

    auto process_axis = [&](int axis,
                            const torch::Tensor& slots_tensor,
                            const torch::Tensor& t_all,
                            const torch::Tensor& d1_all,
                            const torch::Tensor& safe_denom_all,
                            const torch::Tensor& deform1_all) {
        auto axis_mask = slots_tensor.ge(0);
        auto idx_cells = stable_nonzero_1d(axis_mask, device_);
        if (idx_cells.numel() == 0) {
            return;
        }

        auto slots = slots_tensor.index_select(0, idx_cells);
        auto adj_axis = adj_verts.index_select(0, slots).to(options_float.dtype());

        auto base_coords_l = base_coords_used_long.index_select(0, idx_cells);
        auto base_coords_f = base_coords_used_float.index_select(0, idx_cells);
        auto t_vals = t_all.index_select(0, idx_cells);
        auto d0_vals = d0_used.index_select(0, idx_cells);
        auto d1_vals = d1_all.index_select(0, idx_cells);
        auto safe_denom_vals = safe_denom_all.index_select(0, idx_cells);

        torch::Tensor deform0_vals;
        torch::Tensor deform1_vals;
        if (has_deform_offsets) {
            deform0_vals = deform0_used.index_select(0, idx_cells);
            deform1_vals = deform1_all.index_select(0, idx_cells);
        }

        auto zeros = torch::zeros_like(t_vals, options_float.dtype());
        auto ones = torch::ones_like(t_vals, options_float.dtype());

        auto offset_unit = torch::stack({
            axis == 0 ? ones : zeros,
            axis == 1 ? ones : zeros,
            axis == 2 ? ones : zeros
        }, 1);

        auto edge_diff = offset_unit;
        if (has_deform_offsets) {
            edge_diff = edge_diff + (deform1_vals - deform0_vals);
        }

        auto grad_t = (adj_axis * edge_diff).sum(-1);

        auto iso_local = torch::full_like(d0_vals, iso);
        auto raw_t = (iso_local - d0_vals) / safe_denom_vals;
        auto interior_mask = (raw_t > static_cast<Scalar>(0)) & (raw_t < static_cast<Scalar>(1));
        grad_t = grad_t * interior_mask.to(options_float.dtype());

        auto denom_sq = safe_denom_vals * safe_denom_vals;
        auto grad_val0 = grad_t * ((iso_local - d1_vals) / denom_sq);
        auto grad_val1 = grad_t * (-(iso_local - d0_vals) / denom_sq);

        auto base_linear = base_coords_l.select(1, 0) * grid_stride_x +
                           base_coords_l.select(1, 1) * grid_stride_y +
                           base_coords_l.select(1, 2);
        int64_t neighbor_stride = axis == 0 ? grid_stride_x : (axis == 1 ? grid_stride_y : 1);
        auto neighbor_linear = base_linear + neighbor_stride;

        adj_grid_flat.index_add_(0, base_linear, grad_val0);
        adj_grid_flat.index_add_(0, neighbor_linear, grad_val1);

        if (track_deform_grad) {
            auto ones_local = torch::ones_like(t_vals, options_float.dtype());
            auto one_minus_t = (ones_local - t_vals).unsqueeze(1);
            auto t_unsqueezed = t_vals.unsqueeze(1);
            auto adj_axis_deform = adj_axis.to(adj_deform.dtype());

            auto deform_base_linear = base_coords_l.select(1, 0) * deform_stride_x +
                                      base_coords_l.select(1, 1) * deform_stride_y +
                                      base_coords_l.select(1, 2);
            auto deform_neighbor_linear = deform_base_linear + neighbor_stride;

            adj_deform_flat.index_add_(0, deform_base_linear,
                                       adj_axis_deform * one_minus_t);
            adj_deform_flat.index_add_(0, deform_neighbor_linear,
                                       adj_axis_deform * t_unsqueezed);
        }
    };

    process_axis(0, axis_slot.select(1, 0), t_x_used, dx_used, safe_denom_x_used, deform_x1_used);
    process_axis(1, axis_slot.select(1, 1), t_y_used, dy_used, safe_denom_y_used, deform_y1_used);
    process_axis(2, axis_slot.select(1, 2), t_z_used, dz_used, safe_denom_z_used, deform_z1_used);
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
