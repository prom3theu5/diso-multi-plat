#!/usr/bin/env python3
"""Generate a reference mesh using UniversalDiffDMC and validate the topology.

This script is intended as a quick sanity check for the multi-platform backend.
It prefers running on MPS when available, falls back to CPU otherwise, and writes
the resulting mesh to an OBJ so downstream consumers can inspect it easily.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the repository root is on the import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diso import UniversalDiffDMC  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Execution device. 'auto' prefers MPS when available.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Grid resolution for the synthetic SDF volume.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "out",
        help="Directory where the OBJ file will be written.",
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=0.0,
        help="Isovalue used during surface extraction.",
    )
    parser.add_argument(
        "--weld",
        action="store_true",
        help="Merge duplicate vertices before saving to make the mesh manifold-friendly.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")

    if choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS backend requested but not available.")
        return torch.device("mps")

    # Auto selection
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_sphere_sdf(resolution: int, device: torch.device) -> torch.Tensor:
    grid_lin = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=torch.float32)
    coords = torch.meshgrid(grid_lin, grid_lin, grid_lin, indexing="ij")
    sdf = torch.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2) - 0.5
    return sdf.contiguous()


def center_vertices(vertices: torch.Tensor) -> torch.Tensor:
    vert_min = vertices.min(dim=0).values
    vert_max = vertices.max(dim=0).values
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[info] using device: {device}")

    sdf = make_sphere_sdf(args.resolution, device)
    print(
        f"[info] sdf grid ready: shape={tuple(sdf.shape)}, "
        f"min={sdf.min().item():.4f}, max={sdf.max().item():.4f}"
    )

    extractor = UniversalDiffDMC(dtype=torch.float32).to(device)
    print("[info] running marching cubes...")
    verts, faces = extractor(sdf, deform=None, return_quads=False, normalize=True)

    if verts.numel() == 0 or faces.numel() == 0:
        raise RuntimeError("Surface extraction produced an empty mesh.")

    print(
        f"[info] raw mesh: verts={verts.shape[0]}, faces={faces.shape[0]}, "
        f"device={verts.device}"
    )

    verts = center_vertices(verts)

    vertices = verts.detach().cpu().numpy().astype(np.float32)
    faces_np = faces.detach().cpu().numpy()

    if faces_np.max() >= vertices.shape[0] or faces_np.min() < 0:
        raise RuntimeError("Face indices fall outside the vertex buffer bounds.")

    # Trimesh validation ensures the face winding and indexing are sensible.
    import trimesh  # Imported late so the script can fail fast if missing.

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_np, process=True)

    if args.weld:
        # Welding creates shared vertices across triangles which removes the
        # non-manifold edge reports that slicers typically emit for raw MC output.
        mesh.merge_vertices()
    if mesh.is_empty or mesh.faces.shape[0] == 0:
        raise RuntimeError("Trimesh considers the mesh empty or degenerate.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "test_mps_mesh.obj"
    mesh.export(output_path)

    print(f"[info] mesh exported to {output_path}")
    print(
        f"[success] validated mesh: vertices={mesh.vertices.shape[0]}, "
        f"faces={mesh.faces.shape[0]}"
    )


if __name__ == "__main__":
    main()
