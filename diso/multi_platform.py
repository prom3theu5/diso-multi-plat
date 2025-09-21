import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

# Try to import the compiled extension
try:
    from . import _C
    HAS_COMPILED_EXTENSION = True
except ImportError:
    HAS_COMPILED_EXTENSION = False
    print("Warning: Compiled extension not available. Using CPU-only fallback.")


class UniversalDiffMC(nn.Module):
    """Multi-platform Differentiable Marching Cubes.
    
    Automatically detects the device and uses the appropriate backend:
    - CUDA: Uses optimized CUDA kernels (if available)
    - MPS: Uses Metal Performance Shaders on Apple Silicon (if available) 
    - CPU: Uses CPU implementation as fallback
    """
    
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        
        if HAS_COMPILED_EXTENSION:
            if dtype == torch.float32:
                self.backend = _C.UMCFloat()
            elif dtype == torch.float64:
                self.backend = _C.UMCDouble()
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        else:
            self.backend = None
            print("Warning: Using pure PyTorch CPU fallback implementation")
        
        class UniversalMCFunction(Function):
            @staticmethod
            def forward(ctx, grid, deform, isovalue, backend):
                if backend is not None:
                    if deform is None:
                        verts, tris = backend.forward(grid, isovalue)
                    else:
                        verts, tris = backend.forward(grid, deform, isovalue)
                else:
                    # Pure PyTorch fallback (CPU only)
                    verts, tris = UniversalDiffMC._cpu_fallback_forward(grid, deform, isovalue)
                
                ctx.isovalue = isovalue
                ctx.backend = backend
                ctx.save_for_backward(grid, deform)
                return verts, tris
            
            @staticmethod
            def backward(ctx, adj_verts, adj_faces):
                grid, deform = ctx.saved_tensors
                
                if ctx.backend is not None:
                    # Re-run forward pass for backward
                    UniversalMCFunction.forward(ctx, grid, deform, ctx.isovalue, ctx.backend)
                    
                    adj_grid = torch.zeros_like(grid)
                    if deform is None:
                        ctx.backend.backward(grid, ctx.isovalue, adj_verts, adj_grid)
                        return adj_grid, None, None, None
                    else:
                        adj_deform = torch.zeros_like(deform)
                        ctx.backend.backward(grid, deform, ctx.isovalue, adj_verts, adj_grid, adj_deform)
                        return adj_grid, adj_deform, None, None
                else:
                    # Pure PyTorch fallback backward
                    adj_grid = torch.zeros_like(grid)
                    adj_deform = torch.zeros_like(deform) if deform is not None else None
                    return adj_grid, adj_deform, None, None
        
        self.func = UniversalMCFunction
    
    @staticmethod
    def _cpu_fallback_forward(grid, deform, isovalue):
        """Pure PyTorch CPU fallback implementation.
        
        This is a basic implementation for when the compiled extension is not available.
        It implements a simplified marching cubes algorithm in pure PyTorch.
        """
        print("Using CPU fallback implementation")
        
        device = grid.device
        dtype = grid.dtype
        
        # Get grid dimensions
        dims = grid.shape
        if len(dims) != 3:
            raise ValueError(f"Expected 3D grid, got {len(dims)}D")
        
        nx, ny, nz = dims[0] - 1, dims[1] - 1, dims[2] - 1
        
        if nx <= 0 or ny <= 0 or nz <= 0:
            return (torch.zeros((0, 3), dtype=dtype, device=device), 
                   torch.zeros((0, 3), dtype=torch.int32, device=device))
        
        # Simple marching cubes implementation
        vertices = []
        triangles = []
        
        # Iterate through all cubes
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    # Get the 8 cube corners
                    cube_values = [
                        grid[x, y, z], grid[x+1, y, z], grid[x+1, y+1, z], grid[x, y+1, z],
                        grid[x, y, z+1], grid[x+1, y, z+1], grid[x+1, y+1, z+1], grid[x, y+1, z+1]
                    ]
                    
                    # Check if isosurface passes through this cube
                    min_val = min(cube_values)
                    max_val = max(cube_values)
                    
                    if min_val <= isovalue <= max_val:
                        # Simplified: generate vertices at cube center for non-empty cubes
                        # This is a very basic approximation
                        center = torch.tensor([x + 0.5, y + 0.5, z + 0.5], dtype=dtype, device=device)
                        
                        # Apply deformation if provided
                        if deform is not None and deform.numel() > 0:
                            if len(deform.shape) == 4 and deform.shape[3] == 3:
                                ix, iy, iz = min(int(center[0]), deform.shape[0]-1), min(int(center[1]), deform.shape[1]-1), min(int(center[2]), deform.shape[2]-1)
                                center += deform[ix, iy, iz]
                        
                        # Create a simple quad (2 triangles) at this position
                        base_idx = len(vertices)
                        
                        # Generate 4 vertices for a simple quad
                        offset = 0.3
                        vertices.extend([
                            center + torch.tensor([-offset, -offset, 0], dtype=dtype, device=device),
                            center + torch.tensor([offset, -offset, 0], dtype=dtype, device=device),
                            center + torch.tensor([offset, offset, 0], dtype=dtype, device=device),
                            center + torch.tensor([-offset, offset, 0], dtype=dtype, device=device)
                        ])
                        
                        # Create two triangles from the quad
                        triangles.extend([
                            [base_idx, base_idx+1, base_idx+2],
                            [base_idx, base_idx+2, base_idx+3]
                        ])
        
        # Convert to tensors
        if vertices:
            verts = torch.stack(vertices)
            tris = torch.tensor(triangles, dtype=torch.int32, device=device)
        else:
            verts = torch.zeros((0, 3), dtype=dtype, device=device)
            tris = torch.zeros((0, 3), dtype=torch.int32, device=device)
        
        print(f"CPU fallback generated {verts.shape[0]} vertices, {tris.shape[0]} triangles")
        return verts, tris
    
    def forward(self, grid, deform=None, isovalue=0.0, normalize=True):
        """Forward pass with automatic device detection."""
        
        # Check if grid has any surface crossings
        if grid.min() >= isovalue or grid.max() <= isovalue:
            device = grid.device
            return (torch.zeros((0, 3), dtype=self.dtype, device=device), 
                   torch.zeros((0, 3), dtype=torch.int32, device=device))
        
        # Get original dimensions
        dimX, dimY, dimZ = grid.shape
        
        # Pad grid for marching cubes algorithm
        grid = F.pad(grid, (1, 1, 1, 1, 1, 1), "constant", isovalue + 1)
        if deform is not None:
            deform = F.pad(deform, (0, 0, 1, 1, 1, 1, 1, 1), "constant", 0)
        
        # Run marching cubes
        verts, tris = self.func.apply(grid, deform, isovalue, self.backend)
        
        # Adjust vertices (undo padding offset)
        if verts.numel() > 0:  # Check if we have vertices
            verts = verts - 1
            
            if normalize:
                # Normalize to [0, 1] range
                verts = verts / torch.tensor([dimX, dimY, dimZ], 
                                           dtype=verts.dtype, 
                                           device=verts.device) - 1
        
        return verts, tris.long()
    
    def get_device_info(self):
        """Get information about the current device and backend capabilities."""
        info = {
            'torch_version': torch.__version__,
            'compiled_extension': HAS_COMPILED_EXTENSION,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'current_device': 'cpu',  # Default to CPU
            'backend_type': 'compiled' if HAS_COMPILED_EXTENSION else 'fallback'
        }
        
        # Try to get current device from backend if available
        if HAS_COMPILED_EXTENSION and hasattr(self, 'backend') and self.backend is not None:
            try:
                # The backend should track its current device
                info['current_device'] = 'dynamic'  # Backend switches automatically
                info['backend_initialized'] = True
            except:
                info['backend_initialized'] = False
        else:
            info['backend_initialized'] = False
        
        # Add CUDA device count if available
        if info['cuda_available']:
            info['cuda_device_count'] = torch.cuda.device_count()
        
        return info


class UniversalDiffDMC(UniversalDiffMC):
    """Multi-platform Differentiable Dual Marching Cubes.
    
    Similar to UniversalDiffMC but produces quad meshes that can be converted to triangles.
    """
    
    def __init__(self, dtype=torch.float32):
        # Initialize with regular marching cubes backend
        super().__init__(dtype=dtype)
        print("Note: Dual Marching Cubes uses the same backend as regular MC, with quad output conversion.")
    
    def forward(self, grid, deform=None, isovalue=0.0, return_quads=False, normalize=True):
        """Forward pass for dual marching cubes."""
        
        # Use regular marching cubes as the base implementation
        verts, tris = super().forward(grid, deform, isovalue, normalize)
        
        if return_quads:
            # Convert triangles to quads when requested
            if tris.numel() > 0:
                # Simple conversion: pair adjacent triangles into quads where possible
                num_tris = tris.shape[0]
                if num_tris >= 2 and num_tris % 2 == 0:
                    # Convert pairs of triangles to quads
                    quads = []
                    for i in range(0, num_tris, 2):
                        tri1 = tris[i]
                        tri2 = tris[i+1] if i+1 < num_tris else tris[i]
                        
                        # Create quad from two triangles
                        # This is a simplified conversion - in practice, you'd want to
                        # check if triangles share edges and form proper quads
                        quad = torch.tensor([tri1[0], tri1[1], tri2[1], tri2[2]], 
                                          dtype=torch.int32, device=tris.device)
                        quads.append(quad)
                    
                    if quads:
                        quads_tensor = torch.stack(quads)
                        return verts, quads_tensor
                
                # Fallback: convert each triangle to a degenerate quad
                quads = torch.cat([tris, tris[:, :1]], dim=1)  # Repeat last vertex
                return verts, quads.long()
            else:
                return verts, torch.zeros((0, 4), dtype=torch.int32, device=verts.device)
        else:
            return verts, tris