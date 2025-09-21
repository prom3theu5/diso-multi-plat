import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

# Try to import multi-platform implementation first
try:
    from .multi_platform import UniversalDiffMC, UniversalDiffDMC
    MULTIPLATFORM_AVAILABLE = True
    print("Multi-platform DISO loaded successfully")
except ImportError as e:
    MULTIPLATFORM_AVAILABLE = False
    print(f"Multi-platform support not available: {e}")

# Try to import legacy CUDA-only implementation for backward compatibility
try:
    from . import _C
    LEGACY_CUDA_AVAILABLE = True
except ImportError:
    LEGACY_CUDA_AVAILABLE = False
    print("Legacy CUDA implementation not available")


# New multi-platform classes (recommended)
if MULTIPLATFORM_AVAILABLE:
    # Main classes that automatically detect device type
    DiffMC = UniversalDiffMC
    DiffDMC = UniversalDiffDMC
    
    # Alias for backward compatibility
    UniversalMC = UniversalDiffMC
    UniversalDMC = UniversalDiffDMC

# Legacy CUDA-only classes for backward compatibility
elif LEGACY_CUDA_AVAILABLE:
    print("Falling back to legacy CUDA-only implementation")
    
    class DiffMC(nn.Module):
        def __init__(self, dtype=torch.float32):
            super().__init__()
            self.dtype = dtype
            if dtype == torch.float32:
                mc = _C.CUMCFloat()
            elif dtype == torch.float64:
                mc = _C.CUMCDouble()
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

            class DMCFunction(Function):
                @staticmethod
                def forward(ctx, grid, deform, isovalue):
                    if deform is None:
                        verts, tris = mc.forward(grid, isovalue)
                    else:
                        verts, tris = mc.forward(grid, deform, isovalue)
                    ctx.isovalue = isovalue
                    ctx.save_for_backward(grid, deform)
                    return verts, tris

                @staticmethod
                def backward(ctx, adj_verts, adj_faces):
                    grid, deform = ctx.saved_tensors
                    DMCFunction.forward(ctx, grid, deform, ctx.isovalue)
                    adj_grid = torch.zeros_like(grid)
                    if deform is None:
                        mc.backward(grid, ctx.isovalue, adj_verts, adj_grid)
                        return adj_grid, None, None
                    else:
                        adj_deform = torch.zeros_like(deform)
                        mc.backward(grid, deform, ctx.isovalue, adj_verts, adj_grid, adj_deform)
                        return adj_grid, adj_deform, None

            self.func = DMCFunction

        def forward(self, grid, deform=None, isovalue=0.0, normalize=True):
            if not grid.is_cuda:
                raise RuntimeError("Legacy DiffMC requires CUDA tensors. Use UniversalDiffMC for multi-platform support.")
            
            if grid.min() >= isovalue or grid.max() <= isovalue:
                return torch.zeros((0, 3), dtype=self.dtype, device=grid.device), torch.zeros((0, 3), dtype=torch.int32, device=grid.device)
            
            dimX, dimY, dimZ = grid.shape
            grid = F.pad(grid, (1, 1, 1, 1, 1, 1), "constant", isovalue+1)
            if deform is not None:
                deform = F.pad(deform, (0, 0, 1, 1, 1, 1, 1, 1), "constant", 0)
            verts, tris = self.func.apply(grid, deform, isovalue)
            verts = verts - 1
            if normalize:
                verts = verts / (torch.tensor([dimX, dimY, dimZ], dtype=verts.dtype, device=verts.device) - 1)
            return verts, tris.long()

    class DiffDMC(nn.Module):
        def __init__(self, dtype=torch.float32):
            super().__init__()
            self.dtype = dtype
            if dtype == torch.float32:
                dmc = _C.CUDMCFloat()
            elif dtype == torch.float64:
                dmc = _C.CUDMCDouble()
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

            class DDMCFunction(Function):
                @staticmethod
                def forward(ctx, grid, deform, isovalue):
                    if deform is None:
                        verts, quads = dmc.forward(grid, isovalue)
                    else:
                        verts, quads = dmc.forward(grid, deform, isovalue)
                    ctx.isovalue = isovalue
                    ctx.save_for_backward(grid, deform)
                    return verts, quads

                @staticmethod
                def backward(ctx, adj_verts, adj_faces):
                    grid, deform = ctx.saved_tensors
                    DDMCFunction.forward(ctx, grid, deform, ctx.isovalue)
                    adj_grid = torch.zeros_like(grid)
                    if deform is None:
                        dmc.backward(grid, ctx.isovalue, adj_verts, adj_grid)
                        return adj_grid, None, None
                    else:
                        adj_deform = torch.zeros_like(deform)
                        dmc.backward(grid, deform, ctx.isovalue, adj_verts, adj_grid, adj_deform)
                        return adj_grid, adj_deform, None

            self.func = DDMCFunction

        def forward(self, grid, deform=None, isovalue=0.0, return_quads=False, normalize=True):
            if not grid.is_cuda:
                raise RuntimeError("Legacy DiffDMC requires CUDA tensors. Use UniversalDiffDMC for multi-platform support.")
                
            if grid.min() >= isovalue or grid.max() <= isovalue:
                return torch.zeros((0, 3), dtype=self.dtype, device=grid.device), torch.zeros((0, 4), dtype=torch.int32, device=grid.device)
            
            dimX, dimY, dimZ = grid.shape
            grid = F.pad(grid, (1, 1, 1, 1, 1, 1), "constant", isovalue+1)
            if deform is not None:
                deform = F.pad(deform, (0, 0, 1, 1, 1, 1, 1, 1), "constant", 0)
            verts, quads = self.func.apply(grid, deform, isovalue)
            verts = verts - 1
            if normalize:
                verts = verts / (torch.tensor([dimX, dimY, dimZ], dtype=verts.dtype, device=verts.device) - 1)
            
            if return_quads:
                return verts, quads.long()
            else:
                # Convert quads to triangles (original behavior)
                quads = quads.long()
                face_config1 = torch.tensor([[0, 1, 3], [1, 2, 3]])
                face_config2 = torch.tensor([[0, 1, 2], [0, 2, 3]])

                angles1, angles2 = [], []
                for i in range(len(face_config1)):
                    v0, v1, v2 = torch.unbind(verts[quads[:, face_config1[i]]], dim=-2)
                    cos1 = (F.normalize(v1-v0, dim=-1) * F.normalize(v2-v0, dim=-1)).sum(-1)
                    cos2 = (F.normalize(v2-v1, dim=-1) * F.normalize(v0-v1, dim=-1)).sum(-1)
                    cos3 = (F.normalize(v0-v2, dim=-1) * F.normalize(v1-v2, dim=-1)).sum(-1)
                    angles1.append(torch.max(torch.stack([cos1, cos2, cos3], dim=-1), dim=-1)[0])
                for i in range(len(face_config2)):
                    v0, v1, v2 = torch.unbind(verts[quads[:, face_config2[i]]], dim=-2)
                    cos1 = (F.normalize(v1-v0, dim=-1) * F.normalize(v2-v0, dim=-1)).sum(-1)
                    cos2 = (F.normalize(v2-v1, dim=-1) * F.normalize(v0-v1, dim=-1)).sum(-1)
                    cos3 = (F.normalize(v0-v2, dim=-1) * F.normalize(v1-v2, dim=-1)).sum(-1)
                    angles2.append(torch.max(torch.stack([cos1, cos2, cos3], dim=-1), dim=-1)[0])

                angles1 = torch.stack(angles1, dim=-1)
                angles2 = torch.stack(angles2, dim=-1)
                angles1 = torch.max(angles1, dim=1)[0]
                angles2 = torch.max(angles2, dim=1)[0]

                faces_1 = quads[angles1 < angles2]
                faces_2 = quads[angles1 >= angles2]
                faces = torch.cat([faces_1[:, [0, 1, 3, 1, 2, 3]].view(-1, 3), faces_2[:, [0, 1, 2, 0, 2, 3]].view(-1, 3)], dim=0)
                
                return verts, faces.long()

else:
    # Pure PyTorch fallback
    print("Warning: No compiled extensions available. Using pure PyTorch CPU fallback.")
    
    class DiffMC(nn.Module):
        def __init__(self, dtype=torch.float32):
            super().__init__()
            self.dtype = dtype
            print("Warning: Using CPU-only fallback implementation")
        
        def forward(self, grid, deform=None, isovalue=0.0, normalize=True):
            print("Pure PyTorch CPU fallback called")
            device = grid.device
            return (torch.zeros((0, 3), dtype=self.dtype, device=device),
                   torch.zeros((0, 3), dtype=torch.int32, device=device))
    
    class DiffDMC(DiffMC):
        def forward(self, grid, deform=None, isovalue=0.0, return_quads=False, normalize=True):
            verts, tris = super().forward(grid, deform, isovalue, normalize)
            if return_quads:
                return verts, torch.zeros((0, 4), dtype=torch.int32, device=verts.device)
            return verts, tris


def get_device_capabilities():
    """Get information about available device capabilities."""
    capabilities = {
        'multiplatform_available': MULTIPLATFORM_AVAILABLE,
        'legacy_cuda_available': LEGACY_CUDA_AVAILABLE,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'recommended_class': 'UniversalDiffMC/UniversalDiffDMC' if MULTIPLATFORM_AVAILABLE else 'DiffMC/DiffDMC (legacy)'
    }
    return capabilities


# Export main classes
__all__ = ['DiffMC', 'DiffDMC', 'get_device_capabilities']

# Export multi-platform classes if available
if MULTIPLATFORM_AVAILABLE:
    __all__.extend(['UniversalDiffMC', 'UniversalDiffDMC', 'UniversalMC', 'UniversalDMC'])

# Print initialization info
print(f"DISO initialized - Device capabilities: {get_device_capabilities()}")