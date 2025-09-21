#!/usr/bin/env python3
"""
Test script for multi-platform DISO implementation.
This tests the automatic device detection and backend switching.
"""

import torch
import sys
import os

# Add parent directory to path so we can import the diso modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_device_support():
    """Test which devices are available."""
    print("=== Device Support Test ===")
    
    devices_to_test = [
        torch.device('cpu'),
    ]
    
    # Add CUDA if available
    if torch.cuda.is_available():
        devices_to_test.append(torch.device('cuda:0'))
        print("✓ CUDA is available")
    else:
        print("✗ CUDA not available")
    
    # Add MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices_to_test.append(torch.device('mps'))
        print("✓ MPS is available")
    else:
        print("✗ MPS not available")
    
    return devices_to_test

def create_sphere_sdf(size=32, center=None, radius=None):
    """Create a simple sphere SDF for testing."""
    if center is None:
        center = [size//2, size//2, size//2]
    if radius is None:
        radius = size // 4
    
    # Create coordinate grid
    x = torch.arange(size, dtype=torch.float32)
    y = torch.arange(size, dtype=torch.float32)
    z = torch.arange(size, dtype=torch.float32)
    
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    # Compute distance from center
    dist = torch.sqrt((xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2)
    
    # SDF: negative inside, positive outside
    sdf = dist - radius
    
    return sdf

def test_multiplatform_mc():
    """Test the multi-platform marching cubes implementation."""
    print("\n=== Multi-Platform Marching Cubes Test ===")
    
    try:
        from diso.multi_platform import UniversalDiffMC, UniversalDiffDMC
        print("✓ Multi-platform modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import multi-platform modules: {e}")
        return False
    
    # Test different devices
    devices_to_test = test_device_support()
    
    for device in devices_to_test:
        print(f"\n--- Testing device: {device} ---")
        
        try:
            # Create test data
            sdf = create_sphere_sdf(16)  # Small size for quick testing
            sdf = sdf.to(device)
            
            print(f"✓ Test SDF created on {device}: {sdf.shape}")
            print(f"  SDF range: [{sdf.min():.3f}, {sdf.max():.3f}]")
            
            # Test Marching Cubes
            mc = UniversalDiffMC(torch.float32)
            verts, tris = mc(sdf, isovalue=0.0)
            
            print(f"✓ Marching Cubes: {verts.shape[0]} vertices, {tris.shape[0]} triangles")
            print(f"  Output device: {verts.device}")
            
            # Test Dual Marching Cubes
            dmc = UniversalDiffDMC(torch.float32)
            verts2, faces2 = dmc(sdf, isovalue=0.0, return_quads=False)
            
            print(f"✓ Dual Marching Cubes: {verts2.shape[0]} vertices, {faces2.shape[0]} faces")
            print(f"  Output device: {verts2.device}")
            
            # Test device info
            info = mc.get_device_info()
            print(f"✓ Device capabilities: {info}")
            
        except Exception as e:
            print(f"✗ Error testing device {device}: {e}")
            import traceback
            traceback.print_exc()
    
    return True

def test_backward_compatibility():
    """Test that the old interface still works if available."""
    print("\n=== Backward Compatibility Test ===")
    
    try:
        # Try importing with new initialization
        import importlib.util
        spec = importlib.util.spec_from_file_location("diso", "diso/__init__.py")
        diso = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diso)
        
        print("✓ New DISO module loaded")
        
        # Test device capabilities
        caps = diso.get_device_capabilities()
        print(f"✓ Device capabilities: {caps}")
        
        # Test classes
        if caps['multiplatform_available']:
            mc = diso.DiffMC(torch.float32)
            print("✓ Multi-platform DiffMC created")
        else:
            print("✗ Multi-platform support not available")
        
    except Exception as e:
        print(f"✗ Error testing backward compatibility: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("DISO Multi-Platform Support Investigation")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Run tests
    test_multiplatform_mc()
    test_backward_compatibility()
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()