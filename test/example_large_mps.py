import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diso import UniversalDiffDMC

sizes = [18, 34, 66, 130, 258, 506]

mc = UniversalDiffDMC(dtype=torch.float32).to('mps')
for size in sizes:
    print('testing size', size)
    sdf = torch.randn((size, size, size), device='mps', dtype=torch.float32)
    try:
        verts, faces = mc(sdf, deform=None, return_quads=False, normalize=False)
        print(' ok verts', verts.shape, 'faces', faces.shape)
    except Exception as e:
        print(' failed', e)
        break