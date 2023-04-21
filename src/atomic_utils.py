"""Build the Atomic Model Classes."""

import numpy as np
import torch
from gemmi_utils import read_gemmi_atoms, extract_atomic_parameter


class AtomicModel(torch.nn.Module):
    def __init__(self, atomic_model_filepath, atomic_clean_pdb, atomic_center, pdb_out=''):
        super(AtomicModel, self).__init__()

        self.pdb_filepath = atomic_model_filepath
        self.atomic_clean_pdb = atomic_clean_pdb
        self.atomic_center = atomic_center

        self.register_buffer('coords', torch.zeros((1, 3), dtype=torch.float32))
        self.register_buffer('ff_a', torch.zeros((1, 5), dtype=torch.float32))
        self.register_buffer('ff_b', torch.zeros((1, 5), dtype=torch.float32))

        self.update(self.pdb_filepath, self.atomic_clean_pdb, pdb_out=pdb_out)

    def update(self, pdb_filepath, atomic_clean_pdb, pdb_out=''):
        atoms = read_gemmi_atoms(pdb_filepath, pdb_out=pdb_out, clean=atomic_clean_pdb, center=self.atomic_center)
        if pdb_out != '':
            self.pdb_filepath = pdb_out

        self.coords = torch.Tensor(extract_atomic_parameter(atoms, 'cartesian_coordinates')).type_as(
            self.coords)  # <- [natoms, 3]
        self.ff_a = torch.Tensor(extract_atomic_parameter(atoms, 'form_factor_a')).type_as(self.ff_a)  # <- [natoms, 5]
        self.ff_b = torch.Tensor(extract_atomic_parameter(atoms, 'form_factor_b')).type_as(self.ff_b)  # <- [natoms, 5]

    def get_atomic_model(self):
        return self.forward()

    def forward(self):
        return [self.coords[None, :, :], self.ff_a[None, :, :], self.ff_b[None, :, :]]  # add a dummy dimension
