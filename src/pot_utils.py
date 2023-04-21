"""Define Electrostatic Potential."""
import os
import numpy as np
import torch

import pykeops
from pykeops.torch import LazyTensor


class ContiguousBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


contiguous_backward = ContiguousBackward.apply


class Potential(torch.nn.Module):
    def __init__(self, sidelen, pixel_sz, log_dir, clean_pykeops=False):
        super(Potential, self).__init__()

        self.sidelen = sidelen
        self.pixel_sz = pixel_sz
        pykeops.set_build_folder(os.path.join(log_dir, 'tmp/pykeops'))
        if clean_pykeops:
            pykeops.clean_pykeops()

    def calculate_potential(self, batch_sz, n_atoms, coords, grid, ff_a, ff_b):
        dimension = coords.shape[-1]
        invb = (4.0 * np.pi) / ff_b

        ''' Glue with pykeops '''
        n_pix = self.sidelen ** dimension
        atom_coords_sym = LazyTensor(coords.view(batch_sz, 1, n_atoms, 1, dimension))  # -> [b,5,atoms,pix,D]
        pix_coords_sym = LazyTensor(grid.view(batch_sz, 1, 1, n_pix, dimension))  # -> [b,5,atoms,pix,D]
        ff_a_sym = LazyTensor(
            ff_a.view(1, n_atoms, 1, 5, 1).permute(0, 3, 1, 2, 4).contiguous())  # -> [b,5,atoms,pix,1]
        invb_sym = LazyTensor(
            invb.view(1, n_atoms, 1, 5, 1).permute(0, 3, 1, 2, 4).contiguous())  # -> [b,5,atoms,pix,1]

        dist_sq = (invb_sym * (pix_coords_sym - atom_coords_sym) ** 2).sum(dim=-1)

        contrib = (-np.pi * dist_sq).exp() * ff_a_sym[:, :, :, :, 0] * invb_sym[:, :, :, :, 0]  # [b,5,atoms,pix,]
        potential = contiguous_backward(contrib.sum(dim=2))  # sum over atoms -> [b,5,pix,1]
        return torch.sum(potential, dim=1)  # sum over gaussians -> [b,pix,D]

    def forward(self, coords, ff_a, ff_b):
        batch_sz = coords.shape[0]
        n_atoms = coords.shape[-2]

        atom_coords_2d = coords[..., :2].reshape(-1, 2).contiguous()

        ax = torch.arange(-self.sidelen // 2, ((self.sidelen - 1) // 2) + 1).float()
        pix_coords_X, pix_coords_Y = torch.meshgrid(ax, ax)
        pix_coords_2d = torch.stack([pix_coords_Y, pix_coords_X], dim=-1).repeat(batch_sz, 1, 1, 1).reshape(-1, 2).to(
            coords.device) * self.pixel_sz
        potential = self.calculate_potential(batch_sz, n_atoms, atom_coords_2d, pix_coords_2d, ff_a, ff_b)

        return potential.reshape(batch_sz, 1, self.sidelen, self.sidelen)

    def calculate_3d(self, coords, ff_a, ff_b):
        batch_sz = coords.shape[0]
        n_atoms = coords.shape[-2]

        ax = torch.arange(-self.sidelen // 2, ((self.sidelen - 1) // 2) + 1).float()
        pix_coords_X, pix_coords_Y, pix_coords_Z = torch.meshgrid(ax, ax, ax)
        pix_coords_3d = torch.stack([pix_coords_Y, pix_coords_X, pix_coords_Z], dim=-1).repeat(batch_sz, 1, 1,
                                                                                               1).reshape(-1, 3).to(
            coords.device) * self.pixel_sz
        potential = self.calculate_potential(batch_sz, n_atoms, coords, pix_coords_3d, ff_a, ff_b)
        return potential.reshape(self.sidelen, self.sidelen, self.sidelen)


class Potential_Full(torch.nn.Module):
    def __init__(self, sidelen, pixel_sz):
        super(Potential_Full, self).__init__()

        self.sidelen = sidelen
        self.pixel_sz = pixel_sz
        self.map_sz = float(self.pixel_sz) * float(self.sidelen)
        self.pixel_center = self.sidelen // 2
        self.coord_center = self.map_sz // 2

        ax = torch.arange(-self.pixel_center, ((sidelen - 1) // 2) + 1).float()
        Y, X = torch.meshgrid(ax, ax)
        sampled_points = torch.stack((X, Y), dim=-1).reshape(
            [1, -1, 2]) * self.pixel_sz  # 2D grid in real units (angstrom)

        self.register_buffer('sampled_points', sampled_points)

    def forward(self, atomic_coords, atomic_ff_a, atomic_ff_b, atom_block, potential_block):
        batch_sz = atomic_coords.shape[0]
        invb = (4.0 * np.pi) / atomic_ff_b

        natoms = atomic_coords.shape[1]
        npixels = self.sidelen ** 2
        potential = torch.zeros(batch_sz, npixels)  # .to(self.device)

        for i in range(0, natoms, atom_block):
            end_index = min(i + atom_block, natoms)
            a_block = atomic_ff_a[i:end_index]
            invb_block = invb[i:end_index]
            atom_block_coords = atomic_coords[:, i:end_index, :2]

            for j in range(0, npixels, potential_block):
                pend_index = min(j + potential_block, npixels)
                sampled_block = self.sampled_points[:, j:pend_index, :]

                dist = torch.cdist(sampled_block, atom_block_coords, p=2.0)
                for k in range(a_block.shape[-1]):
                    exponent = torch.exp(-np.pi * invb_block[None, None, ..., k] * dist.pow(2))
                    potential[:, j:pend_index] += \
                        torch.bmm(exponent, a_block[None, ..., k, None] * invb_block[None, ..., k, None])[..., 0]

        potential = potential.reshape(batch_sz, 1, self.sidelen, self.sidelen)

        # Transpose the image to match the mesh grids in volume representations (Cartesian to grid)
        return potential.transpose(3, 2)
