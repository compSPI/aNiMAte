"""Build the Dynamics Model Classes."""

import torch
import math, os
from nma_utils import NormalModeAnalysis

class DynamicsModelNone(torch.nn.Module):
    def __init__(self, atomic_model):
        super(DynamicsModelNone, self).__init__()
        self.atomic_model = atomic_model

    def get_atomic_model(self):
        return self.atomic_model()

    def get_eigvals(self):
        return None

    def forward(self, nma_coordinates={}):
        coords, ff_a, ff_b = self.atomic_model()
        return [coords.repeat(nma_coordinates.shape[0],1,1), ff_a, ff_b]

class DynamicsModelNMA(torch.nn.Module):
    def __init__(self, atomic_model, atomic_clean_pdb, atomic_cg_selection='protein and name CA',
                 atomic_nma_cutoff=15., atomic_nma_gamma=1., atomic_nma_number_modes=1,
                 atomic_nma_pkl=None, requires_grad=True, log_dir="./", by_chain=False):
        super(DynamicsModelNMA, self).__init__()
        self.atomic_model = atomic_model
        self.requires_grad = requires_grad

        if atomic_nma_pkl is None or atomic_nma_pkl == "":
            ''' compute normal modes '''
            nma = NormalModeAnalysis(atomic_model.pdb_filepath,
                                     atomic_cg_selection,
                                     atomic_nma_cutoff,
                                     atomic_nma_gamma,
                                     atomic_nma_number_modes,
                                     log_dir = log_dir,
                                     by_chain = by_chain,
                                     clean_pykeops = True)
            self.register_buffer('eigvecs', torch.Tensor(nma.eigvecs).float())
            self.register_buffer('eigvals',torch.Tensor(nma.eigvals).float())
            self.register_buffer('n_atoms_per_chain', torch.Tensor(nma.n_atoms_per_chain).int())
            self.n_chains = len(nma.n_atoms_per_chain)
            torch.save({'eigvecs': self.eigvecs,
                        'eigvals': self.eigvals,
                        'n_atoms_per_chain': self.n_atoms_per_chain,
                        'n_chains': self.n_chains}, os.path.join(log_dir, 'nma_buffers.pt'))
        else:
            buffers_dict = torch.load(atomic_nma_pkl)
            self.register_buffer('eigvecs', buffers_dict['eigvecs']) # <- [num_atoms, 3, num_modes]
            self.register_buffer('eigvals', buffers_dict['eigvals'])
            self.register_buffer('n_atoms_per_chain', buffers_dict['n_atoms_per_chain'])
            self.n_chains = buffers_dict['n_chains']

    def get_atomic_model(self):
        return self.atomic_model()

    def get_eigvals(self):
        return self.eigvals

    def _measure_rmsd(self, nma_coordinates):
        """Measure the Root-Mean Square Deviation between the current and reference model (in Angstrom)."""
        natoms = nma_coordinates.size(0)/3.
        return torch.sqrt(torch.sum(torch.square(nma_coordinates)) / natoms)

    def forward(self, nma_coordinates={}):
        """
        Deform the atomic coordinates along the normal modes according to the
        equation: X(a) = X_0 + Sum(a U_k)
        where X_0 are the initial coordinates, U_k are the eigenvectors, and a
        is approximately related to the rmsd of deformation given by the input
        argument nma_coordinates as follows:
        nma_coordinates = 1/sqrt(N) |a|
        where N is the number of atoms and the units are in Angstroms. The input
        nma_coordinates is a tensor of dimensions [batch_size,n_chains,n_modes].
        If n_chains > 1, nma_coordinates is broadcast such that different amounts
        of deformation are applied to the atoms of each chain for every image.
        """
        batch_sz = nma_coordinates.shape[0]
        num_atoms, num_dims, num_modes = self.eigvecs.shape[-3:]

        eigvecs = self.eigvecs.repeat(batch_sz,1,1,1)

        nma_coordinates_scaled = nma_coordinates * math.sqrt(num_atoms / num_modes)
        nma_coordinates_scaled = torch.repeat_interleave(nma_coordinates_scaled[:,:,None,:],
                                                         self.n_atoms_per_chain, axis=1) # [batch_sz,n_chains,num_modes] -> [batch_sz,num_atoms,1,num_modes]
        delta_coords = \
            torch.bmm(eigvecs.reshape(-1,num_dims,num_modes), # [batch_sz*num_atoms,3,num_modes]
                      nma_coordinates_scaled.reshape(-1,1,num_modes).permute(0,2,1))  # [batch_sz*num_atoms,1,num_modes]
        delta_coords = delta_coords.reshape(batch_sz, num_atoms, num_dims)

        coords, ff_a, ff_b = self.atomic_model()
        return [coords + delta_coords, ff_a, ff_b]