"""Define molecular dynamics models."""
import numpy as np

import torch
import pykeops
from pykeops.numpy import LazyTensor

from prody_utils import *
import os


class NormalModeAnalysis:
    def __init__(self,
                 atomic_model_filepath,
                 atomic_cg_selection='protein and name CA',
                 atomic_nma_cutoff=15.,
                 atomic_nma_gamma=1.,
                 atomic_nma_number_modes=1,
                 log_dir="./",
                 by_chain=False,
                 use_pykeops=True,
                 clean_pykeops=True,
                 use_pytorch=True):
        super(NormalModeAnalysis, self).__init__()

        self.cg_selection = atomic_cg_selection  # atomic selection for coarse-graining
        self.nma_n_modes = atomic_nma_number_modes  # number of normal modes
        self.nma_cutoff = atomic_nma_cutoff  # interatomic threshold distance in Angstroms
        self.nma_gamma = atomic_nma_gamma  # spring constant for computing NMA
        self.by_chain = by_chain  # if True, treat chains separately
        self.use_pykeops = use_pykeops # if True, use pykeops when possible
        self.use_pytorch = use_pytorch

        self.atoms = read_prody_model(atomic_model_filepath)
        # select atoms to permit coarse-graining; 'all' for no coarse-graining
        self.atoms_sel = self.atoms.select(self.cg_selection)
        self.missing_indices = np.empty(0)
        self.kept_indices = self.atoms_sel.getIndices()
        pykeops.set_build_folder(os.path.join(log_dir, 'tmp/pykeops'))
        if clean_pykeops:
            pykeops.clean_pykeops()
        self.build_nma()

    def get_eigenpair(self, nma):
        """
        Retrieve the eigenvectors and eigenvalues of all modes and update the
        corresponding self variables.

        Parameters
        ----------
        nma : ProDy dynamics object
            anisotropic network model
        """
        num_atoms = nma.numAtoms()
        self.eigvals = nma.getEigvals()
        self.eigvecs = nma.getEigvecs().reshape(num_atoms, 3, self.nma_n_modes)

    def identify_outliers(self):
        """
        Identify atoms that are outliers in terms of mean-squared fluctuation.
        MSF = Sum_k Sum_a 1/lambda_k * square(U_k(a))
        where lambda and U are respectively the eigenvalue and eigenvector of
        mode k, and the sums are over all modes and the three coordinate axes.

        Returns
        -------
        outliers : numpy.ndarray, 1d
            indices of atomic outliers of the (possibly coarse-grained) model
        """
        fluct = np.sum(np.sum(np.square(self.eigvecs), axis=1) * 1.0 / self.eigvals, axis=1)
        outliers = np.where(fluct > (np.mean(fluct) + 10 * np.std(fluct)))[0]
        return outliers

    def find_missing_atoms(self):
        """
        Identify indices of atoms whose eigenvectors need to be (re)computed:
        MSF outliers and atoms omitted due to coarse-graining.

        Returns
        -------
        missing_indices : numpy.ndarray, 1d
            indices of atoms whose eigenvectors need to be (re)calculated
        kept_indices : numpy.ndarray, 1d
            indices of atoms whose eigenvectors we keep.
        """

        # identify indices of MSF outliers and omit from eigenvectors list
        outliers = self.identify_outliers()
        if len(outliers) > 0:
            print(f"Detected {len(outliers)} MSF outliers")
            self.missing_indices = self.kept_indices[outliers]
            self.kept_indices = np.setdiff1d(self.kept_indices, self.missing_indices)
            self.eigvecs = np.delete(self.eigvecs, outliers, axis=0)
            self.normalize_eigvecs()

        # update atom selection
        selection = f'index {self.kept_indices[0]}'
        for index in np.arange(1, self.kept_indices.shape[0]):
            selection += f' or index {self.kept_indices[index]}'
        self.atoms_sel = self.atoms_sel.select(selection)

        # identify indices of omitted atoms if coarse-graining
        if self.cg_selection != 'all':
            self.missing_indices = np.setdiff1d(np.arange(self.atoms.numAtoms()), self.kept_indices)

    def extend_model_chain(self):
        """
        Extend the model -- specifically, estimate the values of the eigenvectors
        for atoms that were removed from the initial model as MSF outliers and/or
        omitted due to coarse-graining. The eigenvector for atom j is estimated as:
        V_j = Sum_i w_ij V_i / Sum_i w_ij
        where w_ij = 1/r_ij, the inverse distance between atoms i and j and V_i is
        the eigenvector of atom i.

        Returns
        -------
        V_all : numpy.ndarray, shape (n_atoms,3,n_modes)
            full set of eigenvectors
        """
        # generate array for all eigenvectors and fill with available ones
        V_all = np.zeros((self.atoms.numAtoms(), 3, self.nma_n_modes))
        V_all[self.kept_indices, :, :] = self.eigvecs_initial

        pos_i = self.atoms.getCoords()
        if not self.use_pykeops:
            # compute weights: inverse interatomic distance
            deltas = np.abs(pos_i - pos_i[:, np.newaxis])
            deltas = np.square(np.sum(np.square(deltas), axis=2))
            weights = np.divide(1, deltas, where=deltas != 0, out=np.zeros_like(deltas))  # shape is [n_total, n_total]

        # estimate values for the missing eigenvectors
        start = 0
        for ch in range(len(self.n_atoms_per_chain)):
            prev_start = start
            start += self.n_atoms_per_chain[ch]
            indices_sel_present = np.intersect1d(range(prev_start, start),
                                                 self.kept_indices)
            indices_sel_missing = np.intersect1d(range(prev_start, start),
                                                 self.missing_indices)

            if len(indices_sel_missing) != 0:
                for nm in range(self.nma_n_modes):
                    if self.use_pykeops:
                        x = LazyTensor(pos_i[indices_sel_missing, None, :])
                        y = LazyTensor(pos_i[None, indices_sel_present, :])
                        weights = 1. / (((x - y) ** 2).sum(dim=-1))**2
                        mult = weights @ V_all[indices_sel_present, :, nm]
                        norm = weights.sum(dim=1)
                    else:
                        mult = np.dot(weights[indices_sel_missing][:, indices_sel_present],
                                      V_all[indices_sel_present, :, nm])
                        norm = np.sum(weights[indices_sel_missing][:, indices_sel_present], axis=1)[:, np.newaxis]
                    V_all[indices_sel_missing, :, nm] = mult / norm

        return V_all

    def normalize_eigvecs(self):
        """
        Enforce that each eigenvector is of norm 1.
        """
        for nm in range(self.nma_n_modes):
            norm = np.linalg.norm(self.eigvecs[..., nm].flatten())
            self.eigvecs[..., nm] /= norm

    def project_eigvecs(self):
        """
        Transform the eigenvectors to a different orthonormal basis.
        Here SVD is used
        """
        Q = self.eigvecs.reshape((-1, self.nma_n_modes))
        S = np.matmul(Q, np.diag(1. / self.eigvals))
        U, s, _ = np.linalg.svd(S, full_matrices=False)
        self.eigvecs = U.reshape(self.atoms.numAtoms(), 3, self.nma_n_modes)
        self.eigvals = 1./(s + np.finfo(float).eps)


    def build_nma(self):
        """
        Generate the normal modes model.
        """
        # compute normal modes
        nma = compute_prody_nma(self.atoms_sel,
                                n_modes=self.nma_n_modes,
                                cutoff=self.nma_cutoff,
                                gamma=self.nma_gamma,
                                by_chain=self.by_chain,
                                use_pytorch=self.use_pytorch)

        # retrieve model characteristics
        self.get_eigenpair(nma)
        chains, self.n_atoms_per_chain = np.unique(self.atoms.getChids(),
                                                   return_counts=True)
        if not self.by_chain:
            self.n_atoms_per_chain = [self.atoms.numAtoms()]

        # identify indices of missing atoms and extend model as needed
        self.find_missing_atoms()
        self.eigvecs_initial = self.eigvecs.copy()
        if len(self.missing_indices) > 0:
            print("Extending nma due to coarse-graining and/or MSF outliers")
            self.eigvecs_final = self.extend_model_chain()
            nma.setEigens(self.eigvecs_final.reshape(self.atoms.numAtoms() * 3, self.nma_n_modes),
                          values=self.eigvals)
            self.eigvecs = self.eigvecs_final
        else:
            self.eigvecs = self.eigvecs_initial

        # renormalize the eigenvectors
        self.normalize_eigvecs()
        # reproject the eigenvectors on a different orthonormal basis
        self.project_eigvecs()


def visualize_nma(nma, out_path, rmsd=5.):
    """
    Visualize features of the NMA, including 1. plotting the mean-squared
    fluctuations before/after coarse-graining / outlier removal, 2. saving
    trajectories of the normal modes, and 3. saving PDB files of the model
    with the B-factor column replaced by the MSFs.

    Parameters
    ----------
    nma : NormalModesAnalysis object
        nma object containing initial and final eigenvectors
    out_path : string
        directory to save plots and PDB files
    """
    import matplotlib.pyplot as plt

    # compute MSFs of initial and final models
    msf_initial = np.sum(np.sum(np.square(nma.eigvecs_initial), axis=1) * 1.0 / nma.eigvals, axis=1)
    msf_final = np.sum(np.sum(np.square(nma.eigvecs), axis=1) * 1.0 / nma.eigvals, axis=1)

    # plot MSFs
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

    ax1.plot(nma.kept_indices, msf_initial, c='black')
    ax2.plot(msf_final, c='black')

    ax1.set_title("Pre-coarse graining, outlier removal", fontsize=14)
    ax2.set_title("Final model", fontsize=14)
    ax2.set_xlabel("Atom No.", fontsize=12)
    for ax in [ax1, ax2]:
        ax.set_ylabel("MSF ($\mathrm{\AA}$)", fontsize=12)

    f.savefig(os.path.join(out_path, "msfs.png"), dpi=300, bbox_inches='tight')

    # sample along each mode and output a trajectory in PDB format
    for mode in range(nma.nma_n_modes):
        write_nma_trajectory(os.path.join(out_path, f"nma_mode{mode}.pdb"),
                             nma.atoms, nma.eigvecs, mode, rmsd=rmsd)

    # also write single PDB models with coarse-grained/pre-outlier vs final MSFs
    write_prody_model(os.path.join(out_path, f"nma_final.pdb"), nma.atoms,
                      msf=1.0 / msf_final.mean() * msf_final)
    write_prody_model(os.path.join(out_path, f"nma_initial.pdb"), nma.atoms_sel,
                      msf=1.0 / msf_initial.mean() * msf_initial)

    return

