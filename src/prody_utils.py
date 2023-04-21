import os
import torch
import numpy as np
from prody import *


def build_hessian(model, cutoff, gamma):
    anm = ANM('object')
    anm.buildHessian(model, cutoff=cutoff, gamma=gamma)
    return anm


def compute_prody_nma_base(model, n_modes, cutoff=15., gamma=1., use_pytorch=False):
    """
    Perform Normal Mode Analysis using an anisotropic network model.

    Parameters
    ----------
    model : ProDy AtomGroup object
        atomic model
    n_modes : int
        number of normal modes
    cutoff : float, default: 15
        cutoff distance for pairwise interactions in Angstrom
    gamma : float, default: 1
        spring constant
    use_pytorch : bool, default: False
        use pytorch to calculate the modes. Needs cuda support.

    Returns
    -------
    anm : ProDy dynamics object
        anisotropic network model
    """
    anm = build_hessian(model, cutoff=cutoff, gamma=gamma)
    if use_pytorch:
        H = torch.from_numpy(anm.getHessian().astype(np.float32)).cuda()
        eigvals, eigvecs = torch.linalg.eigh(H)
        eigvecs = eigvecs[:, 6:6 + n_modes].detach().cpu().numpy()
        eigvals = eigvals[6:6 + n_modes].detach().cpu().numpy()
        anm.setEigens(eigvecs, values=eigvals)
    else:
        anm.calcModes(n_modes=n_modes)
    return anm


def compute_prody_nma(model, n_modes, cutoff=15., gamma=1., by_chain=False, use_pytorch=False):
    """
    Wrapper for computing a model's normal modes that allows chains
    to be treated together or as individual units.

    Parameters
    ----------
    model : ProDy AtomGroup object
        atomic model
    n_modes : int
        number of normal modes
    cutoff : float, default: 15
        cutoff distance for pairwise interactions in Angstrom
    gamma : float, default: 1
        spring constant
    by_chain : bool, default: False
        if True, compute normal modes separately per chain.
        Otherwise, treat the model as a single chain.
    use_pytorch : bool, default: False
        use pytorch to calculate the modes. Needs cuda support.

    Returns
    -------
    anm : ProDy dynamics object
        anisotropic network model
    """
    anm = compute_prody_nma_base(model, n_modes, cutoff=cutoff, gamma=gamma, use_pytorch=use_pytorch)

    if by_chain:
        chain_ids = np.unique(model.getChids())
        eigvecs, eigvals = [], []

        # compute a normal modes model for each chain individually
        for ch in chain_ids:
            model_ch = model.select(f'chain {ch}')
            anm_ch = compute_prody_nma_base(model_ch, n_modes, cutoff=15., gamma=1., use_pytorch=use_pytorch)
            eigvecs.append(anm_ch.getEigvecs())
            eigvals.append(anm_ch.getEigvals())

        eigvecs = np.vstack(np.array(eigvecs, dtype='object'))  # stack eigenvectors
        eigvecs /= np.sqrt(len(chain_ids))  # renormalize to account for multiple chains
        eigvals = np.sum(np.vstack(np.array(eigvals)), axis=0)  # sum eigenvalues across chains
        assert eigvecs.shape == anm.getEigvecs().shape
        anm.setEigens(eigvecs, values=eigvals)

    return anm


def extend_prody_nma(nma_small, model_small, model_extended):
    """
    Extend NMA from smaller (coarse-grained) to larger model.

    Parameters
    ----------
    nma_small : ProDy dynamics object
        anisotropic network model
    model_small : ProDy AtomGroup object
        atomic model associated with nma_small
    model_extended : ProDy AtomGroup object
        atomic model, more detailed (nodes) than model_small

    Returns
    -------
    nma : ProDy dynamics object
        extended anisotropic network model
    atoms : ProDy AtomGroup object
        atomic model, more detailed (nodes) than model_small
    """
    return extendModel(nma_small, model_small, model_extended)


def read_prody_model(path):
    """
    Use ProDy library to read PDB or mmCIF file and return a ProDy model.

    Parameters
    ---------
    path : string
        PDB or mmCIF file

    Returns
    -------
    model : ProDy AtomGroup object
        atomic model
    """
    if os.path.isfile(path):
        is_pdb = path.lower().endswith(".pdb")
        is_cif = path.lower().endswith(".cif")
        if is_pdb:
            model = parsePDB(path)
        elif is_cif:
            model = parseMMCIF(path)
        else:
            raise ValueError("File format not recognized.")
    else:
        raise OSError("File could not be found.")
    return model


def write_prody_model(path, model, msf=None):
    """
    Write ProDy model to PDB file.

    Parameters
    ---------
    path : string
        output PDB file
    model : ProDy AtomGroup object
        atomic model
    msf : numpy.ndarray, (n_atoms)
        mean-squared fluctuations to write to B-factor column
    """
    if msf is not None:
        model.setBetas(msf)
    is_pdb = path.lower().endswith(".pdb")
    if is_pdb:
        writePDB(path, model)
    else:
        raise ValueError("File format not recognized.")


def write_nma_trajectory(path, model, eigvecs, n_mode=0, n_frames=20, rmsd=2):
    """
    Write out a trajectory that visualizes the normal modes dynamics.

    Parameters
    ----------
    path : string
        output path for PDB file
    model : ProDy AtomGroup object
        atomic model
    eigvecs : numpy.ndarray, shape (n_atoms, 3, n_modes)
        eigenvectors array
    n_mode : int, default: 0
        index of mode to visualize
    n_frames : int, default: 20
        number of trajectory frames
    rmsd : float
        scale factor for exaggerating dynamics
    """
    traj = model.copy()
    coords = traj.getCoords()

    for i, t in enumerate(np.linspace(0, 1, n_frames)):
        prefac = rmsd * np.sin(2 * np.pi * t) * np.sqrt(model.numAtoms())
        traj.addCoordset(coords + prefac * eigvecs[:, :, n_mode])
    write_prody_model(path, traj)

    return
