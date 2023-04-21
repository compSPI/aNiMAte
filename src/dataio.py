import torch
import torch.fft
import mrcfile
import starfile
from torch.utils.data import Dataset
import numpy as np
import os
from cryonet import AtomicModel, DynamicsModelNMA, DynamicsModelNone, ExplicitAtomicVolume
from pytorch3d.transforms import random_rotations, euler_angles_to_matrix
from ctf_utils import primal_to_fourier_2D, fourier_to_primal_2D
from abc import ABCMeta, abstractmethod
from pytorch3d.transforms import euler_angles_to_matrix, quaternion_to_matrix, rotation_6d_to_matrix

class SimulatorBase(Dataset, metaclass=ABCMeta):
    def __init__(self, projection_sz, num_projs,
                 noise_generator=None, ctf_generator=None, shift_generator=None):
        super(SimulatorBase, self).__init__()
        self.projection_sz = projection_sz
        self.vol_sidelen = self.projection_sz[0]
        self.num_projs = num_projs
        self.noise_generator = noise_generator
        self.ctf_generator = ctf_generator
        self.shift_generator = shift_generator

        # Generate num_projs rotations
        self.rotmat = random_rotations(self.num_projs)  # quaternion2rot3d(get_random_quat(self.num_projs))

        # Generate random CTF defocus values
        if ctf_generator is not None:
            self.ctf_defocus = np.random.lognormal(np.log(ctf_generator.defocus_mean),
                                                     ctf_generator.defocus_stdev,
                                                     num_projs)

        # Generate random CTF defocus values
        if shift_generator is not None:
            self.shift_x = shift_generator.shift_x_stdev * np.random.randn(num_projs) + shift_generator.shift_x_mean
            self.shift_y = shift_generator.shift_y_stdev * np.random.randn(num_projs) + shift_generator.shift_y_mean

        # Keep precomputed projections to avoid recomputing them
        # and to get the same random realizations (for e.g. for noise)
        self.precomputed_projs = [None] * self.num_projs
        self.precomputed_fprojs = [None] * self.num_projs

    def _get_ctf_params(self, idx):
        if self.ctf_generator is not None:
            defocus_u = torch.from_numpy(np.array(self.ctf_defocus[idx], ndmin=2)).float()
            defocus_v = defocus_u
        else:
            defocus_u = defocus_v = None
        return defocus_u, defocus_v

    def _get_shift_params(self, idx):
        if self.shift_generator is not None:
            shift_x = torch.from_numpy(np.array(self.shift_x[idx], ndmin=1)).float()
            shift_y = torch.from_numpy(np.array(self.shift_y[idx], ndmin=1)).float()
        else:
            shift_x = shift_y = None
        return shift_x, shift_y

    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        rotmat = self.rotmat[idx, :]
        defocus_u, defocus_v = self._get_ctf_params(idx)
        shift_x, shift_y = self._get_shift_params(idx)
        # If the projection has been precomputed already, use it
        if self.precomputed_projs[idx] is not None:
            proj = self.precomputed_projs[idx]
            fproj = self.precomputed_fprojs[idx]
        else:  # otherwise precompute it
            proj = self._simulate_projection(rotmat, idx)
            if self.ctf_generator is not None:
                ''' Generate fproj (fourier) '''
                fproj = primal_to_fourier_2D(proj)

                ''' CTF model (fourier) '''
                fproj = self.ctf_generator(fproj, {'defocus_u': defocus_u,
                                                   'defocus_v': defocus_v,
                                                   'angleAstigmatism': self.ctf_generator.global_angleAstigmatism *
                                                                       torch.ones(1, 1, 1)}
                                           )[0, ...]
                if self.shift_generator is not None:
                    fproj = self.shift_generator(fproj, {'shift_x': shift_x, 'shift_y': shift_y}).squeeze(1)
                ''' Update primal proj '''
                proj = fourier_to_primal_2D(fproj).real

            ''' Noise model (primal) '''
            if self.noise_generator is not None:
                proj = self.noise_generator(proj)

            ''' Store precomputed projs / fproj '''
            self.precomputed_projs[idx] = proj
            self.precomputed_fprojs[idx] = primal_to_fourier_2D(proj)

        # TODO: create a misc dict to pass miscellaneous data, that does not
        # go on the GPU, but is still fed to the model (e.g. length of the dataset).
        in_dict = {'proj': proj,
                   # The eventual groundtruth rotation
                   'rotmat': rotmat,
                   'idx': torch.tensor(idx, dtype=torch.long),
                   'defocus_u': defocus_u,
                   'defocus_v': defocus_v,
                   'shift_x': shift_x,
                   'shift_y': shift_y,
                   'angleAstigmatism': self.ctf_generator.global_angleAstigmatism * torch.ones(1, 1)}

        gt_dict = {'proj': proj,
                   'fproj': fproj,
                   'rotmat': rotmat,
                   'shift_x': shift_x,
                   'shift_y': shift_y}  # this is the dict passed to the loss function
        return in_dict, gt_dict

    @abstractmethod
    def _simulate_projection(self, rotmat, idx):
        ...

class RelionDataLoader(Dataset):
    def __init__(self, relion_path, relion_star_file, relion_invert_hand, atomic_nma_number_modes):
        self.relion_path = relion_path
        self.relion_star_file = relion_star_file
        self.df = starfile.read(os.path.join(self.relion_path, self.relion_star_file))

        self.vol_sidelen = self.df['optics']['rlnImageSize'][0]
        self.invert_hand = relion_invert_hand
        self.num_projs = len(self.df['particles'])

        # for simulated starfiles
        self.atomic_nma_number_modes = atomic_nma_number_modes

    def get_df_optics_params(self):
        return self.df['optics']['rlnImageSize'][0], \
               self.df['optics']['rlnVoltage'][0], \
               self.df['optics']['rlnImagePixelSize'][0],\
               self.df['optics']['rlnSphericalAberration'][0],\
               self.df['optics']['rlnAmplitudeContrast'][0]

    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        particle = self.df['particles'].iloc[idx]
        try:
            # Load particle image from mrcs file
            imgnamedf = particle['rlnImageName'].split('@')
            mrc_path = os.path.join(self.relion_path, imgnamedf[1])
            pidx = int(imgnamedf[0]) - 1
            with mrcfile.mmap(mrc_path, mode='r', permissive=True) as mrc:
                proj_np = mrc.data[pidx]
                proj = torch.from_numpy(proj_np).float()
            proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
            # --> (C,H,W)
        except Exception:
            print(f"WARNING: Particle image {particle['rlnImageName']} invalid!\nSetting to zeros.")
            proj = torch.zeros(self.vol_sidelen, self.vol_sidelen)
            proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)

        # Generate CTF from relion paramaters
        defocus_u = torch.from_numpy(np.array(particle['rlnDefocusU'] / 1e4, ndmin=2)).float()
        defocus_v = torch.from_numpy(np.array(particle['rlnDefocusV'] / 1e4, ndmin=2)).float()
        angleAstigmatism = torch.from_numpy(np.radians(np.array(particle['rlnDefocusAngle'], ndmin=2))).float()

        # Read relion "GT" orientations
        relion_euler_np = np.radians(np.stack([-particle['rlnAnglePsi'],                                     # convert Relion to our convention
                                               particle['rlnAngleTilt'] * (-1 if self.invert_hand else 1),   # convert Relion to our convention + invert hand
                                               -particle['rlnAngleRot']]))                                   # convert Relion to our convention
        rotmat = euler_angles_to_matrix(torch.from_numpy(relion_euler_np[np.newaxis,...]), convention='ZYZ')
        rotmat = torch.squeeze(rotmat).float()

        # Read relion "GT" shifts
        shift_x = torch.from_numpy(np.array(particle['rlnOriginXAngst']))
        shift_y = torch.from_numpy(np.array(particle['rlnOriginYAngst']))

        in_dict = {'proj': proj,
                   # The eventual groundtruth rotation
                   'rotmat': rotmat,
                   # The eventual groundtruth CTF parameters
                   'defocus_u': defocus_u,
                   'defocus_v': defocus_v,
                   'angleAstigmatism': angleAstigmatism,
                   'shift_x': shift_x,
                   'shift_y': shift_y,
                   # The sample idx for any autodecoder type of model
                   'idx': torch.tensor(idx, dtype=torch.long)}  # this is the dict passed to the model
        gt_dict = {'proj': proj,
                   'rotmat': rotmat,
                   'shift_x': shift_x,
                   'shift_y': shift_y}  # this is the dict passed to the loss function

        if 'nmaAlphas' in particle:
            nmaAlphas = np.fromstring(particle['nmaAlphas'], sep=',')
            gt_dict.update({'nma_alphas': torch.from_numpy(nmaAlphas[np.newaxis,:self.atomic_nma_number_modes]),
                            'pdb_index': int(particle['pdbIndex'])})

        return in_dict, gt_dict


class AtomicProjectionSimulator(SimulatorBase):
    def __init__(self, config, projection_sz, num_projs,
                 noise_generator=None, ctf_generator=None, shift_generator=None):
        super(AtomicProjectionSimulator, self).__init__(projection_sz, num_projs,
                                                        noise_generator, ctf_generator, shift_generator)

        self.config = config
        atomic_model = AtomicModel(self.config.atomic_pdb, self.config.atomic_clean_pdb, self.config.atomic_center,
                                   pdb_out=os.path.join(config.root_dir, 'curated_gemmi.pdb'))

        if config.dynamic_model == 'nma':
            atomic_model = DynamicsModelNMA(atomic_model, atomic_clean_pdb=config.atomic_clean_pdb,
                                            atomic_cg_selection=config.atomic_cg_selection,
                                            atomic_nma_cutoff=config.atomic_nma_cutoff,
                                            atomic_nma_gamma=config.atomic_nma_gamma,
                                            atomic_nma_number_modes=config.atomic_nma_number_modes,
                                            atomic_nma_pkl=config.atomic_nma_pkl,
                                            log_dir=config.root_dir, by_chain=config.atomic_nma_by_chain)
            rmsd_means = np.random.choice(config.atomic_nma_gauss_means,
                                          size=(self.num_projs, atomic_model.n_chains, config.atomic_nma_number_modes))
            rmsd_stdevs = np.random.choice(config.atomic_nma_gauss_stdevs,
                                           size=(self.num_projs, atomic_model.n_chains, config.atomic_nma_number_modes))
            self.nma_coordinates = torch.normal(mean=torch.tensor(rmsd_means).float(),
                                                std=torch.tensor(rmsd_stdevs).float())
        else:
            atomic_model = DynamicsModelNone(atomic_model)
            self.nma_coordinates = torch.zeros(self.num_projs, 1, config.atomic_nma_number_modes)

        self.simul_map = ExplicitAtomicVolume(atomic_model,
                                              sidelen=config.map_shape[0],
                                              pixel_size=config.resolution,
                                              log_dir=config.root_dir)

    def _simulate_projection(self, rotmat, idx):
        nma_coordinates = self.nma_coordinates[idx, :]
        return self.simul_map(rotmat[None, :, :], nma_coordinates[None, :])  # add a dummy batch dimension

    def __getitem__(self, idx):
        in_dict, gt_dict = super().__getitem__(idx)
        gt_dict.update({'nma_coords': self.nma_coordinates[idx, :]})
        return in_dict, gt_dict
