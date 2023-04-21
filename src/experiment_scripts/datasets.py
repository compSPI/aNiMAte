import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataio import RelionDataLoader, AtomicProjectionSimulator
from ctf_utils import CTFIdentity, CTF
from noise_utils import AWGNGenerator
from shift_utils import Shift

def train_val_datasets(config):
    noise = AWGNGenerator(snr=config.sim_awgn_snr)
    # ctf = CTFIdentity()

    config.ctf_size = config.map_shape[0]
    ctf = CTF(size=config.ctf_size, resolution=config.resolution,
              defocus_mean=config.sim_ctf_defocus_mean, defocus_stdev=config.sim_ctf_defocus_stdev,
              angleAstigmatism=config.sim_ctf_angle_astigmatism,
              cs=config.sim_ctf_spherical_abberations, requires_grad=False)
    shift = Shift(size=config.map_shape[0], resolution=config.resolution,
                  shift_x_mean=config.sim_shift_x_mean, shift_x_stdev=config.sim_shift_x_stdev,
                  shift_y_mean=config.sim_shift_y_mean, shift_y_stdev=config.sim_shift_y_stdev)

    # We use the parameters of the simulation here, to match the ones of the model
    config.spherical_aberration = config.sim_ctf_spherical_abberations
    config.num_particles = config.simul_num_projs

    if config.experiment_type == 'exp_relion_reconstruct':
        train_dataset = RelionDataLoader(relion_path=config.relion_path,
                                         relion_star_file=config.relion_star_file,
                                         relion_invert_hand=config.relion_invert_hand,
                                         atomic_nma_number_modes=config.atomic_nma_number_modes)
        config.side_len, \
        config.kV, \
        config.resolution, \
        config.spherical_aberration, \
        config.amplitude_contrast = train_dataset.get_df_optics_params()
        config.map_shape = [config.side_len] * 3
        config.num_particles = len(train_dataset)
        config.ctf_size = config.side_len

        if config.val_relion_star_file:
            val_dataset = RelionDataLoader(relion_path=config.relion_path,
                                           relion_star_file=config.val_relion_star_file,
                                           relion_invert_hand=config.relion_invert_hand,
                                           atomic_nma_number_modes=config.atomic_nma_number_modes)
        else:
            val_dataset = None
    elif config.experiment_type == 'exp_simul_atomic':
        train_dataset = AtomicProjectionSimulator(config,
                                                  projection_sz=config.map_shape,
                                                  num_projs=config.simul_num_projs,
                                                  ctf_generator=ctf,
                                                  noise_generator=noise,
                                                  shift_generator=shift)
        val_dataset = AtomicProjectionSimulator(config,
                                                projection_sz=config.map_shape,
                                                num_projs=config.val_sz,
                                                ctf_generator=ctf,
                                                noise_generator=noise,
                                                shift_generator=shift)

    return train_dataset, val_dataset