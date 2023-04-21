import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
import configargparse
import json, ast

import torch
import torch.distributed as dist

from utils import cond_mkdir
from datasets import train_val_datasets
from experiment import experiment


class json_dict(dict):
    def __str__(self):
        return json.dumps(self)


def init_config(parser):
    parser.add_argument('--data_loss', type=str, choices=['L2', 'CC'], default='L2',
                        help='Data loss term type: mean squared error(L2) or normalized cross correlation (CC).')
    parser.add_argument('--encoder', type=str, choices=['CNN', 'EfficientNetV2', 'VAE'], default='CNN',
                        help='Encoder type (CNN | VAE).')
    parser.add_argument('--encoder_conv_layers', type=int, nargs='*', default=[32, 64, 128],
                        help='The number of features per layer in the CNN encoder (image -> latent code).')
    parser.add_argument('--encoder_batch_norm', type=int, default=1,
                        help='Use batch-norm in the CNN encoder.')
    parser.add_argument('--encoder_max_pool', type=int, default=1,
                        help='Use max pool in the encoder.')
    parser.add_argument('--encoder_dropout', type=int, default=0,
                        help='Use dropout in the encoder.')
    parser.add_argument('--nma_reg_weight', type=float, default=0.,
                        help='NMA coordinate prediction regularization weight.')
    parser.add_argument('--proj_scaling_init', type=float, default=0.,
                        help='Initialization value for solving for a global projection scaling. Projections will be divided by exp(proj_scaling).')

    parser.add_argument('--regressor_orientation_layers', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--regressor_conformation_layers', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--regressor_shift_layers', type=int, nargs='*', default=[256, 256])

    parser.add_argument('--train_phases', type=str, nargs='*',
                        default=['{"name": "all", "end": -1, "lr_mult": 1}'],
                        help='Phases for training in JSON format. Defaults to one phase')
    parser.add_argument('--train_num_workers', type=int, default=20,
                        help='The number of workers to use for the dataloader.')
    parser.add_argument('--train_batch_sz', type=int, default=256,
                        help='The number of projections in a batch for training.')
    parser.add_argument('--train_chunk_sz', type=int, default=256,
                        help='The size of a chunk of views that can fit on the GPU,'
                             'chunk_sz < batch_size. The batch will be automatically divided'
                             'in chunks of chunk_sz and gradients are accumulated over the chunk'
                             'so that we perform one SGD step every batch_sz.')
    parser.add_argument('--train_learning_rate', type=float, default=1e-4,
                        help='The learning rate used during training.')
    parser.add_argument('--train_epochs', type=int, default=1000,
                        help='The number of epochs of training.')
    parser.add_argument('--train_steps_til_summary', type=int, default=100,
                        help='The number of steps (in #batches) until summary is called on training data.')
    parser.add_argument('--train_epochs_til_checkpoint', type=int, default=1,
                        help='The number epochs until model checkpoint is performed.')

    parser.add_argument('--val_sz', type=int, default=1024,
                        help='The number of projections in the validation data (only needed in simulation).')
    parser.add_argument('--val_relion_star_file', type=str,
                        help='The filepath to RELION\'s star file that contains the validation data.'
                             'Should be relative to --relion_path.')
    parser.add_argument('--val_chunk_sz', type=int, default=256,
                        help='The size of a chunk of projections that can fit on the GPU, val_chunk_sz < val_sz.')
    parser.add_argument('--val_steps_til_summary', type=int, default=10000,
                        help='The number of steps (in #batches) until summary is called on validation data.')

    parser.add_argument('--relion_path', type=str,
                        help='The root directory for a RELION run.')
    parser.add_argument('--relion_star_file', type=str,
                        help='The filepath to RELION\'s star file used for training.'
                             'Should be relative to --relion_path.')
    parser.add_argument('--relion_invert_hand', type=int, default=0,
                        help='Setting to true will mirror the raw images vertically.\n'
                             'Use in combination with GT orientations if relion\'s reconstruction had incorrect handedness')

    parser.add_argument('--ctf_size', type=int, default=128,
                        help='The size of the CTF filter used in reconstructions.')
    parser.add_argument('--ctf_valueNyquist', type=float, default=0.001,
                        help='Reconstruction CTF value at Nyquist.')
    parser.add_argument('--downsampling', type=int, default=1,
                        help='Particle image downsampling factor.')
    parser.add_argument('--kV', type=float, default=300.0,
                        help='Electron beam energy used.')
    parser.add_argument('--resolution', type=float, default=.8,
                        help='Particle image resolution (in Angstrom).')
    parser.add_argument('--spherical_abberation', type=float, default=2.7,
                        help='Spherical aberration.')
    parser.add_argument('--amplitude_contrast', type=float, default=0.1,
                        help='Amplitude contrast.')
    parser.add_argument('--num_particles', type=int, default=10000,
                        help='Total number of particle images to use in simulation.')

    parser.add_argument('--side_len', type=int, default=128,
                        help='The shape of the density map as determined by one side of the volume.')

    parser.add_argument('--so3_parameterization', type=str, default='s2s2',
                        choices=['s2s2', 'euler', 'quaternion', 'gt'],
                        help='The parameterization of SO3 influences the interpretation of the output of the orientation regressor.')
    parser.add_argument('--so3_refinement', type=int, default=0,
                        help='A flag to refine per-particle orientations. Default 0')

    parser.add_argument('--shift_input', type=str, default='gt',
                        choices=['gt', 'encoder'],
                        help='The source of shift parameters.')
    parser.add_argument('--mask_2D_diam', type=float, default=-1,
                        help='The diameter (in Angstrom) of a circular mask to applied to the predicted and input particle images.')

    parser.add_argument('--simul_num_projs', type=int, default=10000,
                        help='The number of projections to simulate in the volume.')
    parser.add_argument('--sim_awgn_snr', type=float, default=32,
                        help='SNR of the AWGN in the simulator in dB.')
    parser.add_argument('--sim_ctf_defocus_mean', type=float, default=1.0,
                        help='Mean of CTF defocus (U and V directions) distribution used in simulation. In micrometer units')
    parser.add_argument('--sim_ctf_defocus_stdev', type=float, default=0.1,
                        help='Standard deviation of CTF defocus (U and V directions) distribution used in simulation.')
    parser.add_argument('--sim_ctf_angle_astigmatism', type=float, default=0.0,
                        help='Angle of astigmatism of the CTF used in the simulations (in radians).')
    parser.add_argument('--sim_ctf_spherical_abberations', type=float, default=2.7,
                        help='Spherical abberations of the CTF used in the simulations.')
    parser.add_argument('--sim_shift_x_mean', type=float, default=0,
                        help='Mean of shifts (X direction) distribution used in simulation. In pixel units')
    parser.add_argument('--sim_shift_x_stdev', type=float, default=0,
                        help='Standard deviation of shifts (X direction) distribution used in simulation.')
    parser.add_argument('--sim_shift_y_mean', type=float, default=0,
                        help='Mean of shifts (Y direction) distribution used in simulation. In pixel units')
    parser.add_argument('--sim_shift_y_stdev', type=float, default=0,
                        help='Standard deviation of shifts (Y direction) distribution used in simulation.')

    parser.add_argument('--atomic_pdb', type=str, default='data/riboswitch.pdb',
                        help='The filepath to the PDB file containing the starting atomic model.')
    parser.add_argument('--atomic_cg_selection', type=str, default='protein and name CA',
                        # choices=['protein and name CA', 'nucleic and name P', 'all'],
                        help='The ProDy lingo for coarse-grained selection.')
    parser.add_argument('--atomic_nma_cutoff', type=float, default=15.,
                        help='The distance cutoff to pair atoms in the ProDy elastic network model.')
    parser.add_argument('--atomic_nma_gamma', type=float, default=1.,
                        help='The spring constant between paired atoms in the ProDy elastic network model.')
    parser.add_argument('--atomic_nma_number_modes', type=int, default=1,
                        help='The number of normal modes used to deform the atomic model.')
    parser.add_argument('--atomic_nma_by_chain', type=int, default=0,
                        help='Compute the chain-by-chain rather than global elastic network model.')
    parser.add_argument('--atomic_nma_gauss_means', type=float, nargs='*', default=[0, 3],
                        help='Mean RMSD in Angstrom of each conformational centroid along simulated mode(s).')
    parser.add_argument('--atomic_nma_gauss_stdevs', type=float, nargs='*', default=[0.5],
                        help='RMSD std deviation in Angstrom for conformational centroid(s).')
    parser.add_argument('--atomic_nma_pkl', type=str,
                        help='The filepath to a pickle file containing precomputed NMA coordinates (with eigvals).'
                             'The saved arrays must match other NMA config arguments (number of modes, chain, etc...).')
    parser.add_argument('--atomic_bfac_offset', type=float, default=0.,
                        help='The B-factor applied evenly to all the atoms in the atomic model.')
    parser.add_argument('--atomic_clean_pdb', type=int, default=1,
                        help='A flag to pass to gemmi when reading atoms from PDB files.')
    parser.add_argument('--atomic_center', type=int, default=1,
                        help='A flag to pass to gemmi when reading atoms from PDB files. Set to 1 centers the atomic model to its center of mass (COM) ')
    parser.add_argument('--atomic_global_pose', type=int, default=0,
                        help='A flag to solve for global pose (rotation and translation) for the atomic model. Default 0')
    parser.add_argument('--atomic_global_nma', type=int, default=0,
                        help='A flag to solve for global NMA for the atomic model. Default 0')
    parser.add_argument('--dynamic_model', type=str, default='nma', choices=['nma', 'none'],
                        help='The dynamic model to choose when an atomic model is present.')
    # We must give an experiment name
    parser.add_argument('--experiment_type', type=str, default='exp_relion_reconstruct',
                        choices=['exp_relion_reconstruct', 'exp_simul_atomic'],
                        help='The experiment to run.')
    parser.add_argument('--experiment_name', type=str, default=None, required=True,
                        help='An identifier for the train run.')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='Output directory for logging.')


def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, is_config_file=True,
                        help='Path to config file.')

    # This function will initialize a config file if it does not already exist,
    # by filling it with default parameters
    init_config(parser)

    config = parser.parse_args()

    config.map_shape = [config.side_len] * 3

    if config.experiment_name is None:
        parser.error('Error: --experiment_name is required.')

    config_dict = vars(config)
    config_dict['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        config_dict['world_size'] = int(os.environ['WORLD_SIZE'])
        config_dict['local_rank'] = int(os.environ['LOCAL_RANK'])

    config_dict['world_rank'] = 0
    if config_dict['world_size'] > 1:
        torch.cuda.set_device(config.local_rank)
        dist.init_process_group(backend='nccl')
        config_dict['world_rank'] = dist.get_rank()
        config_dict['global_batch_size'] = config.train_batch_sz
        config_dict['global_chunk_size'] = config.train_chunk_sz
        config_dict['train_batch_sz'] = int(config.train_batch_sz // config.world_size)
        config_dict['train_chunk_sz'] = int(config.train_chunk_sz // config.world_size)
    # Create root directory where models, logs and config files will be written
    config.root_dir = os.path.join(config.log_dir, config.experiment_name + f'_{config.world_rank}')
    if not cond_mkdir(config.root_dir):
        print(f"Error: cannot create root path.")
        return -1

    config.model_dir = os.path.join(config.root_dir, 'models')
    cond_mkdir(config.model_dir)

    # Write the config file
    config.train_phases = [json_dict(json.loads(s.replace("'", '"'))) for s in config.train_phases]
    parser.write_config_file(config, [os.path.join(config.root_dir, 'config.ini')])

    # Finally: launch the experiment!
    print(f"Launching experiment {config.experiment_type}")
    train_dataset, val_dataset = train_val_datasets(config)
    experiment(config, train_dataset, val_dataset)

    return 0, 'Training successful'


if __name__ == '__main__':
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Training failed.'

    print(status_message)
    exit(retval)
