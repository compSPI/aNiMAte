import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import traceback
import configargparse

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from utils import cond_mkdir, to_numpy, traj2ic
from training_chunks import evaluate_model
from main import init_config
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from cryonet import CryoNet
from dataio import AtomicProjectionSimulator, RelionDataLoader
from ctf_utils import CTF
from noise_utils import AWGNGenerator
from atomic_utils import AtomicModel
from dynamics_utils import DynamicsModelNMA
from prody_utils import read_prody_model, write_prody_model
from starfile_utils import create_starfile


def _noise_generator(config):
    return AWGNGenerator(snr=config.sim_awgn_snr)


def _ctf_generator(config):
    config.ctf_size = config.map_shape[0]
    config.spherical_aberration = config.sim_ctf_spherical_abberations
    config.num_particles = config.simul_num_projs
    return CTF(size=config.ctf_size, resolution=config.resolution,
               defocus_mean=config.sim_ctf_defocus_mean, defocus_stdev=config.sim_ctf_defocus_stdev,
               angleAstigmatism=config.sim_ctf_angle_astigmatism,
               cs=config.sim_ctf_spherical_abberations, requires_grad=False)


def _filter_outliers(pred_nma_coords, sigma_num=3.0):
    filter_idx = []
    for i in range(pred_nma_coords.shape[-1]):
        filter_idx.extend(np.where(abs(pred_nma_coords[..., i] - np.median(pred_nma_coords[..., i]))
                                   > sigma_num * pred_nma_coords[..., i].std())[0])
    filter_idx = np.unique(np.array(filter_idx))
    if filter_idx.shape[0] == 0:
        return pred_nma_coords, None
    return np.delete(pred_nma_coords, filter_idx, 0), filter_idx


def plot_nma(gt, pred, config):
    sim = False
    if 'nma_coords' in gt.keys():
        gt_nma_coords = to_numpy(gt['nma_coords'])
        sim = True
    pred_nma_coords = to_numpy(pred['nma_coords'])
    if config.atomic_global_nma:
        pred_nma_coords += to_numpy(pred['global_nma_coords'])
    if config.remove_nma_outliers:
        pred_nma_coords, filter_idx = _filter_outliers(pred_nma_coords, config.outliers_sigma_num)
        if sim:
            gt_nma_coords = np.delete(gt_nma_coords, filter_idx, 0)
    for i in range(pred_nma_coords.shape[-1]):
        pred_nma = pred_nma_coords[..., i]
        if pred_nma.ndim > 1:
            pred_nma = np.squeeze(pred_nma)
        if sim:
            gt_nma = gt_nma_coords[..., i]
            plt.figure()
            plt.title(f'NMA Mode {i + 1}')
            plt.scatter(gt_nma, pred_nma, alpha=0.5, label='prediction')
            plt.plot(gt_nma, gt_nma, color='r', label="perfect")
            plt.legend()
            plt.savefig(os.path.join(config.eval_dir, f'pred_corr_nma{i + 1}.png'))
        plt.figure()
        plt.title(f'NMA Mode {i + 1}')
        if sim:
            plt.hist(gt_nma, bins=50, density=True, label='GT')
        plt.hist(pred_nma, bins=100, histtype='step', density=True, label='Prediction')
        plt.legend()
        plt.savefig(os.path.join(config.eval_dir, f'pred_dist_nma{i + 1}.png'))


def sample_latent(pred, config, dataset):
    def find_nearest(samples):
        return z[[np.argmin(np.linalg.norm(s - decomp, axis=-1)) for s in samples]]

    if config.atomic_nma_number_modes < 2:
        sampled_pred = pred['nma_coords'][::pred['nma_coords'].shape[0] // config.sample_num]
        sampled_pred, _ = torch.sort(sampled_pred, dim=0)
    else:
        pred_nma_coords = to_numpy(pred['nma_coords'])
        if config.atomic_global_nma:
            pred_nma_coords += to_numpy(pred['global_nma_coords'])
        if config.remove_nma_outliers:
            pred_nma_coords, filter_idx = _filter_outliers(pred_nma_coords, config.outliers_sigma_num)
            dataset = torch.utils.data.Subset(dataset, list(set(np.arange(len(dataset))) - set(filter_idx)))
        z = np.squeeze(pred_nma_coords)
        if config.latent_decomposition == 'PCA':
            pca = PCA(z.shape[1])
            decomp = pca.fit_transform(z)
            scores = pca.explained_variance_ratio_
            inverse = pca.inverse_transform
        elif config.latent_decomposition == 'ICA':
            _, _, decomp, scores = traj2ic(z, z.shape[1])
            inverse = find_nearest

        ii, jj = config.sample_axes

        plt.figure()
        plt.hexbin(decomp[:, ii], decomp[:, jj], mincnt=1, gridsize=50)
        plt.xlabel('Axes{} ({:3f})'.format(ii + 1, scores[ii]))
        plt.ylabel('Axes{} ({:3f})'.format(jj + 1, scores[jj]))
        s = np.zeros((config.sample_num, z.shape[1]))
        si = np.random.choice(decomp.shape[0], config.sample_num, replace=False)
        s[:, ii] = decomp[si, ii]
        s[:, jj] = decomp[si, jj]
        s = s[np.argsort(s[:, ii])]
        plt.scatter(s[:, ii], s[:, jj], c=np.arange(len(s)), marker='x', cmap='hsv')
        plt.savefig(os.path.join(config.eval_dir, f'{config.latent_decomposition}_Ax{ii + 1}_vs_Ax{jj + 1}.png'))
        x = inverse(s)

        sampled_pred = torch.as_tensor(x, dtype=torch.float32)[:, None, :]

    print(sampled_pred.shape)
    print("Initializing atomic model ...")
    atomic_model = AtomicModel(config.atomic_pdb, config.atomic_clean_pdb, config.atomic_center,
                               pdb_out=os.path.join(config.eval_dir, 'curated_gemmi.pdb'))
    atomic_model = DynamicsModelNMA(atomic_model, atomic_clean_pdb=config.atomic_clean_pdb,
                                    atomic_cg_selection=config.atomic_cg_selection,
                                    atomic_nma_cutoff=config.atomic_nma_cutoff,
                                    atomic_nma_gamma=config.atomic_nma_gamma,
                                    atomic_nma_number_modes=config.atomic_nma_number_modes,
                                    atomic_nma_pkl=config.atomic_nma_pkl if "atomic_nma_pkl" in config else None,
                                    by_chain=config.atomic_nma_by_chain)
    prody_model = read_prody_model(os.path.join(config.eval_dir, 'curated_gemmi.pdb'))
    print("Done.")
    print("Sampling atomic models ...")
    pred_coords = atomic_model(sampled_pred)[0]
    print("Done.")
    print("Saving atomic models to PDB ...")
    traj = prody_model.copy()
    for pred_coordset in pred_coords:
        traj.addCoordset(pred_coordset)
    write_prody_model(os.path.join(config.eval_dir, "pred_trajectory.pdb"), traj)
    print("Done.")

    if config.cluster_num > 1:
        print("Clustering latent space ...")
        if config.latent_clustering == "GMM":
            clf = GaussianMixture(n_components=config.cluster_num)
        elif config.latent_clustering == "KMeans":
            clf = KMeans(n_clusters=config.cluster_num)
        elif config.latent_clustering == "DBSCAN":
            clf = DBSCAN(n_jobs=-1)
        labels = clf.fit_predict(decomp)
        config.cluster_num = len(set(labels)) - (1 if -1 in labels else 0)
        sns.jointplot(x=decomp[:, ii], y=decomp[:, jj], kind="scatter", alpha=0.25, hue=labels)
        plt.savefig(os.path.join(config.eval_dir, f'{config.latent_decomposition}_clusters.png'))
        for c in range(config.cluster_num):
            root_dir = os.path.join(config.eval_dir, f'cluster_{c}')
            cond_mkdir(root_dir)
            relative_mrcs_path_prefix = 'Particles/'
            mrcs_dir = os.path.join(root_dir, relative_mrcs_path_prefix)
            cond_mkdir(mrcs_dir)

            ci = np.argwhere(labels == c)
            if ci.shape[0] == 0:
                continue
            sub_dataset = torch.utils.data.Subset(dataset, np.squeeze(ci))
            print(f"Creating starfile for cluster {c} ...")
            create_starfile(DataLoader(sub_dataset, shuffle=False, batch_size=config.val_chunk_sz),
                            config, root_dir, relative_mrcs_path_prefix, f'cluster_{c}')


def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, is_config_file=True,
                        help='Path to config file.')
    parser.add_argument('--checkpoint_path', required=True, help='Checkpoint to trained model.')
    parser.add_argument('--plot_nma_dist', type=int, default=1,
                        help='Save plots for NMA predicted distribution(s).')
    parser.add_argument('--load_evaluation', type=int, default=0,
                        help='Whether to load a previous evaluation form pytorch pickle files or run the checkpoint.'
                             'Default 0, i.e. run the checkpoint')
    parser.add_argument('--remove_nma_outliers', type=int, default=0,
                        help='Remove values outside `outliers_sigma_num` sigma of the predicted nma distributions.')
    parser.add_argument('--outliers_sigma_num', type=float, default=6.,
                        help='The sigma multiplier which is used to filter outliers. Default 6 sigma')
    parser.add_argument('--latent_decomposition', type=str, choices=['PCA', 'ICA'],
                        default='PCA', help='Decomposition method to use for latent space visualization and sampling.')
    parser.add_argument('--latent_clustering', type=str, choices=['DBSCAN', 'KMeans', 'GMM'],
                        default='GMM', help='Clustering method to use for latent space.')
    parser.add_argument('--sample_axes', type=int, nargs='*', default=[0, 1],
                        help='Sampling predictions from decomposed latent space. '
                             'Default [0, 1] to sample axis 0 (eg PC1) and plot axis 0 vs axis 1 (eg PC1 vs PC2).')
    parser.add_argument('--sample_num', type=int, default=10,
                        help='Number of predictions to sample from a decomposition of the predicted NMA latent space.')
    parser.add_argument('--cluster_num', type=int, default=2,
                        help='Number of clusters for the decomposition of the predicted NMA latent space. A starfile '
                             'will be created for each clusters with the input particle images within it.')
    parser.add_argument('--thread_num', type=int, default=4,
                        help='Number of threads to use for the dataloader.')
    init_config(parser)  # use the default arguments from main.init_config to stay synchronized
    config = parser.parse_args()
    config.map_shape = [config.side_len] * 3

    # Create root directory where models, logs and config files will be written
    config.root_dir = os.path.join(config.log_dir, config.experiment_name)
    if not cond_mkdir(config.root_dir):
        print(f"Error: cannot create root path.")
        return -1

    config.eval_dir = os.path.join(config.root_dir, 'eval')
    cond_mkdir(config.eval_dir)

    print(f"Launching evaluation for {config.experiment_type}")
    dataset = None
    if (config.experiment_type == 'exp_relion_reconstruct'):
        star_file = config.val_relion_star_file if config.val_relion_star_file else config.relion_star_file
        dataset = RelionDataLoader(relion_path=config.relion_path,
                                   relion_star_file=star_file,
                                   relion_invert_hand=config.relion_invert_hand,
                                   atomic_nma_number_modes=config.atomic_nma_number_modes)
        config.side_len, \
        config.kV, \
        config.resolution, \
        config.spherical_aberration, \
        config.amplitude_contrast = dataset.get_df_optics_params()
        if not config.val_relion_star_file:
            dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), config.val_sz))
        config.map_shape = [config.side_len] * 3
        config.num_particles = len(dataset)
        config.ctf_size = config.side_len
    elif (config.experiment_type == 'exp_simul_atomic'):
        dataset = AtomicProjectionSimulator(config, projection_sz=config.map_shape,
                                            num_projs=config.val_sz,
                                            ctf_generator=_ctf_generator(config),
                                            noise_generator=_noise_generator(config))

    if config.load_evaluation:
        print("Loading evaluation from files ...")
        total_gt = torch.load(os.path.join(config.eval_dir, 'gt.pt'), map_location=torch.device('cpu'))
        total_input = torch.load(os.path.join(config.eval_dir, 'input.pt'), map_location=torch.device('cpu'))
        total_output = torch.load(os.path.join(config.eval_dir, 'output.pt'), map_location=torch.device('cpu'))
        print("Done.")
    else:
        dataloader = DataLoader(dataset, shuffle=False,
                                batch_size=config.val_chunk_sz,
                                pin_memory=True, num_workers=config.thread_num)

        model = CryoNet(config)
        model.cuda()
        print("Loading checkpoint ...")
        state_dict = torch.load(config.checkpoint_path)
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        print("Done.")
        print("Evaluating model ...")
        with tqdm(total=len(dataloader)) as pbar:
            total_gt, total_input, total_output = evaluate_model(model, dataloader, pbar, True)
        print("Done.")

        print("Saving evaluation outputs ...")
        torch.save(total_gt, os.path.join(config.eval_dir, 'gt.pt'))
        torch.save(total_input, os.path.join(config.eval_dir, 'input.pt'))
        torch.save(total_output, os.path.join(config.eval_dir, 'output.pt'))
        print("Done.")
        del model
        del dataloader

    if total_output['nma_coords'] is not None:
        if config.plot_nma_dist:
            plot_nma(total_gt, total_output, config)
        sample_latent(total_output, config, dataset)

    return 0, 'Evaluation successful'


if __name__ == '__main__':
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Evaluation failed.'

    print(status_message)
    exit(retval)
