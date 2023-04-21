import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import mcubes
import mrcfile
from utils import generate_rotmats_video, to_numpy

''' Some utils '''


def normalize_proj(proj):
    # assumes proj is N,C,H,W
    num_projs = proj.shape[0]

    vmin, _ = torch.min(proj.reshape(num_projs, -1), dim=-1)
    vmin = vmin[:, None, None, None]
    vmax, _ = torch.max(proj.reshape(num_projs, -1), dim=-1)
    vmax = vmax[:, None, None, None]
    proj_norm = (proj - vmin) / (vmax - vmin)

    return proj_norm


def visualize_rot(rotmat_gt, rotmat_pred, writer, global_step, summary_prefix):
    color_gt = torch.tensor([[1, 0, 0]])  # red
    color_pred = torch.tensor([[0, 0, 1]])  # blue

    all_colors = torch.cat((color_gt.repeat(rotmat_gt.shape[0], 1),
                            color_pred.repeat(rotmat_gt.shape[0], 1)), dim=0)

    video_rots, err = generate_rotmats_video(rotmat_gt, rotmat_pred,
                                             all_colors)
    writer.add_video(f"{summary_prefix}: Rotations", video_rots[None, ...], global_step=global_step, fps=25)
    return err


def generate_mesh_from_volume(volume, name):
    vertices, triangles = mcubes.marching_cubes(volume, 1e-2)
    mcubes.export_obj(vertices, triangles, f"{name}.obj")


''' The summary function '''


def write_summary(model, gt, model_output,
                  writer, total_steps, root_dir_path,
                  write_mrc=False, summary_prefix='train',
                  write_mesh=False):
    if 'module' in dir(model):
        if type(model.module).__name__ == 'CryoNet':
            model = model.module
    # Output the training set prediction and gt projs
    proj_gt = gt['proj']
    proj_pred = model_output['proj']
    # above, but there could be more.

    ''' Plot non normalized figure with colorbar '''
    fig = plt.figure(dpi=96)
    plt.imshow(to_numpy(gt['proj'][0, ...].squeeze()), cmap='plasma')
    plt.colorbar()
    plt.tight_layout()
    writer.add_figure(f"{summary_prefix}: GT (colorbar)", fig, global_step=total_steps)

    ''' Plot the rotations '''
    rotmat_gt = gt['rotmat']
    rotmat_pred = model_output['rotmat']
    rot_err = visualize_rot(rotmat_gt, rotmat_pred, writer, total_steps, summary_prefix)
    writer.add_scalar(f'{summary_prefix}: pred_rot_mae (degree)', rot_err.mean(), global_step=total_steps)

    ''' Visualize a single image '''
    proj_pred = normalize_proj(proj_pred)
    proj_gt = normalize_proj(proj_gt)

    idx = 0
    writer.add_image(f"{summary_prefix}: GT", proj_gt[idx, ...], global_step=total_steps)
    writer.add_image(f"{summary_prefix}: Pred", proj_pred[idx, ...], global_step=total_steps)

    if "val" not in summary_prefix:  # costly with large validation sets
        ''' Visualize multiple images '''
        # add_videos takes a B,T,C,H,W input, here we simply use it to tile images using T=1 and B=N
        writer.add_video(f"{summary_prefix}: GT (multi)", proj_gt[:, None, :, :, :], global_step=total_steps)
        writer.add_video(f"{summary_prefix}: Pred (multi)", proj_pred[:, None, :, :, :], global_step=total_steps)

    ''' Visualize mesh'''
    volume = model.global_map.make_volume().cpu().numpy()
    writer.add_histogram(f" Volume values", values=torch.from_numpy(volume).reshape(1, -1),
                         global_step=total_steps)
    if write_mesh:
        isoline = 1e-2
        vertices, triangles = mcubes.marching_cubes(volume, isoline)
        vertices = torch.from_numpy(vertices)[None, :, :]
        triangles = torch.from_numpy(triangles.astype(np.int32))[None, :, :]  # not clean: triangles are uint64
        print(f"vert={vertices.shape}, triangles={triangles.shape}")
        writer.add_mesh(f"Mesh iso={isoline}", vertices=vertices, faces=triangles)

    ''' Report error in nma_coordinates if atomic reconstruction '''
    if model_output['nma_coords'] is not None:
        writer.add_histogram(f"{summary_prefix}: nma_coords values", values=model_output['nma_coords'].reshape(1, -1),
                             global_step=total_steps)
        gt_nma_coords = None
        if 'nma_alphas' in gt.keys():
            num_atoms = model.atomic_model.get_atomic_model()[0].shape[1]
            n_modes = model_output['nma_coords'].shape[2]
            gt_nma_coords = gt['nma_alphas'] / math.sqrt(num_atoms / n_modes)
        elif 'nma_coords' in gt.keys():
            gt_nma_coords = gt['nma_coords']
        if gt_nma_coords is not None:
            writer.add_scalar(f"{summary_prefix}: nma_coords MSE",
                              ((model_output['nma_coords'] - gt_nma_coords) ** 2).mean(),
                              global_step=total_steps)

            gt_nma_coords = to_numpy(gt_nma_coords)
            pred_nma_coords = to_numpy(model_output['nma_coords'])
            plot_length = 3
            n_chains, n_modes = gt_nma_coords.shape[1], gt_nma_coords.shape[2]
            fig, axs = plt.subplots(n_modes, n_chains, dpi=96, figsize=(n_chains * plot_length, n_modes * plot_length))

            for xi in range(n_modes):
                for xj in range(n_chains):
                    if n_modes == 1 and n_chains == 1:
                        axs.scatter(gt_nma_coords[:, xj, xi], pred_nma_coords[:, xj, xi], alpha=0.5, label='prediction')
                        axs.plot(gt_nma_coords[:, xj, xi], gt_nma_coords[:, xj, xi], color='r', label="perfect")
                        axs.set_aspect(1.0)
                        axs.legend(bbox_to_anchor=(1.1, 0.8))
                        axs.set_ylabel(f"NMA Mode {xi}")
                        axs.set_xlabel(f"Chain {xj}")
                    elif n_modes == 1 and n_chains > 1:
                        axs[xj].scatter(gt_nma_coords[:, xj, xi], pred_nma_coords[:, xj, xi], alpha=0.5,
                                        label='prediction')
                        axs[xj].plot(gt_nma_coords[:, xj, xi], gt_nma_coords[:, xj, xi], color='r', label="perfect")
                        axs[xj].set_aspect(1.0)
                        if xi == 0 and xj == n_chains - 1: axs[xj].legend(bbox_to_anchor=(1.1, 0.8))
                        if xj == 0: axs[xj].set_ylabel(f"NMA Mode {xi}")
                        if xi == n_modes - 1: axs[xj].set_xlabel(f"Chain {xj}")
                    elif n_chains == 1 and n_modes > 1:
                        axs[xi].scatter(gt_nma_coords[:, xj, xi], pred_nma_coords[:, xj, xi], alpha=0.5,
                                        label='prediction')
                        axs[xi].plot(gt_nma_coords[:, xj, xi], gt_nma_coords[:, xj, xi], color='r', label="perfect")
                        axs[xi].set_aspect(1.0)
                        if xi == 0: axs[xi].legend(bbox_to_anchor=(1.1, 0.8))
                        if xj == 0: axs[xi].set_ylabel(f"NMA Mode {xi}")
                        if xi == n_modes - 1: axs[xi].set_xlabel(f"Chain {xj}")
                    else:
                        axs[xi, xj].scatter(gt_nma_coords[:, xj, xi], pred_nma_coords[:, xj, xi], alpha=0.5,
                                            label='prediction')
                        axs[xi, xj].plot(gt_nma_coords[:, xj, xi], gt_nma_coords[:, xj, xi], color='r', label="perfect")
                        axs[xi, xj].set_aspect(1.0)
                        if xi == 0 and xj == n_chains - 1: axs[xi, xj].legend(bbox_to_anchor=(1.1, 0.8))
                        if xj == 0: axs[xi, xj].set_ylabel(f"NMA Mode {xi}")
                        if xi == n_modes - 1: axs[xi, xj].set_xlabel(f"Chain {xj}")

            writer.add_figure(f"{summary_prefix}: nma_prediction", fig, global_step=total_steps)

    ''' Report error in shifts '''
    if model_output['shifts'] is not None:
        shift_x = model_output['shifts'][..., 0, None]
        shift_y = model_output['shifts'][..., 1, None]
        writer.add_histogram(f"{summary_prefix}: shift_x values", values=shift_x.reshape(1, -1),
                             global_step=total_steps)
        writer.add_histogram(f"{summary_prefix}: shift_y values", values=shift_y.reshape(1, -1),
                             global_step=total_steps)
        if 'shift_x' in gt.keys():
            writer.add_scalar(f"{summary_prefix}: shift_x MAE",
                              torch.abs(shift_x - gt['shift_x']).mean(), global_step=total_steps)
        if 'shift_y' in gt.keys():
            writer.add_scalar(f"{summary_prefix}: shift_y MAE",
                              torch.abs(shift_y - gt['shift_y']).mean(), global_step=total_steps)

    ''' Report predicted projection scaling '''
    if 'global_proj_scaling' in dir(model):
        global_proj_scaling = model.global_proj_scaling
        if global_proj_scaling.shape[0] > 1:
            global_proj_scaling = global_proj_scaling[0]
        writer.add_scalar("global_proj_scaling", global_proj_scaling, global_step=total_steps)

    ''' Report predicted global pose '''
    if 'global_s2s2' in dir(model):
        writer.add_histogram(f"{summary_prefix}: global_s2s2", values=model.global_s2s2, global_step=total_steps)
    if 'global_shift' in dir(model):
        writer.add_histogram(f"{summary_prefix}: global_shift", values=model.global_shift, global_step=total_steps)

    ''' Report predicted global nma '''
    if 'global_nma_coords' in dir(model):
        writer.add_histogram(f"{summary_prefix}: global_nma_coords",
                             values=model.global_nma_coords, global_step=total_steps)

    ''' Report gradients '''
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'{name}_gradient', param.grad, global_step=total_steps)

    ''' Write the MRC '''
    if write_mrc:
        print(f"Outputing volume of size {volume.shape}")
        filename = os.path.join(root_dir_path, 'reconstruction.mrc')
        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(volume)
            mrc.voxel_size = model.ctf.resolution
    else:
        print("No MRC file will be built.")
