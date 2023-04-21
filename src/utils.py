import os
import matplotlib.backends.backend_agg as plt_backend_agg
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import FastICA
import numpy as np
import torch

import inspect


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pshape(tensor, name):
    callerframerecord = inspect.stack()[1]  # 0 represents this line
    # 1 represents line at caller
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)

    print(f"{info.filename}:{info.lineno} {name}={tensor.shape}")


def cond_mkdir(path):
    if not os.path.exists(path):  # path does not exist, create it
        os.makedirs(path)
        return True

    return True
    '''
    else:
        val = input(f"Directory {path} exists, do you want to overwrite it [Y]/n? ")
        if val == '' or val == 'Y': # path exists, overwrite
            shutil.rmtree(path)
            os.makedirs(path)
            return True
        else:   # path exists, return
            return False
        '''


def to_numpy(t):
    return t.detach().cpu().numpy()


def render_to_rgb(figure):
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    return image_chw


def generate_rotmats_video(rotmat_gt, rotmat_pred, colors):
    from matplotlib import pyplot as plt
    from scipy.spatial.transform import Rotation as R

    ''' align the two set of vectors'''
    num_rots_gt = rotmat_gt.shape[0]
    unitvec_gt = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(rotmat_gt.device)
    rot_unitvecs_gt = torch.bmm(unitvec_gt.repeat(num_rots_gt, 1, 1), rotmat_gt)

    num_rots_pred = rotmat_pred.shape[0]
    unitvec_pred = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(rotmat_pred.device)
    rot_unitvecs_pred = torch.bmm(unitvec_pred.repeat(num_rots_pred, 1, 1), rotmat_pred)

    rot_unitvecs_gt = to_numpy(rot_unitvecs_gt.squeeze())
    rot_unitvecs_pred = to_numpy(rot_unitvecs_pred.squeeze())
    relativeR, rmsd = R.align_vectors(rot_unitvecs_gt, rot_unitvecs_pred)
    rot_unitvecs_pred = relativeR.apply(rot_unitvecs_pred)
    diff = np.degrees(np.arccos([np.dot(x[0], x[1]) for x in zip(rot_unitvecs_gt, rot_unitvecs_pred)]))

    rot_unitvecs = np.concatenate([rot_unitvecs_gt, rot_unitvecs_pred], axis=0)

    # generate latitutde/longitude lines along unit sphere
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    # plot coordinates and unit sphere mesh
    fig_imgs = []

    def plot_scatter3d(ax, pts, colors, elev, azim):
        ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, alpha=0.2)
        ax.scatter(pts[:, 0],
                   pts[:, 1],
                   pts[:, 2],
                   s=10, c=colors.detach().numpy(), zorder=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.view_init(elev=elev, azim=azim)
        plt.axis('off')
        plt.tight_layout()
        return ax

    for azim in range(60, 120):
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

        plot_scatter3d(ax, rot_unitvecs, colors,
                       elev=30., azim=azim)

        # Draw lines for first part of data
        pts1 = rot_unitvecs_gt
        pts2 = rot_unitvecs_pred
        segments = [(pts1[i, :], pts2[i, :]) for i in range(pts1.shape[0])]
        lc = Line3DCollection(segments)
        ax.add_collection3d(lc)

        fig_imgs.append(torch.from_numpy(render_to_rgb(fig))[None, ...])
        plt.close(fig)

    for azim in reversed(range(60, 120)):
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

        plot_scatter3d(ax, rot_unitvecs, colors,
                       elev=30., azim=azim)

        fig_imgs.append(torch.from_numpy(render_to_rgb(fig))[None, ...])
        plt.close(fig)

    video = torch.cat(fig_imgs, dim=0)
    return video, diff


def ave_score(X, n, niter=100, fun='logcosh'):
    # notice that X components are column-oriented
    score_ave = []
    score_var = []
    for i in np.arange(0, n, 1):
        score_tmp = []
        for j in np.arange(0, niter, 1):
            score_tmp.append(negent_score(X[:, i], fun))
        score_ave.append(np.mean(score_tmp))
        score_var.append(np.var(score_tmp))
    return score_ave, score_var


def negent_score(X, fun='logcosh'):
    # We compute J(X) = [E(G(X)) - E(G(Xgauss))]**2
    # We consider X (and Xgauss) to be white, in the sense that E(X,X.T)=I
    # The expectation being approximated by the sample mean in our case: np.dot(X,X.T)/n=I
    # In practice, we assume that X has already been normalized by its length [np.dot(X,X.T)=I]
    # so we rescale by np.sqrt(n) before we take the expectation value of G(X).
    length = len(X)
    Xscale = X * np.sqrt(length)
    Xgauss = np.random.randn(length)
    if (fun == 'logcosh'):
        n1 = np.mean(f_logcosh(Xscale))  # np.sum(f_logcosh(Xscale))
        n2 = np.mean(f_logcosh(Xgauss))  # np.sum(f_logcosh(Xgauss))
    elif (fun == 'exp'):
        n1 = np.mean(f_exp(Xscale))  # np.sum(f_exp(Xscale))
        n2 = np.mean(f_exp(Xgauss))  # np.sum(f_exp(Xgauss))
    elif (fun == 'rand'):
        n1 = np.mean(f_logcosh(Xgauss))  # np.sum(f_logcosh(Xgauss))
        n2 = 0
    negent = (n2 - n1) ** 2
    return negent


def f_logcosh(X):
    return np.log(np.cosh(X))


def f_exp(X):
    return -np.exp(-(X ** 2) / 2)


def traj2ic(x_pc, n_components=1, fun='logcosh', algo='parallel', negent_sort=True):
    ica = FastICA(whiten=True, algorithm=algo, fun=fun)
    x_ic = ica.fit_transform(x_pc)
    m_ic = ica.mixing_
    v_ic = ica.components_
    if (negent_sort):
        Jscore, Jtmp = ave_score(x_ic, n_components)
        index = np.argsort(Jscore)[::-1]
        x_ic = x_ic[:, index]
        m_ic = m_ic[:, index]
        v_ic = v_ic[index, :]
    return v_ic, m_ic, x_ic, np.array(Jscore)[index]
