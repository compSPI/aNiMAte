import numpy as np
import torch
import torch.fft
from torch import nn
import os
from pytorch3d.transforms import euler_angles_to_matrix, quaternion_to_matrix, rotation_6d_to_matrix
from ml_modules import FCBlock, CNNEncoder, EfficientNetV2Encoder, VAEEncoder
from ctf_utils import CTF, primal_to_fourier_2D, fourier_to_primal_2D
from atomic_utils import AtomicModel
from dynamics_utils import DynamicsModelNMA, DynamicsModelNone
from shift_utils import Shift
from volume_utils import ExplicitAtomicVolume

''' CryoNet '''


class CryoNet(nn.Module):
    def __init__(self, config):
        super(CryoNet, self).__init__()

        self.config = config
        self.code_dim = 0
        self.cryonet = []

        self.atomic_model = AtomicModel(config.atomic_pdb, config.atomic_clean_pdb, config.atomic_center,
                                        pdb_out=os.path.join(config.root_dir, 'curated_gemmi.pdb'))
        if config.dynamic_model == 'nma':
            self.atomic_model = DynamicsModelNMA(self.atomic_model, atomic_clean_pdb=config.atomic_clean_pdb,
                                                 atomic_cg_selection=config.atomic_cg_selection,
                                                 atomic_nma_cutoff=config.atomic_nma_cutoff,
                                                 atomic_nma_gamma=config.atomic_nma_gamma,
                                                 atomic_nma_number_modes=config.atomic_nma_number_modes,
                                                 atomic_nma_pkl=config.atomic_nma_pkl,
                                                 log_dir=config.root_dir, by_chain=config.atomic_nma_by_chain)
        else:
            self.atomic_model = DynamicsModelNone(self.atomic_model)

        if config.atomic_global_nma:
            self.global_nma_coords = nn.Parameter(torch.zeros(self.atomic_model.n_chains,
                                                              self.config.atomic_nma_number_modes),
                                                  requires_grad=True)
        if config.atomic_global_pose:
            self.global_s2s2 = nn.Parameter(torch.Tensor([[1, 0, 0, 0, 1, 0]]).cuda(), requires_grad=True)
            self.global_shift = nn.Parameter(torch.zeros((1, 3)), requires_grad=True)

        if config.data_loss == "L2":
            ''' Learn a global image scaling '''
            self.global_proj_scaling = nn.Parameter(config.proj_scaling_init * torch.ones(1), requires_grad=True)
        self.global_map = ExplicitAtomicVolume(self.atomic_model,
                                               sidelen=config.map_shape[0],
                                               pixel_size=config.resolution,
                                               log_dir=config.root_dir)

        if config.so3_refinement:
            self.refine_s2s2 = nn.Parameter(torch.Tensor([[1, 0, 0, 0, 1, 0]]).cuda().repeat(config.num_particles, 1),
                                            requires_grad=True)

        ''' Encoder '''
        self.encoder = self._init_encoder(config)
        latent_code_size = self.encoder.get_out_shape(*config.map_shape[0:2])

        ''' FCNet orientation regressor and orientation interpretation '''
        if self.config.so3_parameterization == 'euler':
            self.orientation_dims = 3
            self.last_nonlinearity = 'tanh'
            self.latent_to_rot3d_fn = \
                lambda x: euler_angles_to_matrix(np.pi * x, convention="ZYZ")
        elif self.config.so3_parameterization == 'quaternion':
            self.orientation_dims = 3
            self.last_nonlinearity = 'sigmoid'
            self.latent_to_rot3d_fn = \
                lambda x: quaternion_to_matrix(self._project_quat(x))
        elif self.config.so3_parameterization == 's2s2':
            self.orientation_dims = 6
            self.last_nonlinearity = 'tanh'
            self.latent_to_rot3d_fn = rotation_6d_to_matrix
        elif self.config.so3_parameterization == 'gt':
            self.latent_to_rot3d_fn = None

        if self.latent_to_rot3d_fn is not None:
            # We split the regressor in 2 to have access to the latent code
            self.orientation_encoder = FCBlock(in_features=latent_code_size + self.code_dim,
                                               out_features=config.regressor_orientation_layers[-1],
                                               features=config.regressor_orientation_layers[:-1],
                                               nonlinearity='relu', last_nonlinearity='relu',
                                               batch_norm=config.encoder_batch_norm,
                                               equalized=config.encoder_lr_equalization,
                                               dropout=config.encoder_dropout)
            self.orientation_regressor = FCBlock(in_features=config.regressor_orientation_layers[-1],
                                                 out_features=self.orientation_dims,
                                                 features=[],
                                                 nonlinearity='relu',
                                                 last_nonlinearity=self.last_nonlinearity,
                                                 batch_norm=config.encoder_batch_norm,
                                                 equalized=config.encoder_lr_equalization,
                                                 dropout=config.encoder_dropout)

        if self.config.dynamic_model == 'nma':
            self.conformation_encoder = FCBlock(in_features=latent_code_size + self.code_dim,
                                                out_features=config.atomic_nma_number_modes * self.atomic_model.n_chains,
                                                features=config.regressor_conformation_layers,
                                                nonlinearity='relu', last_nonlinearity=None,
                                                batch_norm=config.encoder_batch_norm,
                                                equalized=config.encoder_lr_equalization,
                                                dropout=config.encoder_dropout)

        if self.config.shift_input == 'encoder':
            self.shift_encoder = FCBlock(in_features=latent_code_size,
                                         out_features=2,
                                         features=config.regressor_shift_layers,
                                         nonlinearity='relu', last_nonlinearity=None,
                                         batch_norm=config.encoder_batch_norm,
                                         equalized=config.encoder_lr_equalization,
                                         dropout=config.encoder_dropout)

        ''' CTF model '''
        self.ctf = CTF(size=config.ctf_size, resolution=config.resolution,
                       kV=config.kV, valueNyquist=config.ctf_valueNyquist, cs=config.spherical_abberation,
                       amplitudeContrast=config.amplitude_contrast, requires_grad=False)

        ''' Shift model'''
        self.shift = Shift(size=config.map_shape[0], resolution=config.resolution)

        if config.mask_2D_diam > 0:
            d2 = config.map_shape[0] / 2
            x = y = np.linspace(-d2 * config.resolution, d2 * config.resolution, config.map_shape[0])
            X, Y = np.meshgrid(x, y)
            self.register_buffer(name='mask', tensor=torch.Tensor(np.sqrt(X ** 2 + Y ** 2) < config.mask_2D_diam / 2))

        print(self)
        print("Parameter #:", sum([np.prod(p.size()) for p in self.parameters()]))

    def forward(self, in_dict):
        # This is the projection from the simulator
        proj = in_dict['proj']

        if self.config.so3_parameterization == 'gt' and \
                self.config.shift_input == 'gt' and \
                self.config.dynamic_model != 'nma':
            encoding = None
            latent_code = None
        else:
            encoding = self.encoder(proj)
            latent_code = encoding['latent_code']

        # These are the eventual params of the ctf
        ctf_params = {k: in_dict[k] for k in ('defocus_u', 'defocus_v', 'angleAstigmatism')
                      if k in in_dict}

        ''' Encode the input through the CNN '''
        if self.config.so3_parameterization == 'gt':
            pred_rotmat = in_dict['rotmat']
        else:
            # Pass it through the Orientation regressor
            latent_code_orient = self.orientation_encoder(latent_code)
            latent_code_prerot = self.orientation_regressor(latent_code_orient)
            # Interpret the latent code as a rotation
            pred_rotmat = self.latent_to_rot3d_fn(latent_code_prerot)

        if self.config.so3_refinement:
            refine_rotmat = rotation_6d_to_matrix(self.refine_s2s2[in_dict['idx']])
            pred_rotmat = torch.bmm(pred_rotmat, refine_rotmat)

        pred_shifts = None
        if self.config.shift_input == 'gt':
            shift_params = {k: in_dict[k].reshape(-1) for k in ('shift_x', 'shift_y')
                            if k in in_dict}
        elif self.config.shift_input == 'encoder':
            pred_shifts = self.shift_encoder(latent_code)
            shift_params = {'shift_x': pred_shifts[..., 0].reshape(-1),
                            'shift_y': pred_shifts[..., 1].reshape(-1)}

        global_nma = None

        if self.config.dynamic_model == 'nma':
            pred_nma_coords = self.conformation_encoder(latent_code).reshape(proj.shape[0],
                                                                             self.atomic_model.n_chains,
                                                                             self.config.atomic_nma_number_modes)
        else:
            pred_nma_coords = torch.zeros(proj.shape[0],
                                          1, self.config.atomic_nma_number_modes,
                                          device=proj.device)
        if self.config.atomic_global_nma:
            global_nma = self.global_nma_coords.repeat(proj.shape[0], 1, 1)
        global_pose = {}
        if self.config.atomic_global_pose:
            global_pose['global_rotmat'] = rotation_6d_to_matrix(torch.tanh(self.global_s2s2))
            global_pose['global_shift'] = self.global_shift
        pred_proj = self.global_map(pred_rotmat, pred_nma_coords, global_pose, global_nma)
        if self.config.data_loss == "L2":
            pred_proj /= torch.exp(self.global_proj_scaling)
        nma_eigvals = self.atomic_model.get_eigvals()
        nma_eigvals = None if nma_eigvals is None else nma_eigvals.repeat(proj.shape[0], 1, 1)

        pred_fproj = primal_to_fourier_2D(pred_proj)
        pred_fproj = self.ctf(pred_fproj, ctf_params)
        pred_fproj = self.shift(pred_fproj, shift_params)
        pred_proj = fourier_to_primal_2D(pred_fproj)

        if self.config.mask_2D_diam > 0:
            masks = self.shift(primal_to_fourier_2D(self.mask.repeat(pred_proj.shape[0], 1, 1)[:, None, :, :]),
                               shift_params)
            masks = torch.abs(fourier_to_primal_2D(masks))
            pred_proj *= masks
        else:
            masks = None

        if encoding is not None:
            latent_mu = encoding['latent_mu'] if 'latent_mu' in encoding else None
            latent_logvar = encoding['latent_logvar'] if 'latent_logvar' in encoding else None
        else:
            latent_mu = None
            latent_logvar = None

        return {'rotmat': pred_rotmat,
                'latent_code': latent_code,
                'latent_mu': latent_mu,
                'latent_logvar': latent_logvar,
                'proj': pred_proj.real,
                'proj_imag': pred_proj.imag,
                'fproj': pred_fproj,
                'nma_coords': pred_nma_coords,
                'global_nma_coords': global_nma,
                'nma_eigvals': nma_eigvals,
                'shifts': pred_shifts,
                'masks': masks}

    @staticmethod
    def _project_quat(vec3d):
        u1 = vec3d[:, 0]
        u2 = vec3d[:, 1]
        u3 = vec3d[:, 2]
        return torch.stack([torch.sqrt(1 - u1) * torch.sin(np.pi * u2),
                            torch.sqrt(1 - u1) * torch.cos(np.pi * u2),
                            torch.sqrt(u1) * torch.sin(np.pi * u3),
                            torch.sqrt(u1) * torch.cos(np.pi * u3)], dim=1)

    @staticmethod
    def _init_encoder(config):
        if config.encoder == "CNN":
            return CNNEncoder(in_channels=1,
                              feature_channels=config.encoder_conv_layers,
                              padding=True,
                              batch_norm=config.encoder_batch_norm,
                              max_pool=config.encoder_max_pool,
                              lr_equalization=config.encoder_lr_equalization,
                              dropout=config.encoder_dropout,
                              attention=config.encoder_attention,
                              global_avg_pool=config.encoder_global_avg_pool)
        elif config.encoder == "EfficientNetV2":
            return EfficientNetV2Encoder(in_channels=1)
        elif config.encoder == "VAE":
            return VAEEncoder(in_channels=1,
                              latent_dim=config.encoder_conv_layers[-1],
                              hidden_dims=config.encoder_conv_layers[:-1],
                              in_dims=config.map_shape[0:2])
        else:
            raise NotImplementedError(f"Encoder type {config.encoder} not implemented!")
