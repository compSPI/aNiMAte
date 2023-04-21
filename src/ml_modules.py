import torch
from torch import nn
import torch.fft
import numpy as np
import math
import functools
import warnings
from torchvision.models import efficientnet_v2_s
from math import sqrt


# TODO: changed initialization for backward graph 'fan out'
def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')
        if hasattr(m, 'bias'):
            nn.init.uniform_(m.bias, -1, 1)
            # m.bias.data.fill_(0.)


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


def init_weights_uniform(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def first_layer_sine_wavelet_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))
            m.bias.uniform_(-1, 1)


class Sine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        return torch.sin(self.w0 * input)


class SineWavelet(nn.Module):
    def __init__(self, w0=20, num_features=256):
        super().__init__()
        self.w0 = torch.tensor(w0)
        self.sigma = nn.Parameter(torch.rand(num_features, dtype=torch.float32),
                                  requires_grad=True)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        # return torch.sin(self.w0*input)*torch.exp(-self.sigma * input**2)
        return torch.sin(self.w0 * input) * torch.relu(1. - self.sigma * torch.abs(input))


class RandSine(nn.Module):
    def __init__(self, mu_w0=50, std_w0=40, num_features=256):  # 30, 29
        super().__init__()
        self.w0 = mu_w0 + 2. * std_w0 * (torch.rand(num_features, dtype=torch.float32) - .5).cuda()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input) - self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class ReQLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_sq = 1 ** 2

    def forward(self, input):
        r_input = torch.relu(input)
        return self.p_sq * (torch.sqrt(1. + r_input ** 2 / self.p_sq) - 1.)


''' FCNet'''


def layer_factory(layer_type):
    layer_dict = \
        {'relu': (nn.ReLU(inplace=True), init_weights_normal),
         'lrelu': (nn.LeakyReLU(inplace=True), init_weights_normal),
         'reqlu': (ReQLU, init_weights_normal),
         'sigmoid': (nn.Sigmoid(), init_weights_xavier),
         'fsine': (Sine(), first_layer_sine_init),
         'sine': (Sine(), sine_init),
         'randsine': (RandSine(), sine_init),
         'tanh': (nn.Tanh(), init_weights_xavier),
         'htanh': (nn.Hardtanh(), init_weights_xavier),
         'ssign': (nn.Softsign(), init_weights_xavier),
         'selu': (nn.SELU(inplace=True), init_weights_selu),
         'gelu': (nn.GELU(), init_weights_selu),
         'silu': (nn.SiLU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         'softplus': (nn.Softplus(), init_weights_normal),
         'msoftplus': (MSoftplus(), init_weights_normal),
         'elu': (nn.ELU(), init_weights_elu)
         }
    return layer_dict[layer_type]


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, features, out_features,
                 nonlinearity='relu', last_nonlinearity=None,
                 batch_norm=False, equalized=False, dropout=False):
        super().__init__()

        # Create hidden features list
        self.hidden_features = [int(in_features)]
        if features != []:
            self.hidden_features.extend(features)
        self.hidden_features.append(int(out_features))

        self.net = []
        for i in range(len(self.hidden_features) - 1):
            # Not the last
            hidden = False
            if i < len(self.hidden_features) - 2:
                nl = layer_factory(nonlinearity)[0]
                init = layer_factory(nonlinearity)[1]
                hidden = True
            # The last layer
            else:
                if last_nonlinearity is not None:
                    nl = layer_factory(last_nonlinearity)[0]
                    init = layer_factory(last_nonlinearity)[1]
            layer = nn.Linear(self.hidden_features[i], self.hidden_features[i + 1])

            if hidden:
                init(layer)
                self.net.append(layer)
                self.net.append(nl)
                if dropout:
                    self.net.append(nn.Dropout(p=0.5))
                if batch_norm:
                    self.net.append(nn.BatchNorm1d(num_features=self.hidden_features[i + 1]))
            else:
                # init_weights_normal(layer)
                self.net.append(layer)
                if last_nonlinearity is not None:
                    self.net.append(nl)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


''' PE '''


class PositionalEncoding(nn.Module):  # MetaModule):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=False, gaussian_variance=38):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)

            # TODO: Normalization?

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                # self.normalization = nn.Parameter(1/self.frequency_bands.clone(), requires_grad=False)
                self.normalization = torch.tensor(1 / self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx] * func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


''' CNNs '''


def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm,
                 equalized=False, dropout=False, attention=False):
        super(DoubleConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.attention = attention
        self.conv1 = conv3x3(in_size, out_size)
        self.conv2 = conv3x3(out_size, out_size)

        self.relu = nn.ReLU(inplace=True)
        if dropout:
            self.drop = nn.Dropout(p=0.2)
        if batch_norm:
            self.bn = nn.InstanceNorm2d(out_size, affine=True)  # nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.bn(out)
        if self.dropout:
            out = self.drop(out)

        return out


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_channels=[64, 64, 64],
                 padding=True, batch_norm=True, max_pool=True, dropout=False):
        super(CNNEncoder, self).__init__()

        self.in_channels = in_channels
        self.depth = len(feature_channels)
        self.max_pool = max_pool
        self.feature_channels = feature_channels

        assert (len(feature_channels) > 0,
                'Error CNNEncoder must have at least 1 conv layer')

        print(f"feature_channels={feature_channels}")

        self.input_bn = nn.BatchNorm2d(in_channels)
        self.net = []
        prev_channels = in_channels

        for channels in self.feature_channels:
            self.net.append(
                DoubleConvBlock(prev_channels, channels,
                                padding, batch_norm, dropout)
            )
            if self.max_pool:
                self.net.append(
                    nn.AvgPool2d(kernel_size=2)
                )
            prev_channels = channels

        self.net = nn.Sequential(*self.net)

    def get_out_shape(self, h, w):
        out = self.forward(torch.rand(1, self.in_channels, h, w))
        return out['latent_code'].shape[1]

    def forward(self, input):
        input = self.input_bn(input)
        out = self.net(input)
        z = torch.flatten(out, start_dim=1)
        return {'latent_code': z}


class EfficientNetV2Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(EfficientNetV2Encoder, self).__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(list(efficientnet_v2_s().children())[0])
        org_input_layer = self.net[0][0][0]
        self.net[0][0][0] = nn.Conv2d(in_channels,
                                      org_input_layer.out_channels,
                                      kernel_size=org_input_layer.kernel_size,
                                      stride=org_input_layer.stride,
                                      padding=org_input_layer.padding,
                                      bias=org_input_layer.bias)

    def get_out_shape(self, h, w):
        out = self.forward(torch.rand(1, self.in_channels, h, w))
        return out['latent_code'].shape[1]

    def forward(self, input):
        out = self.net(input)
        z = torch.flatten(out, start_dim=1)
        return {'latent_code': z}


class VAEEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 latent_dim=1024,
                 hidden_dims=[32, 64, 128, 256, 512],
                 in_dims=[128, 128]):
        super(VAEEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential
                           (nn.Conv2d(in_channels, out_channels=h_dim,
                                      kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.LeakyReLU())
                           )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        dummy_input = torch.zeros(1, self.in_channels, in_dims[0], in_dims[1])
        out_dims = torch.flatten(self.encoder(dummy_input), start_dim=1).shape[1]
        self.fc_mu = nn.Linear(out_dims, latent_dim)
        self.fc_var = nn.Linear(out_dims, latent_dim)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        """
        Reparametrization trick for Gaussian distributions
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_out_shape(self, h, w):
        return self.latent_dim

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return {'latent_code': z,
                'latent_mu': mu,
                'latent_logvar': log_var}
