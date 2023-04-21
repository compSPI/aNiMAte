import torch
import numpy as np


class Shift(torch.nn.Module):
    """
    A class containing method to shift an image in Fourier domain.
    ...
    Attributes
    ----------
    size : int
            side length of images in pixels
    resolution : float
            physical size of the pixels in Angstrom (A)
    Methods
    -------
    forward(x_fourier, shift_params={}):
    outputs shifted x_fourier depending on shift_params.

    modulate(x_fourier, t_x,t_y):
    outputs modulated (fourier equivalent of shifting in primal domain) input images given in batch wise format.
    The modulation depends on t_x, t_y.

    """

    def __init__(self, size=128, resolution=0.8,
                 shift_x_mean=0, shift_x_stdev=0,
                 shift_y_mean=0, shift_y_stdev=0):
        super(Shift, self).__init__()

        self.size = size
        self.resolution = resolution
        self.frequency = 1. / (self.size * resolution)
        self.shift_x_mean = shift_x_mean
        self.shift_x_stdev = shift_x_stdev
        self.shift_y_mean = shift_y_mean
        self.shift_y_stdev = shift_y_stdev

        n2 = float(self.size // 2)
        ax = torch.arange(-n2, n2 + self.size % 2)
        ax = torch.flip(ax, dims=[0])
        mx, my = torch.meshgrid(ax, ax)

        # shape SizexSize
        self.register_buffer("mx", mx.clone())
        self.register_buffer("my", my.clone())

    def modulate(self, x_fourier, t_x, t_y):
        '''
        outputs modulated (fourier equivalent of shifting in primal domain) input images given in batch wise format.
        The modulation depends on t_x, t_y.

        Parameters
        ----------
        x_fourier : torch.Tensor (Bx1xSizexSize)
            batch of input images in Fourier domain
        t_x: torch.Tensor (B,)
            batch of shifts along horizontal axis
        t_y: torch.Tensor (B,)
            batch of shifts along vertical axis
        Returns
        -------
        output: torch.Tensor (Bx1xSizexSize)
            batch of modulated fourier images given by
            output(f_1,f_2)=e^{-2*pi*j*[f_1,f_2]*[t_x, t_y] }*input(f_1,f_2)
        '''
        t_y = t_y[:, None, None, None]  # [B,1,1,1]
        t_x = t_x[:, None, None, None]  # [B,1,1,1]

        modulation = torch.exp(-2 * np.pi * 1j * self.frequency * (self.mx * t_y + self.my * t_x))  # [B,1,Size,Size]

        return x_fourier * modulation  # [B,1,Size,Size]*[B,1,Size,Size]

    def forward(self, x_fourier, shift_params={}):
        '''
        outputs modulated (fourier equivalent of shifting in primal domain) input images given in batch wise format.
        The modulation depends on t_x, t_y.

        Parameters
        ----------
        x_fourier : torch.Tensor (Bx1xSizexSize)
            batch of input images in Fourier domain

        shift_params:
            dictionary containing
            'shift_x': torch.Tensor (B,)
                batch of shifts along horizontal axis
            'shift_y': torch.Tensor (B,)
                batch of shifts along vertical axis
        Returns
        -------
        output: torch.Tensor (Bx1xSizexSize)
            batch of modulated fourier images if shift_params is not empty else input is outputted
        '''
        if shift_params:
            x_fourier = self.modulate(x_fourier, shift_params['shift_x'], shift_params['shift_y'])  # [B,1,Size,Size]
        return x_fourier
