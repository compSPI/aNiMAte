import torch
from torch import nn
import numpy as np

from utils import to_numpy

def primal_to_fourier_2D(r):
    r = torch.fft.fftshift(r, dim=(-2, -1))
    return torch.fft.ifftshift(torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1))

def fourier_to_primal_2D(f):
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1))

class CTFIdentity(nn.Module):
    def __init__(self):
        super(CTFIdentity, self).__init__()

    def forward(self, x_fourier, idcs=0, ctf_params={}):
        return x_fourier

class CTF(nn.Module):
    def __init__(self, size=257, resolution=0.8, downsampling=1,
                 defocus_mean=1., defocus_stdev=0.1,
                 angleAstigmatism=0., kV=300, valueNyquist=.001, cs=2.7, phasePlate=0.,
                 amplitudeContrast=.1, bFactor=0, requires_grad=False):

        super(CTF, self).__init__()
        self.requires_grad = requires_grad

        self.size = size
        self.resolution = resolution
        self.downsampling = downsampling

        self.defocus_mean = defocus_mean
        self.defocus_stdev = defocus_stdev
        self.global_angleAstigmatism = angleAstigmatism

        self.kV = kV
        self.valueNyquist = valueNyquist

        self.phasePlate = phasePlate
        self.amplitudeContrast = amplitudeContrast
        self.bFactor = bFactor

        ''' parameters derived from resolution'''
        self.wavelength, _ = self._get_ewavelength()
        self.resolution = self.resolution * self.downsampling
        self.frequency = 1. / (self.size * resolution)

        self.cs = cs
        ''' polar buffer'''
        n2 = float(self.size // 2)
        ax = torch.arange(-n2, n2 + self.size % 2)
        mx, my = torch.meshgrid(ax, ax)
        self.register_buffer("r2", mx ** 2 + my ** 2)
        self.register_buffer("r", torch.sqrt(self.r2))
        self.register_buffer("angleFrequency", torch.atan2(my, mx))

    def _get_ewavelength(self):
        wavelength = 12.2639 / np.sqrt(self.kV * 1e3 + 0.97845 * self.kV ** 2)
        u0 = 511  # electron rest energy in kilovolts
        sigma = (2 * np.pi / (wavelength * self.kV * 1e3)) * ((u0 + self.kV) / (2 * u0 + self.kV))

        return wavelength, sigma

    def get_psf(self):
        hFourier = self.get_ctf()
        hSpatial = torch.fft.fftshift(
                        torch.fft.ifftn(
                            torch.fft.ifftshift(hFourier,
                                                dim=(-2,-1)),
                                        s=(hFourier.shape[-2],hFourier.shape[-1]),
                                        dim=(-2,-1))) # is complex
        return hSpatial

    def get_ctf(self, ctf_params):
        defocus_u = ctf_params['defocus_u']
        defocus_v = ctf_params['defocus_v']
        angleAstigmatism = ctf_params['angleAstigmatism']

        elliptical = defocus_v * self.r2 + (defocus_u - defocus_v) * self.r2 * \
                     torch.cos(self.angleFrequency - angleAstigmatism)**2
        defocusContribution = np.pi * self.wavelength * 1e4 * elliptical * self.frequency ** 2
        abberationContribution = -np.pi / 2.0 * self.cs * (self.wavelength ** 3) * \
                                 1e7 * self.frequency ** 4 * self.r2 ** 2

        argument = self.phasePlate * np.pi / 2. + abberationContribution + defocusContribution

        hFourier = ((1 - self.amplitudeContrast ** 2) ** 0.5 * torch.sin(argument) +
                    self.amplitudeContrast * torch.cos(argument))

        if self.bFactor == 0:
            decay = np.sqrt(-np.log(self.valueNyquist)) * 2. * self.resolution
            envelope = torch.exp(-self.frequency ** 2 * decay ** 2 * self.r2)
        else:
            envelope = torch.exp(-self.frequency ** 2 * self.bFactor / 4. * self.r2)

        hFourier *= envelope

        return hFourier

    def forward(self, x_fourier, ctf_params):
        hFourier = self.get_ctf(ctf_params)
        return x_fourier * hFourier[:,None,:,:]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    filter = CTF(size=64, resolution=3.2)
    ctf = filter.get_ctf()
    psf = filter.get_psf()
    print(f"psf={psf.shape}")

    for i in range(10):
        fp, (axp1, axp2) = plt.subplots(1, 2, sharey=True)
        axp1.imshow(to_numpy(psf[i, :, :].real))
        axp2.imshow(to_numpy(ctf[i, :, :]))
        plt.show()

