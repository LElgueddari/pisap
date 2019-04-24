# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 3017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Fourier operators for cartesian and non-cartesian space.
"""


# Package import
import warnings
from .utils import convert_locations_to_mask_3D
from pysap.plugins.mri.reconstruct.fourier import FourierBase
from modopt.interface.errors import warn

# Third party import
try:
    import pynfft
except Exception:
    warnings.warn("pynfft python package has not been found. If needed use "
                  "the master release.")
    pass

try:
    from pynufft import NUFFT_hsa
    from pynufft import NUFFT_cpu

except Exception:
    warnings.warn("pyNUFFT python package has not been found. Try Pynfft"
                  "if non uniform Fourier transform needed")

import numpy as np
import scipy.fftpack as pfft


class FFT3(FourierBase):
    """ Standard 3D Fast Fourrier Transform class.

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    """
    def __init__(self, samples, shape):
        """ Initilize the 'FFT3' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        """
        self.samples = samples
        self.shape = shape
        self._mask = convert_locations_to_mask_3D(self.samples, self.shape)

    def op(self, img):
        """ This method calculates the masked Fourier transform of a 3-D image.

        Parameters
        ----------
        img: np.ndarray
            input 3D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        return self._mask * pfft.fftn(img) / np.sqrt(np.prod(self.shape))

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a 3-D
        image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data.

        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        return pfft.ifftn(self._mask * x) * np.sqrt(np.prod(self.shape))


class NFFT3(FourierBase):
    """ Standard 3D non cartesian Fast Fourrier Transform class

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    """

    def __init__(self, samples, shape):
        """ Initilize the 'NFFT3' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        """
        self.plan = pynfft.NFFT(N=shape, M=len(samples))
        self.shape = shape
        self.samples = samples
        self.plan.x = self.samples
        self.plan.precompute()

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 3-D image.

        Parameters
        ----------
        img: np.ndarray
            input 3D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        self.plan.f_hat = img
        return (1.0 / np.sqrt(self.plan.M)) * self.plan.trafo()

    def adj_op(self, x):
        """ This method calculates inverse masked non-cartesian Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-cartesian Fourier transform 1D data.

        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        self.plan.f = x
        return (1.0 / np.sqrt(self.plan.M)) * self.plan.adjoint()


class Singleton:
    numOfInstances = 0

    def countInstances(cls):
        cls.numOfInstances += 1

    countInstances = classmethod(countInstances)

    def getNumInstances(cls):
        return cls.numOfInstances
    getNumInstances = classmethod(getNumInstances)

    def __init__(self):
        self.countInstances()


class NUFFT(FourierBase, Singleton):
    """  N-D non uniform Fast Fourrier Transform class

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (necessarly a square/cubic matrix).
    nufftObj: The pynufft object
        depending on the required computational platform
    platform: string, 'cpu', 'mcpu' or 'gpu'
        string indicating which hardware platform will be used to compute the
        NUFFT
    Kd: int or tuple
        int or tuple indicating the size of the frequency grid, for regridding.
        if int, will be evaluated to (Kd,)*nb_dim of the image
    Jd: int or tuple
        Size of the interpolator kernel. If int, will be evaluated
        to (Jd,)*dims image
    """
    numOfInstances = 0

    def __init__(self, samples, shape, platform='cpu', Kd=None, Jd=None,
                 n_coils=1, verbosity=0):
        """ Initilize the 'NUFFT' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (necessarly a square/cubic matrix).
        platform: string, 'cpu', 'mcpu' or 'gpu'
            string indicating which hardware platform will be used to
            compute the NUFFT
        Kd: int or tuple
            int or tuple indicating the size of the frequency grid,
            for regridding. If int, will be evaluated
            to (Kd,)*nb_dim of the image
        Jd: int or tuple
            Size of the interpolator kernel. If int, will be evaluated
            to (Jd,)*dims image
        n_coils: int
            Number of coils used to acquire the signal in case of multiarray
            receiver coils

        """
        if (n_coils < 1) or (type(n_coils) is not int):
            raise ValueError('The number of coils should be an integer >= 1')
        self.nb_coils = n_coils
        self.shape = shape
        self.platform = platform
        self.samples = samples * (2 * np.pi)  # Pynufft use samples in
        # [-pi, pi[ instead of [-0.5, 0.5[
        self.dim = samples.shape[1]  # number of dimensions of the image

        if type(Kd) == int:
            self.Kd = (Kd,)*self.dim
        elif type(Kd) == tuple:
            self.Kd = Kd
        elif Kd is None:
            # Preferential option
            self.Kd = tuple([2*ix for ix in shape])

        if type(Jd) == int:
            self.Jd = (Jd,)*self.dim
        elif type(Jd) == tuple:
            self.Jd = Jd
        elif Jd is None:
            # Preferential option
            self.Jd = (5,)*self.dim

        for (i, s) in enumerate(shape):
            assert(self.shape[i] <= self.Kd[i]), 'size of frequency grid' + \
                   'must be greater or equal than the image size'

        print('Creating the NUFFT object...')
        if self.platform == 'cpu':
            self.nufftObj = NUFFT_cpu()
            self.nufftObj.plan(om=self.samples,
                               Nd=self.shape,
                               Kd=self.Kd,
                               Jd=self.Jd,
                               batch=self.nb_coils)

        elif self.platform == 'mcpu':
            warn('Attemping to use OpenCL plateform. Make sure to '
                 'have  all the dependecies installed')
            self.nufftObj = NUFFT_hsa(API='ocl',
                                      platform_number=None,
                                      device_number=None,
                                      verbosity=verbosity)

            self.nufftObj.plan(om=self.samples,
                               Nd=self.shape,
                               Kd=self.Kd,
                               Jd=self.Jd,
                               batch=1,  # self.nb_coils,
                               ft_axes=tuple(range(samples.shape[1])),
                               radix=None)

        elif self.platform == 'gpu':
            warn('Attemping to use Cuda plateform. Make sure to '
                 'have  all the dependecies installed and '
                 'to create only one instance of NUFFT GPU')
            if self.getNumInstances() > 1:
                raise RuntimeError('You have created more than one GPU NUFFT'
                                   ' object')
            self.nufftObj = NUFFT_hsa(API='cuda',
                                      platform_number=None,
                                      device_number=None,
                                      verbosity=verbosity)

            self.nufftObj.plan(om=self.samples,
                               Nd=self.shape,
                               Kd=self.Kd,
                               Jd=self.Jd,
                               batch=1,  # self.nb_coils,
                               ft_axes=tuple(range(samples.shape[1])),
                               radix=None)
            Singleton.__init__(self)

        else:
            raise ValueError('Wrong type of platform. Platform must be'
                             '\'cpu\', \'mcpu\' or \'gpu\'')

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 3-D image.

        Parameters
        ----------
        img: np.ndarray
            input 3D array with the same shape as shape.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        if self.nb_coils == 1:
            if (self.platform == 'cpu'):
                y = np.squeeze(self.nufftObj.forward(img))
            else:
                dtype = np.complex64
                # Send data to the mCPU/GPU platform
                self.nufftObj.x_Nd = self.nufftObj.thr.to_device(
                    img.astype(dtype))
                gx = self.nufftObj.thr.copy_array(self.nufftObj.x_Nd)
                # Forward operator of the NUFFT
                gy = self.nufftObj.forward(gx)
                y = np.squeeze(gy.get())
        else:
            if (self.platform == 'cpu'):
                y = np.moveaxis(self.nufftObj.forward(np.copy(np.moveaxis(
                    img, 0, -1))), -1, 0)
            else:
                dtype = np.complex64
                # Send data to the mCPU/GPU platform
                y = []
                for ch in range(self.nb_coils):
                    self.nufftObj.x_Nd = self.nufftObj.thr.to_device(
                        np.copy(img[ch]).astype(dtype))
                    gx = self.nufftObj.thr.copy_array(self.nufftObj.x_Nd)
                    # Forward operator of the NUFFT
                    gy = self.nufftObj.forward(gx)
                    y.append(np.squeeze(gy.get()))
                y = np.asarray(y)
        return y * 1.0 / np.sqrt(np.prod(self.Kd))

    def adj_op(self, x):
        """ This method calculates inverse masked non-uniform Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-uniform Fourier transform 1D data.

        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        if self.nb_coils == 1:
            if self.platform == 'cpu':
                img = np.squeeze(self.nufftObj.adjoint(x))
            else:
                dtype = np.complex64
                cuda_array = self.nufftObj.thr.to_device(x.astype(dtype))
                gx = self.nufftObj.adjoint(cuda_array)
                img = np.squeeze(gx.get())
        else:
            if self.platform == 'cpu':
                img = np.moveaxis(self.nufftObj.adjoint(np.moveaxis(x, 0, -1)),
                                  -1, 0)
            else:
                dtype = np.complex64
                img = []
                for ch in range(self.nb_coils):
                    cuda_array = self.nufftObj.thr.to_device(np.copy(
                        x[ch]).astype(dtype))
                    gx = self.nufftObj.adjoint(cuda_array)
                    img.append(gx.get())
                img = np.asarray(np.squeeze(img))
        return img * np.sqrt(np.prod(self.Kd))
