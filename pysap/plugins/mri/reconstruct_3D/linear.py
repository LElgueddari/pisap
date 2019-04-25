##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains linears operators classes.
"""


# Package import
import pysap
from pysap.plugins.mri.reconstruct_3D.utils import flatten_swtn
from pysap.plugins.mri.reconstruct_3D.utils import unflatten_swtn
from pysap.plugins.mri.reconstruct_3D.utils import flatten_wave
from pysap.plugins.mri.reconstruct_3D.utils import unflatten_wave

# Third party import
import numpy
import pywt
import warnings


class pyWavelet3(object):
    """ The 3D wavelet transform class from pyWavelets package.
    """
    def __init__(self, wavelet_name, nb_scale=4, verbose=0, undecimated=False,
                 multichannel=False):
        """ Initialize the 'pyWavelet3' class.
            (x_new.shape)
        Parameters
        ----------
        wavelet_name: str
            the wavelet name to be used during the decomposition.
        nb_scales: int, default 4
            the number of scales in the decomposition.
        verbose: int, default 0
            the verbosity level.
        undecimated: bool, default False
            enable use undecimated wavelet transform.
        """
        self.nb_scale = nb_scale
        if wavelet_name not in pywt.wavelist():
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        self.transform = pywt.Wavelet(wavelet_name)
        self.nb_scale = nb_scale - 1
        self.undecimated = undecimated
        self.multichannel = multichannel
        self.unflatten = unflatten_swtn if undecimated else unflatten_wave
        self.flatten = flatten_swtn if undecimated else flatten_wave
        self.coeffs_shape = None

    def get_coeff(self):
        return self.coeffs

    def set_coeff(self, coeffs):
        self.coeffs = coeffs

    def op(self, data):
        """ Define the wavelet operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray or Image
            input 3D data array.

        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        # if isinstance(data, numpy.ndarray):
        #     data = pysap.Image(data=data)
        axes = tuple(range(1, data.ndim)) if self.multichannel else None
        if self.undecimated:
            warnings.warn('Data size should a power of 2')
            coeffs_dict = pywt.swtn(data,
                                    self.transform,
                                    level=self.nb_scale,
                                    axes=axes)
        else:
            coeffs_dict = pywt.wavedecn(data,
                                        self.transform,
                                        level=self.nb_scale,
                                        mode='zero',
                                        axes=axes)
        self.coeffs, self.coeffs_shape = self.flatten(
            coeffs_dict,
            multichannel=self.multichannel)
        return self.coeffs

    def adj_op(self, coeffs, dtype="array"):
        """ Define the wavelet adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pysap.Image.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if self.coeffs_shape is None:
            raise ValueError("The attributes coeffs_shape must be",
                             " instanciated you can do it by calling",
                             " linear_op.op with the right dimension")
        self.coeffs = coeffs
        coeffs_dict = self.unflatten(coeffs, self.coeffs_shape,
                                     multichannel=self.multichannel)

        if self.undecimated:
            axes = tuple(range(1, coeffs_dict[0][list(
                coeffs_dict[0])[0]].ndim)) if self.multichannel else None
            data = pywt.iswtn(coeffs_dict,
                              self.transform,
                              axes=axes)
        else:
            axes = tuple(range(
                1, coeffs_dict[0].ndim)) if self.multichannel else None
            data = pywt.waverecn(coeffs=coeffs_dict,
                                 wavelet=self.transform,
                                 mode='zero',
                                 axes=axes)
        if dtype == "array":
            return data
        return pysap.Image(data=data)

    def l2norm(self, shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        shape: uplet
            the data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        (shape)
        shape = numpy.asarray(shape)
        shape += shape % 2
        fake_data = numpy.zeros(shape)

        fake_data[[(int(i[0]),) for i in list(zip(shape/2))]] = 1
        # WARNING: this line is overly complicated, but it basically does this:
        # fake_data[zip(shape / 2)] = 1
        # It is written as such to help Python2.x/3.x compatibility

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)
