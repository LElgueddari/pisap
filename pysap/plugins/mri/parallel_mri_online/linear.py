# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
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
import pywt
from .utils import unflatten_wave2, unflatten_swt2, unflatten_dwt2
from .utils import flatten_swt2, flatten_wave2, flatten_dwt2

# Third party import
import numpy


class Identity(object):
    """ The 2D wavelet transform class.
    """
    def op(self, data):
        self.coeffs_shape = data.shape
        return data

    def adj_op(self, coeffs):
        """ Define the wavelet adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        return coeffs

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
        shape = numpy.asarray(shape)
        shape += shape % 2
        fake_data = numpy.zeros(shape)
        fake_data[list(zip(shape // 2))] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)

class Pywavelet2(object):

    def __init__(self, wavelet_name, nb_scale=4, verbose=0, undecimated=False):
        """ Initialize the 'pyWavelet3' class.
            print(x_new.shape)
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
        if wavelet_name not in pywt.wavelist():
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        self.pywt_transform = pywt.Wavelet(wavelet_name)
        self.single_level=False
        if nb_scale == 1 and not undecimated:
            self.single_level = True
            self.unflatten = unflatten_dwt2
            self.flatten = flatten_dwt2
        else:
            self.nb_scale = nb_scale-1
            self.undecimated = undecimated
            self.unflatten = unflatten_swt2 if undecimated else unflatten_wave2
            self.flatten = flatten_swt2 if undecimated else flatten_wave2
        self.coeffs_shape = None

    def get_coeff(self):
        """ Return the wavelet coeffiscients
        Return:
        -------
        The wavelet coeffiscients value
        """
        return self.coeffs

    def set_coeff(self, coeffs):
        """ Set the wavelet coefficients value
        """
        self.coeffs = coeffs  # XXX: TODO: add some checks

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
        if isinstance(data, numpy.ndarray):
            data = pysap.Image(data=data)

        if self.single_level:
            coeffs_dict = pywt.dwt2(data, self.pywt_transform)
        else:
            if self.undecimated:
                coeffs_dict = pywt.swt2(data, self.pywt_transform,
                                        level=self.nb_scale)
            else:
                coeffs_dict = pywt.wavedec2(data,
                                            self.pywt_transform,
                                            level=self.nb_scale)
        self.coeffs, self.coeffs_shape = self.flatten(coeffs_dict)
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
        self.coeffs = coeffs
        if self.single_level:
            coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
            data = pywt.idwt2(coeffs_dict, self.pywt_transform)
        else:
            if self.undecimated:
                coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
                data = pywt.iswt2(coeffs_dict,
                                  self.pywt_transform)
            else:
                coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
                data = pywt.waverec2(
                    coeffs=coeffs_dict,
                    wavelet=self.pywt_transform)
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
        print(shape)
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
