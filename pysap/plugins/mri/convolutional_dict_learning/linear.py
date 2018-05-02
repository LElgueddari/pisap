# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains convolutional sparse coding operators classes.
"""

# Package import
import pysap

# Third party import
import numpy
from sporco.admm import cbpdn


class ConvSparseCode2D(object):
    """ The 2D convolutional sparse coding class.
    """

    def __init__(self, dictionary_atoms, image_shape, lbd=5e-2, verbose=False,
                 sparse_cod_iter=200, relative_tolerance=5e-3):
        """ Initialize the 'ConvSparseCode2D' class.

        Parameters
        ----------

        """
        self.lbd = lbd
        self.opt = cbpdn.ConvBPDN.Options({'Verbose': verbose,
                                           'MaxMainIter': sparse_cod_iter,
                                           'RelStopTol': relative_tolerance,
                                           'AuxVarObj': False})
        self.atoms = dictionary_atoms
        self.coeffs_shape = None
        self.coder = cbpdn.ConvBPDN(self.atoms, numpy.zeros(image_shape),
                                    self.lbd, self.opt, dimK=0)

    def op(self, data):
        """ Define the wavelet operator.

        This method returns the input data convolved with the learned atoms.

        Parameters
        ----------
        data: ndarray or Image
            input 2D data array.

        Returns
        -------
        coeffs: ndarray
            the sparse coefficients.
        """
        # if ~isinstance(data, numpy.ndarray):
        #     data = data.data
        self.coder = cbpdn.ConvBPDN(self.atoms, numpy.abs(data),
                                    self.lbd, self.opt, dimK=0)
        coeffs = self.coder.solve()
        self.coeffs_shape = coeffs.shape
        return coeffs.flatten()

    def adj_op(self, coeffs, dtype="array"):
        """ Define the wavelet adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            the sparse coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pysap.Image.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        image = self.coder.reconstruct(coeffs.reshape(self.coeffs_shape))
        if dtype == "array":
            return image.data
        return image

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
