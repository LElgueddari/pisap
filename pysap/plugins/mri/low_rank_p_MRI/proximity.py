# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Overload the proximity class from modopt.
"""

import numpy as np
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.feature_extraction.image import extract_patches_2d


class NuclearNorm(object):
    """The proximity of the nuclear norm operator

    This class defines the nuclear norm proximity operator on a patch based
    method

    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    thresh_type : str {'hard', 'soft'}, optional
        Threshold type (default is 'soft')
    patch_size: int
        Size of the patches to impose the low rank constraints
    """
    def __init__(self, weights, patch_shape):
        self.weights = weights
        self.patch_shape = patch_shape

    def op(self, data, extra_factor=1.0):
        """ Operator

        This method returns the input data thresholded by the weights

        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        DictionaryBase thresholded data

        """
        import ipdb;
        ipdb.set_trace()
        P_r = extract_patches_2d(np.moveaxis(np.real(data), -1, 0),
                                 self.patch_shape)
        P_i = extract_patches_2d(np.moveaxis(np.imag(data), -1, 0),
                                 self.patch_shape)
        P = P_r + 1j * P_i
        number_of_patches = P.shape[2]
        threshold = self.weights * extra_factor
        for patch_n in range(number_of_patches):
            u, s, vh = np.linalg.svd(np.reshape(
             P[:, :, patch_n, :], np.prod(self.patches_shape), data.shape[0]))
            s = s * np.maximum(1 - threshold / np.maximum(
                                                    np.finfo(np.float32).eps,
                                                    np.abs(s)), 0)
            P[:, :, patch_n, :] = np.reshape(np.dot(u, np.dot(s, vh)),
                                             self.patch_shape, data.shape[2])
        images_r = np.moveaxis(reconstruct_from_patches_2d(np.real(P)), -1, 0)
        images_i = np.moveaxis(reconstruct_from_patches_2d(np.imag(P)), -1, 0)
        return images_r + 1j * images_i

    def get_cost(self, x):
        """Cost function
        This method calculate the cost function of the proximable part.

        Parameters
        ----------
        x: np.ndarray
            Input array of the sparse code.

        Returns
        -------
        The cost of this sparse code
        """
        return 0.0
