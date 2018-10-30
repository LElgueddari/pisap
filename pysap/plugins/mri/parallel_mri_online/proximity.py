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
import warnings
from pysap.plugins.mri.parallel_mri_online.utils import extract_patches_2d
from pysap.plugins.mri.parallel_mri_online.utils import \
                                    reconstruct_non_overlapped_patches_2d
from pysap.plugins.mri.parallel_mri_online.utils import \
                                    reconstruct_overlapped_patches_2d
from joblib import Parallel, delayed
from pysap.plugins.mri.parallel_mri_online.utils import \
                                    _oscar_weights
from sklearn.isotonic import isotonic_regression


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
    overlapping_factor: int
        if 1 no overlapping will be made,
        if = 2,means 2 patches overlaps
    """
    def __init__(self, weights, patch_shape, overlapping_factor=1):
        """
        Parameters:
        -----------
        """
        self.weights = weights
        self.patch_shape = patch_shape
        self.overlapping_factor = overlapping_factor
        if self.overlapping_factor == 1:
            print("Patches doesn't overlap")

    def _prox_nuclear_norm(self, patch, threshold):
        u, s, vh = np.linalg.svd(np.reshape(
            patch,
            (np.prod(self.patch_shape[:-1]), patch.shape[-1])),
            full_matrices=False)
        s = s * np.maximum(1 - threshold / np.maximum(
                                            np.finfo(np.float32).eps,
                                            np.abs(s)), 0)
        patch = np.reshape(
            np.dot(u * s, vh),
            patch.shape)
        return patch

    def _nuclear_norm_cost(self, patch):
        _, s, _ = np.linalg.svd(np.reshape(
            patch,
            (np.prod(self.patch_shape[:-1]), patch.shape[-1])),
            full_matrices=False)
        return np.sum(np.abs(s.flatten()))

    def op(self, data, extra_factor=1.0, num_cores=1):
        """ Operator

        This method returns the input data thresholded by the weights

        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor
        num_cores: int
            Number of cores used to parrallelize the computation

        Returns
        -------
        DictionaryBase thresholded data

        """
        threshold = self.weights * extra_factor
        if data.shape[1:] == self.patch_shape:
            images = np.moveaxis(data, 0, -1)
            images = self._prox_nuclear_norm(patch=np.reshape(
                np.moveaxis(data, 0, -1),
                (np.prod(self.patch_shape), data.shape[0])),
                threshold=threshold)
            return np.moveaxis(images, -1, 0)
        elif self.overlapping_factor == 1:
            P = extract_patches_2d(np.moveaxis(data, 0, -1),
                                   self.patch_shape,
                                   overlapping_factor=self.overlapping_factor)
            number_of_patches = P.shape[0]
            num_cores = num_cores
            if num_cores==1:
                for idx in range(number_of_patches):
                    P[idx, :, :, :] = self._prox_nuclear_norm(
                        patch=P[idx, :, :, :,],
                        threshold = threshold
                        )
            else:
                print("Using joblib")
                P = Parallel(n_jobs=num_cores)(delayed(self._prox_nuclear_norm)(
                            patch=P[idx, : ,: ,:],
                            threshold=threshold) for idx in range(number_of_patches))

            output = reconstruct_non_overlapped_patches_2d(patches=P,
                                                 img_size=data.shape[1:])
            return output
        else:

            P = extract_patches_2d(np.moveaxis(data, 0, -1), self.patch_shape,
                                   overlapping_factor=self.overlapping_factor)
            number_of_patches = P.shape[0]
            threshold = self.weights * extra_factor
            extraction_step_size=[int(P_shape/self.overlapping_factor) for P_shape
                                  in self.patch_shape]
            if num_cores==1:
                for idx in range(number_of_patches):
                    P[idx, :, :, :] = self._prox_nuclear_norm(
                        patch=P[idx, :, :, :,],
                        threshold = threshold
                        )
            else:
                print("Using joblib")
                P = Parallel(n_jobs=num_cores)(delayed(self._prox_nuclear_norm)(
                            patch=P[idx, : ,: ,:],
                            threshold=threshold) for idx in range(number_of_patches))
            image = reconstruct_overlapped_patches_2d(
                img_size=np.moveaxis(data, 0, -1).shape,
                patches=P,
                extraction_step_size=extraction_step_size)
            return np.moveaxis(image, -1, 0)

    def get_cost(self, data, extra_factor=1.0, num_cores=1):
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
        cost = 0
        threshold = self.weights * extra_factor
        if data.shape[1:] == self.patch_shape:
            cost += self._nuclear_norm_cost(patch=np.reshape(
                np.moveaxis(data, 0, -1),
                (np.prod(self.patch_shape), data.shape[0])))
            return cost * threshold
        elif self.overlapping_factor == 1:
            P = extract_patches_2d(np.moveaxis(data, 0, -1),
                                   self.patch_shape,
                                   overlapping_factor=self.overlapping_factor)
            number_of_patches = P.shape[0]
            num_cores = num_cores
            if num_cores==1:
                for idx in range(number_of_patches):
                    cost += self._nuclear_norm_cost(
                        patch=P[idx, :, :, :,]
                        )
            else:
                print("Using joblib")
                cost += Parallel(n_jobs=num_cores)(delayed(
                    self._cost_nuclear_norm)(
                        patch=P[idx, : ,: ,:]
                        ) for idx in range(number_of_patches))

            return cost * threshold
        else:
            P = extract_patches_2d(np.moveaxis(data, 0, -1), self.patch_shape,
                                   overlapping_factor=self.overlapping_factor)
            number_of_patches = P.shape[0]
            threshold = self.weights * extra_factor
            if num_cores==1:
                for idx in range(number_of_patches):
                    cost += self._nuclear_norm_cost(
                        patch=P[idx, :, :, :,])
            else:
                print("Using joblib")
                cost += Parallel(n_jobs=num_cores)(delayed(self._nuclear_norm_cost)(
                            patch=P[idx, : ,: ,:])
                            for idx in range(number_of_patches))
            return cost * threshold


class GroupLasso(object):
    """The proximity of the group-lasso regularisation

    This class defines the group-lasso penalization

    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    """
    def __init__(self, weights):
        """
        Parameters:
        -----------
        """
        self.weights = weights

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
        threshold = self.weights * extra_factor
        norm_2 = np.linalg.norm(data, axis=0)

        np.maximum((1.0 - threshold /
                         np.maximum(np.finfo(np.float64).eps, np.abs(data))),
                         0.0) * data
        return data * np.maximum(0, 1.0 - self.weights*extra_factor /
                                 np.maximum(norm_2, np.finfo(np.float32).eps))

    def get_cost(self, data):
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
        return np.sum(np.linalg.norm(data, axis=0))


class OWL(object):
    """The proximity of the OWL regularisation

    This class defines the OWL penalization

    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    """
    def __init__(self, alpha, beta=None, data_shape=None, mode='all',
                 n_channel=1, num_cores=1):
        """
        Parameters:
        -----------
        """
        self.weights = alpha
        self.mode = mode
        self.num_cores = num_cores
        if beta is not None:
            print("Uses OSCAR: Octogonal Shrinkage and Clustering Algorithm for"
                   "Regression")
            if data_shape is None:
                raise('Data size must be specified if OSCAR is used')
            else:
                if self.mode is 'all':
                    self.weights = _oscar_weights(alpha, beta,
                                                  data_shape * n_channel)
                elif self.mode is 'band_based':
                    self.band_shape = data_shape
                    self.weights = []
                    for band_shape in data_shape:
                        self.weights.append(_oscar_weights(
                            alpha, beta, n_channel * np.prod(band_shape)))
                elif self.mode is 'coeff_based':
                    self.weights = _oscar_weights(alpha, beta, n_channel)
                else:
                    raise('Unknow mode')

    def _prox_owl(self, data, threshold):
        data_abs = np.abs(data)
        ix = np.argsort(data_abs)[::-1]
        data_abs = data_abs[ix]  # Sorted absolute value of the data

        # Project on the monotone non-negative deacresing cone
        data_abs = isotonic_regression(data_abs - threshold, y_min=0,
                                       increasing=False)
        # Undo the sorting
        inv_x = np.zeros_like(ix)
        inv_x[ix] = np.arange(len(data))
        data_abs = data_abs[inv_x]

        sign_data = data/np.abs(data)
        sign_data[np.isnan(sign_data)] = 0

        return sign_data * data_abs

    def _reshape_mode_based(self, data):
        output = []
        start = 0
        n_channel = data.shape[0]
        for band_shape_idx in self.band_shape:
            n_coeffs = np.prod(band_shape_idx)
            stop = start + n_coeffs
            output.append(np.reshape(data[:, start: stop], (n_channel*n_coeffs)))
            start = stop
        return output

    def op(self, data, extra_factor=1.0):
        """
        Define the proximity operator of the OWL norm
        """
        if self.mode is 'all':
            threshold = self.weights * extra_factor
            output = self._prox_owl(data.flatten(), threshold)
        elif self.mode is 'band_based':
            data_r = self._reshape_mode_based(data)
            output = []
            output = Parallel(n_jobs=self.num_cores)(delayed(self._prox_owl)(
                        data=data_band,
                        threshold=weights * extra_factor)
                        for data_band, weights in zip(data_r, self.weights))
            reshaped_data = np.zeros(data.shape, dtype=data.dtype)
            start = 0
            n_channel = data.shape[0]
            for band_shape_idx, band_data in zip(self.band_shape, output):
                stop = start + np.prod(band_shape_idx)
                reshaped_data[:, start : stop] = np.reshape(band_data, (n_channel, np.prod(band_shape_idx)))
                start = stop
            output = np.asarray(reshaped_data).T
        elif self.mode is 'coeff_based':
            threshold = self.weights * extra_factor
            output = Parallel(n_jobs=self.num_cores)(delayed(self._prox_owl)(
                        data=np.squeeze(data[:, idx]),
                        threshold=threshold) for idx in range(data.shape[1]))
        return np.asarray(output).T

    def get_cost(self, data):
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
        warnings.warn('Cost function not implemented yet', UserWarning)
        return 0
