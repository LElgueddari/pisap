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
from pysap.plugins.mri.low_rank_p_MRI.utils import extract_patches_2d
from pysap.plugins.mri.low_rank_p_MRI.utils import \
                                    reconstruct_non_overlapped_patches_2d
from joblib import Parallel, delayed
import multiprocessing


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
            (np.prod(self.patch_shape), patch.shape[-1])),
            full_matrices=False)
        s = s * np.maximum(1 - threshold / np.maximum(
                                            np.finfo(np.float32).eps,
                                            np.abs(s)), 0)
        patch = np.reshape(
            np.dot(u * s, vh),
            (*self.patch_shape, patch.shape[-1]))
        return patch

    def _nuclear_norm_cost(self, patch):
        _, s, _ = np.linalg.svd(np.reshape(
            patch,
            (np.prod(self.patch_shape), patch.shape[-1])),
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

            raise('Nuclear norm with overlapped patches not implemented yet')

            # P = extract_patches_2d(np.moveaxis(data, 0, -1), self.patch_shape)
            # number_of_patches = P.shape[0]
            # threshold = self.weights * extra_factor
            # num_cores = 1  # int(multiprocessing.cpu_count()/2)
            # if num_cores==1:
            #     for idx in range(number_of_patches):
            #         P[idx, :, :, :] = self._prox_nuclear_norm(
            #             patch=P[idx, :, :, :,],
            #             threshold = threshold
            #             )
            # else:
            #     print("Using joblib")
            #     P = Parallel(n_jobs=num_cores)(delayed(self._prox_nuclear_norm)(
            #                 patch=P[idx, : ,: ,:],
            #                 threshold=threshold) for idx in range(number_of_patches))
            #
            # images_r = np.moveaxis(reconstruct_from_patches_2d(
            #     np.real(P),
            #     np.moveaxis(data, 0, -1).shape), 0, -1)
            # images_i = np.moveaxisaxes(reconstruct_from_patches_2d(
            #     np.imag(P),
            #     np.moveaxis(data, 0, -1).shape), 0, -1)
            # return images_r + 1j * images_i

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

            return cost
        else:

            raise('Nuclear norm with overlapped patches not implemented yet')


            # P = extract_patches_2d(np.moveaxis(data, 0, -1), self.patch_shape)
            # number_of_patches = P.shape[0]
            # threshold = self.weights * extra_factor
            # num_cores = 1  # int(multiprocessing.cpu_count()/2)
            # if num_cores==1:
            #     for idx in range(number_of_patches):
            #         P[idx, :, :, :] = self._prox_nuclear_norm(
            #             patch=P[idx, :, :, :,],
            #             threshold = threshold
            #             )
            # else:
            #     print("Using joblib")
            #     P = Parallel(n_jobs=num_cores)(delayed(self._cost_nuclear_norm)(
            #                 patch=P[idx, : ,: ,:],
            #                 threshold=threshold) for idx in range(number_of_patches))
            #
            # images_r = np.moveaxis(reconstruct_from_patches_2d(
            #     np.real(P),
            #     np.moveaxis(data, 0, -1).shape), 0, -1)
            # images_i = np.moveaxisaxes(reconstruct_from_patches_2d(
            #     np.imag(P),
            #     np.moveaxis(data, 0, -1).shape), 0, -1)
            # return images_r + 1j * images_i
