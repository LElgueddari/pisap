# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import unittest
import numpy as np

# Package import
from pysap.plugins.mri.low_rank_p_MRI.utils import extract_patches_2d
from pysap.plugins.mri.low_rank_p_MRI.utils import \
                                    reconstruct_non_overlapped_patches_2d
from pysap.plugins.mri.low_rank_p_MRI.proximity import NuclearNorm

class Test_extraction_reconstruction_patches(unittest.TestCase):

    def test_extraction_reconstruction_nonoverlapped_patches(self):
        data_shape = (16, 64, 128)
        patch_shape = (8, 32)
        data = np.random.randn(*data_shape) + 1j * np.random.randn(*data_shape)
        patches = extract_patches_2d(image=np.moveaxis(data, 0, -1),
                                     patch_shape=patch_shape,
                                     overlapping_factor=1)
        print("data shape: ", data.shape)
        print("patch's shape: ", patches.shape)
        reconstructed_data = reconstruct_non_overlapped_patches_2d(
            patches=patches,
            img_size=data.shape[1:]
            )
        mismatch = (1. - np.mean(
            np.allclose(reconstructed_data, data)))
        print("      mismatch = ", mismatch)
        print(" Test extract and reconstruct non overlapped patches passes ")

    def test_proximity_op_NuclearNorm_for_zeros_thr(self):
        data_shape = (32, 64, 128)
        patch_shape = (8, 32)
        weights = 0
        data = np.random.randn(*data_shape) + 1j * np.random.randn(*data_shape)
        prox_op = NuclearNorm(weights=weights,
                              patch_shape=patch_shape,
                              overlapping_factor=1)
        reconstructed_data = prox_op.op(
            data=data,
            extra_factor=1.0,
            num_cores=1
            )
        mismatch = (1. - np.mean(
            np.allclose(reconstructed_data, data)))
        print("      mismatch = ", mismatch)
        print(" Proximity operator doesn't distord input when patches are",
              "applied")

    def test_proximity_op_NuclearNorm_for_zeros_thr_no_patches(self):
        data_shape = (32, 64, 128)
        patch_shape = (64, 128)
        weights = 0
        data = np.random.randn(*data_shape) + 1j * np.random.randn(*data_shape)
        prox_op = NuclearNorm(weights=weights,
                              patch_shape=patch_shape,
                              overlapping_factor=1)
        reconstructed_data = prox_op.op(
            data=data,
            extra_factor=1.0,
            num_cores=1
            )
        mismatch = (1. - np.mean(
            np.allclose(reconstructed_data, data)))
        print("      mismatch = ", mismatch)
        print(" Proximity operator doesn't distord input in no patches setting")

if __name__ == "__main__":
    unittest.main()
