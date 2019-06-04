# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains all the utils tools needed in the p_MRI reconstruction.
"""


# System import

# Package import

# Third party import
import numpy as np

def virtual_coil_reconstruction(imgs):
    """
    Calculate the combination of all the coils thanks to the virtual coil
    methods as defined:

    Parameters
    ----------
    imgs: np.ndarray
        The images reconstructed channel by channel [Nch, Nx, Ny, Nz]

    Returns
    -------
    I: np.ndarray
        The combination of all the channels in a complex valued [Nx, Ny, Nz]
    """
    # Compute first the virtual coil
    nch, nx, ny, nz = imgs.shape
    weights = np.sum(np.abs(imgs), axis=0)
    weights[weights==0] = 1e-16
    phase_reference = np.asarray([np.angle(np.sum(
        imgs[ch].flatten())) for ch in range(nch)])
    reference = np.asarray([(imgs[ch] / weights) / np.exp(1j * phase_reference[ch])
                           for ch in range(nch)])
    virtual_coil = np.sum(reference, axis=0)
    difference_original_vs_virtual = np.asarray(
        [np.conjugate(imgs[ch]) * virtual_coil for ch in range(nch)])
    hanning_1d_nx = np.expand_dims(np.hanning(nx), 1)
    hanning_1d_ny = np.expand_dims(np.hanning(ny), 1)
    hanning_1d_nz = np.expand_dims(np.hanning(nz), 1)
    hanning_2d = np.expand_dims(np.dot(hanning_1d_nx, hanning_1d_ny.T), -1)
    hanning_3d = np.fft.fftshift(np.dot(hanning_2d, hanning_1d_nz.T))
    hanning_Nd = np.tile(hanning_3d, (nch, 1, 1, 1))
    # Removing the background noise via low pass filtering
    difference_original_vs_virtual = np.fft.ifftn(np.fft.fftn(
        difference_original_vs_virtual, axes=(1, 2, 3)) * hanning_Nd,
        axes=(1, 2, 3))
    I = np.asarray([imgs[ch] * np.exp(1j * np.angle(
                    difference_original_vs_virtual[ch])) for ch in range(nch)])
    return np.sum(I, 0)

def check_lipschitz_cst(f, x_shape, lipschitz_cst, max_nb_of_iter=10):
    """
    This methods check that for random entrees the lipschitz constraint are
    statisfied:

    * ||f(x)-f(y)|| < lipschitz_cst ||x-y||

    Parameters
    ----------
    f: callable
        This lipschitzien function
    x_shape: tuple
        Input data shape
    lipschitz_cst: float
        The Lischitz constant for the function f
    max_nb_of_iter: int
        The number of time the constraint must be satisfied

    Returns
    -------
    out: bool
        If is True than the lipschitz_cst given in argument seems to be an
        upper bound of the real lipschitz constant for the function f
    """
    is_lips_cst = True
    n = 0

    while is_lips_cst and n < max_nb_of_iter:
        n += 1
        x = np.random.randn(*x_shape)
        y = np.random.randn(*x_shape)
        is_lips_cst = (np.linalg.norm(f(x)-f(y)) <= (lipschitz_cst *
                                                     np.linalg.norm(x-y)))

    return is_lips_cst
