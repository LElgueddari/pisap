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
from sklearn.feature_extraction.image import extract_patches
from sklearn.feature_extraction.image import _compute_n_patches
# from sklearn.feature_extraction.image import
from skimage.measure import compare_ssim


def mat2gray(image):
    """Rescale the image between 0 and 1

    Parameters:
    ----------
    image: np.ndarray
        The image complex or not that has to be rescaled. If the image is
        complex the returned image will be taken by the abs of the image
    Returns:
    -------
    out: np.ndarray
        The returned image
    """
    abs_image = np.abs(image)
    return (abs_image - abs_image.min())/(abs_image.max() - abs_image.min())


def compute_ssim(ref, image, mask=None):
    """Compute the SSIM from the refernce on a rescaled image

    Parameter:
    ----------
    ref: np.ndarray
        The reference image
    image: np.ndarray

    mask: np.ndarray
        A binary mask where the ssim should be calvulated
    Output:
    ------
    ssim; np.float
        SSIM value between 0 and 1
    """
    if mask is None:
        return compare_ssim(mat2gray(ref), mat2gray(image))
    else:
        _, maps_ssim = compare_ssim(mat2gray(ref), mat2gray(image),
                                            full=True)
        maps_ssim = mask*maps_ssim
        return maps_ssim.sum()/mask.sum()


def prod_over_maps(S, X):
    """
    Computes the element-wise product of the two inputs over the first two
    direction

    Parameters
    ----------
    S: np.ndarray
        The sensitivity maps of size [N,M,L]
    X: np.ndarray
        An image of size [N,M]

    Returns
    -------
    Sl: np.ndarray
        The product of every L element of S times X
    """
    Sl = np.copy(S)
    if Sl.shape == X.shape:
        for i in range(S.shape[2]):
            Sl[:, :, i] *= X[:, :, i]
    else:
        for i in range(S.shape[2]):
            Sl[:, :, i] *= X
    return Sl


def function_over_maps(f, x):
    """
    This methods computes the callable function over the third direction

    Parameters
    ----------
    f: callable
        This function will be applyed n times where n is the last element in
        the shape of x
    x: np.ndarray
        Input data

    Returns
    -------
    out: np.list
        the results of the function as a list where the length of the list is
        equal to n
    """
    yl = []
    for i in range(x.T.shape[0]):
        yl.append(f((x.T[i]).T))
    return np.stack(yl, axis=len(yl[0].shape))


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

def extract_patches_2d(image, patch_shape, overlapping_factor=0):

    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_shape
    patch_step_size = (int(p_h/overlapping_factor),
                       int(p_w/overlapping_factor),
                       image.shape[-1])
    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]
    extracted_patches = extract_patches(image,
                                        patch_shape=(p_h, p_w, n_colors),
                                        extraction_step=patch_step_size)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, None)

    patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches

def reconstruct_non_overlapped_patches_2d(patches, img_size):
    patches_size = patches.shape[1:-1]
    number_of_patch_x = int(img_size[0]/patches_size[0])
    number_of_patch_y = int(img_size[1]/patches_size[1])
    IMG = np.zeros((patches.shape[-1], *img_size), dtype=patches.dtype)
    for idx_y in range(number_of_patch_y):
        for idx_x in range(number_of_patch_x):
            patch_idx = idx_x*number_of_patch_y+idx_y
            patch_n = np.moveaxis(patches[patch_idx], -1, 0)
            IMG[:, idx_x*patches_size[0]: (idx_x+1)*patches_size[0],
                idx_y*patches_size[1]: (idx_y+1)*patches_size[1]] = patch_n
    return IMG
