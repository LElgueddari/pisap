"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L Elgueddari, B. Sarthou

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
import pysap
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct.reconstruct import FFT2
from pysap.plugins.mri.reconstruct.gradient import GradAnalysis2
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.convolutional_dict_learning.linear import ConvSparseCode2D

# Third party import
import numpy as np
import scipy.fftpack as pfft

# Loading input data
# Loading input data
image = get_sample_data("mri-slice-nifti")
image.data += np.random.randn(*image.shape) * 20.
mask = get_sample_data("mri-mask")
image.show()
mask.show()


#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.


# Generate the subsampled kspace
kspace_mask = pfft.ifftshift(mask.data)
kspace_data = pfft.fft2(image.data) * kspace_mask

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(kspace_mask)

# Zero order solution
image_rec0 = pysap.Image(data=pfft.ifft2(kspace_data), metadata=image.metadata)
image_rec0.show()
# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(mask.data)


#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
max_iter = 10
dictionary_atoms = np.load("/home/loubnaelgueddari/src" +
                           "/retreat_2018_p17_conv_dict_learning/datas/" +
                           "database_2018426_1546/D1.npy")

print(dictionary_atoms.shape)

linear_op = ConvSparseCode2D(dictionary_atoms, image_shape=image.shape,
                             lbd=1e-3, verbose=False,
                             sparse_cod_iter=200, relative_tolerance=5e-3)

fourier_op = FFT2(samples=kspace_loc, shape=(512, 512))

#############################################################################
# Condata-Vu optimization
# -----------------------
#
# We now want to refine the zero order solution using a Condata-Vu
# optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the CONDAT-VU reconstruction
max_iter = 1
gradient_op_cd = GradAnalysis2(data=kspace_data,
                               fourier_op=fourier_op)
x_final, transform = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    std_est=None,
    std_est_method="dual",
    std_thr=2.,
    mu=0,
    tau=1.6326340031341833,
    sigma=0.5,
    relaxation_factor=1.0,
    nb_of_reweights=0,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)

image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
