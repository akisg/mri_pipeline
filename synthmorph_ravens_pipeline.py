#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import io
import surfa as sf
import tensorflow as tf
import voxelmorph as vxm
import matplotlib.pyplot as plt
import pandas as pd
import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk



# Read environment variable for Freesurfer Prefix on Cubic
freesurfer_prefix = os.getenv('FREESURFERSIF', '')
print(f"The value of $FREESURFERSIF is: {freesurfer_prefix}.")

### -- Input Images -- ###

# Read environment variable for input directory
input_dir = os.getenv('INPUT_DIR', '../inputs')
print(f"The value of $INPUT_DIR is: {input_dir}.")

# Read environment variable for output directory
output_dir = os.getenv('OUTPUT_DIR', 'out_synth')
print(f"The value of $OUTPUT_DIR is: {output_dir}.")


# CSF Test Input Files and Template
subject_nifti_paths = {
    "subj1": f"{input_dir}/csf_test/OAS30574_MR_d1917/OAS30574_MR_d1917_T1.nii.gz",
    # "subj2": f"{input_dir}/csf_test/OAS30597_MR_d3137/OAS30597_MR_d3137_T1.nii.gz",
    # "subj3": f"{input_dir}/csf_test/OAS31169_MR_d0620/OAS31169_MR_d0620_T1.nii.gz",
}
template_nifti_path = f"{input_dir}/csf_test/OAS30001_MR_d0129/OAS30001_MR_d0129_T1.nii.gz"

# Inverted Input Files and Template
# subject_nifti_paths = {
#     "subj1": f"{input_dir}/inverted/in/subj1/subj1_T1_LPS.nii.gz",
#     # "subj2": f"{input_dir}/inverted/in/subj2/subj2_T1_LPS.nii.gz",
#     # Add more as needed
# }
# template_nifti_path = f"{input_dir}/inverted/template/colin27_t1_tal_lin_INV.nii.gz"


# Yuhan's Input Files and Template
# subject_nifti_paths = {
#     "subj1": f"{input_dir}/mri_samples/T1/137_S_4227_2011-09-21_T1_LPS.nii.gz", # extremely high VN scan
#     # "subj2": f"{input_dir}/mri_samples/T1/013_S_4236_2011-10-13_T1_LPS.nii.gz", # extremely high VN scan
#     # "subj3": f"{input_dir}/mri_samples/T1/019_S_6635_2020-01-28_T1_LPS.nii.gz", # extremely small VN scan
#     # "subj4": f"{input_dir}/mri_samples/T1/072_S_4226_2012-10-12_T1_LPS.nii.gz", # extremely small VN scan
#     # "subj5": f"{input_dir}/mri_samples/T1/023_S_1247_2007-02-21_T1_LPS.nii.gz", # mean VN scan
#     # "subj6": f"{input_dir}/mri_samples/T1/027_S_0118_2008-02-23_T1_LPS.nii.gz", # mean VN scan
# }
# template_nifti_path = f"{input_dir}/mri_samples/template/BLSA_SPGR+MPRAGE_averagetemplate.nii.gz"

### -- Options -- ###
# segmentation = 'fast' # Use Fast segmentation from the FSL installation
segmentation = 'synthseg_freesurfer' # Use SynthSeg from the FreeSurfer installation
# segmentation = 'synthseg_github'  # Use SynthSeg from the GitHub repository (NOT WORKING CURRENTLY)

# affine = 'synthmorph_freesurfer' # Synthmorph affine registration from the FreeSurfer installation
affine = 'itk'  # ITK-based affine registration using SimpleITK
# affine = 'flirt' # NOT WORKING CURRENTLY

# deformable = 'synthmorph_freesurfer'
deformable = 'synthmorph_voxelmorph'

skull_strip = False

SHOW_IMAGES = False



# # Packages from GitHub.
# !pip -q install git+https://github.com/adalca/neurite.git@0776a575eadc3d10d6851a4679b7a305f3a31c65
# !pip -q install git+https://github.com/freesurfer/surfa.git@ec5ddb193fd1caf22ec654c457b5678f6bd8e460
# !pip -q install git+https://github.com/voxelmorph/voxelmorph.git@2cd706189cb5919c6335f34feec672f05203e36b
# !pip -q install nilearn

# Download models
# !curl -O https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5
# !curl -O https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/brains-dice-vel-0.5-res-16-256f.h5


# !pip install tensorflow==2.18
# !pip install keras==3.08

# get_ipython().system('pip show tensorflow')
# get_ipython().system('pip show keras')


# Helper functions. The shape has to be divisible by 16.
# shape = (128, 128, 128)
# shape = (160, 192, 160)

shape = (256, 256, 256)

def normalize(x):
    x = np.float32(x)
    x -= x.min()
    x /= x.max()
    return x[None, ..., None]

def show(x, title=None):
    if not SHOW_IMAGES:
        return
    x = np.squeeze(x)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    i, j, k = (f // 2 for f in x.shape)

    slices = x[:, j, :], x[:, :, k], x[i, :, :].T
    for x, ax in zip(slices, axes):
        ax.imshow(x, cmap='gray')
        ax.set_axis_off()

    if title:
        axes[1].text(0.50, 1.05, title, ha='center', transform=axes[1].transAxes, size=14)




# Helper function to get all paths for a subject
def get_subject_paths(subj_id):
    base = f"{output_dir}/{subj_id}/init/"
    lin_reg = f"{output_dir}/{subj_id}/lin_reg/"
    def_reg = f"{output_dir}/{subj_id}/def_reg/"
    return {
        "t1": base + f"{subj_id}_t1.nii.gz",
        "t1_seg": base + f"{subj_id}_t1_seg.nii.gz",
        "t1_mask": base + f"{subj_id}_t1_mask.nii.gz",
        "t1_lin_reg": lin_reg + f"{subj_id}_t1_lin_reg.nii.gz",
        "t1_trans": lin_reg + f"{subj_id}_t1_trans.lta",
        "t1_seg_lin_reg": lin_reg + f"{subj_id}_t1_seg_lin_reg.nii.gz",
        "t1_def_field": def_reg + f"{subj_id}_t1_def_field.nii.gz",
        "t1_def_reg": def_reg + f"{subj_id}_t1_def_reg.nii.gz",
        "t1_seg_def_reg": def_reg + f"{subj_id}_t1_seg_def_reg.nii.gz",
        "jac_det": def_reg + f"{subj_id}_jac_det.nii.gz",
        "ravens": def_reg + f"{subj_id}_t1_RAVENS.nii.gz",
        "ravens_temp": def_reg + f"{subj_id}_t1_RAVENS_temp.nii.gz",
    }

# Labels for DLICV segmentation

# csf_labels = [1,4,11,46,49,50,51,52]

# 4,3rd Ventricle
# 11,4th Ventricle
# 46,CSF
# 49,Right Inf Lat Vent
# 50,Left Inf Lat Vent
# 51,Right Lateral Ventricle
# 52,Left Lateral Ventricle


# # Labels for Fast segmentation

# csf_labels = [1]

# white_matter_labels = [2]

# gray_matter_labels = [3]

# background_label = [0]

# # Labels for SynthSeg segmentation

# # Cerebrospinal Fluid (CSF)
csf_labels = [4, 5, 14, 15, 24, 43, 44]

# # Gray Matter (includes cortex and deep gray matter structures)
# gray_matter_labels = [3, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28, 42, 47, 49, 50, 51, 52, 53, 54, 58, 60]

# # White Matter
# white_matter_labels = [2, 7, 41, 46]

# # Background
# background_label = [0]


def calculate_volume_change_from_matrix(matrix: np.ndarray) -> float:
    """
    Calculates the volume change factor from a 4x4 affine matrix.

    The factor is the absolute value of the determinant of the top-left
    3x3 submatrix (the linear transformation part).
    
    Args:
        matrix: A 4x4 NumPy array representing the affine transformation.
        
    Returns:
        The volume change factor k, where new_volume = k * old_volume.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix.")
    
    # Extract the top-left 3x3 linear transformation submatrix
    linear_part = matrix[0:3, 0:3]
    
    # The volume change factor is the determinant of this submatrix
    determinant = np.linalg.det(linear_part)
    
    return abs(determinant)

def parse_flirt_mat_file(filepath: str) -> np.ndarray:
    """
    Parses a simple space-delimited text file, like a FSL FLIRT .mat file.
    """
    print(f"📄 Reading FLIRT file: {filepath}")
    try:
        # np.loadtxt is perfect for simple text-based matrices
        matrix = np.loadtxt(filepath)
        return matrix
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def parse_freesurfer_lta_file(filepath: str) -> np.ndarray:
    """
    Parses a FreeSurfer LTA file to extract the 4x4 transformation matrix.
    """
    print(f"📄 Reading LTA file: {filepath}")
    matrix_lines = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # The line '1 4 4' signals that the next 4 lines are the matrix
        start_index = lines.index('1 4 4\n') + 1
        matrix_lines = lines[start_index : start_index + 4]

        # Use numpy to parse the extracted lines
        matrix = np.loadtxt(io.StringIO("".join(matrix_lines)))
        return matrix
        
    except (ValueError, IndexError) as e:
        # Handles cases where '1 4 4' is not found or file is malformed
        print(f"Error parsing {filepath}: Could not find the matrix start signal. {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with {filepath}: {e}")
        return None


def calculate_physical_volume_change(
    flirt_matrix: np.ndarray,
    moving_image_path: str,
    ref_image_path: str
) -> float:
    """
    Calculates the true physical volume change factor from a FLIRT matrix
    by incorporating the voxel dimensions of the moving and reference images.
    
    Args:
        flirt_matrix: 4x4 NumPy array from the FLIRT .mat file.
        moving_image_path: Filepath to the original moving image (.nii or .nii.gz).
        ref_image_path: Filepath to the reference image (.nii or .nii.gz).
        
    Returns:
        The physical volume change factor k.
    """
    # 1. Calculate the determinant of the linear part of the matrix
    if flirt_matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix.")
    
    linear_part = flirt_matrix[0:3, 0:3]
    matrix_determinant = np.linalg.det(linear_part)
    
    # 2. Get voxel dimensions from the NIfTI headers
    try:
        moving_img = nib.load(moving_image_path)
        ref_img = nib.load(ref_image_path)
        
        # get_zooms() returns voxel dims (dx, dy, dz, ...)
        moving_vox_dims = moving_img.header.get_zooms()[:3]
        ref_vox_dims = ref_img.header.get_zooms()[:3]
        
    except FileNotFoundError as e:
        print(f"🚨 Error: Could not find image file - {e}")
        return None
    except Exception as e:
        print(f"🚨 Error loading NIfTI files: {e}")
        return None

    # 3. Calculate the physical volume of a single voxel for each image
    moving_voxel_volume = np.prod(moving_vox_dims)
    ref_voxel_volume = np.prod(ref_vox_dims)
    
    # 4. Calculate the voxel volume ratio
    if moving_voxel_volume == 0:
        raise ValueError("Moving image voxel volume cannot be zero.")
    voxel_volume_ratio = ref_voxel_volume / moving_voxel_volume
    
    # 5. Compute the final physical volume change factor
    k_physical = abs(matrix_determinant) * voxel_volume_ratio
    
    print("--- Calculation Details ---")
    print(f"Matrix Determinant    : {matrix_determinant:.4f}")
    print(f"Moving Voxel Volume   : {moving_voxel_volume:.4f} mm³")
    print(f"Reference Voxel Volume: {ref_voxel_volume:.4f} mm³")
    print(f"Voxel Volume Ratio    : {voxel_volume_ratio:.4f}")
    print("-------------------------")
    
    return k_physical




# ## Validate Scale Factor Value from the actual data

from pathlib import Path
from typing import Union

def calculate_nifti_volume(filepath: Union[str, Path], verbose: bool = False) -> float:
    """
    Calculates the total volume of non-zero voxels in a NIfTI file.

    This function loads a NIfTI file, counts the number of voxels with a
    value other than zero, determines the volume of a single voxel from the
    file's header, and computes the total volume.

    Args:
        filepath (str | Path): The full path to the input NIfTI file 
                               (e.g., 'data/subject_01.nii.gz').
        verbose (bool): If True, prints the voxel count, volume per voxel,
                        and total volume to the console. Defaults to False.

    Returns:
        float: The total volume of the non-zero voxels, typically in mm³.
    
    Raises:
        FileNotFoundError: If the file specified by filepath does not exist.
        Exception: Catches and re-raises other potential errors from nibabel.
    """
    try:
        # Ensure the filepath is a Path object for robust handling
        nii_path = Path(filepath)
        if not nii_path.exists():
            raise FileNotFoundError(f"Error: The file was not found at {filepath}")

        # Load the NIfTI file
        nii_file = nib.load(nii_path)

        # Get the image data as a NumPy array
        data = nii_file.get_fdata()

        # 1. Count the number of non-zero voxels
        voxel_count = np.count_nonzero(data)

        # 2. Get the volume of a single voxel from the header's zoom info
        # The first three zoom values correspond to the dimensions (x, y, z)
        voxel_volume = np.prod(nii_file.header.get_zooms()[:3])

        # 3. Calculate the total volume
        total_volume = voxel_count * voxel_volume

        if verbose:
            print(f"--- Volume Calculation Details ---")
            print(f"File: {nii_path.name}")
            print(f"Number of non-zero voxels: {voxel_count}")
            print(f"Volume per voxel: {voxel_volume:.4f} mm³")
            print(f"Total volume of structure: {total_volume:.4f} mm³")
            print(f"------------------------------------")

        return total_volume

    except FileNotFoundError as e:
        print(e)
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def itk_linear_register_nifti(
    moving_image_path: str,
    fixed_image_path: str,
    transform_type: str = "affine",
    register_as_image_type = sitk.sitkFloat32,
    interpolator_type = sitk.sitkNearestNeighbor,
) -> tuple[sitk.Image, sitk.Transform]:
    """
    Performs linear registration of a moving NIfTI image to a fixed NIfTI image using ITK.

    This function aligns the moving image to the fixed image's space using a
    specified linear transformation (rigid, affine, or similarity).

    Args:
        moving_image_path (str): The file path for the moving NIfTI image (.nii or .nii.gz).
        fixed_image_path (str): The file path for the fixed (target) NIfTI image.
        transform_type (str): The type of linear transformation to use.
                                Options: "rigid", "affine", "similarity".
                                Defaults to "affine".
        register_as_image_type: The SimpleITK image type for registration.
        interpolator_type: The SimpleITK interpolator type.

    Returns:
        tuple[sitk.Image, sitk.Transform]: A tuple containing:
            - registered_image (sitk.Image): The moving image resampled to the
                                             fixed image's space.
            - final_transform (sitk.Transform): The calculated transformation.
    """
    # --- 1. Load the images ---
    moving_image = sitk.ReadImage(moving_image_path, register_as_image_type)
    fixed_image = sitk.ReadImage(fixed_image_path, register_as_image_type)

    # --- 2. Initialize the registration method ---
    registration_method = sitk.ImageRegistrationMethod()

    # --- 3. Set up the transform ---
    if transform_type.lower() == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type.lower() == "affine":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.AffineTransform(fixed_image.GetDimension()),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type.lower() == "similarity":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Similarity3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    else:
        raise ValueError(f"Unknown transform type '{transform_type}'. Options are 'rigid', 'affine', 'similarity'.")

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    # print(f"Initialized with {transform_type.capitalize()} transform.")

    # --- 4. Configure the registration components ---
    # Similarity Metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    # Interpolator
    registration_method.SetInterpolator(interpolator_type)

    # Optimizer
    lr = 1.0
    min_step = 1e-4
    num_iter = 200
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=lr,
        minStep=min_step,
        numberOfIterations=num_iter,
        estimateLearningRate=registration_method.EachIteration
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # --- 5. Setup multi-resolution framework ---
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # --- 6. Add command observers to track progress ---
    def on_iteration():
        iteration = registration_method.GetOptimizerIteration()
        metric = registration_method.GetMetricValue()
        # print(f"Iteration: {iteration:3} | Metric Value: {metric:10.5f}")

    registration_method.AddCommand(sitk.sitkIterationEvent, on_iteration)

    # --- 7. Execute the registration ---
    print("\nStarting ITK registration...")
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print("ITK registration completed.")

    # --- 8. Print the final results ---
    # print(f"\nOptimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")
    # print(f"Final metric value: {registration_method.GetMetricValue()}")
    # print("Final Transform Parameters:")
    # print(final_transform)

    # --- 9. Resample the moving image to the fixed image's space ---
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)

    registered_image = resampler.Execute(moving_image)

    return registered_image, final_transform


def reorient_to_LPS(image: sitk.Image) -> sitk.Image:
    """Reorient image to LPS coordinate system."""
    return sitk.DICOMOrient(image, 'LPS')


def parse_itk_transform(transform: sitk.Transform) -> np.ndarray:
    """
    Parse ITK transform parameters to extract the 4x4 transformation matrix.
    
    Args:
        transform: SimpleITK transform object
        
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    try:
        # Get the transform parameters
        parameters = transform.GetParameters()
        
        # For affine transform, we have 12 parameters (3x3 matrix + 3 translation)
        if len(parameters) == 12:
            # Extract the 3x3 linear part and 3 translation components
            linear_part = np.array(parameters[:9]).reshape(3, 3)
            translation = np.array(parameters[9:12])
            
            # Create 4x4 matrix
            matrix = np.eye(4)
            matrix[:3, :3] = linear_part
            matrix[:3, 3] = translation
            
            return matrix
        else:
            print(f"Warning: Unexpected number of transform parameters: {len(parameters)}")
            return None
            
    except Exception as e:
        print(f"Error parsing ITK transform: {e}")
        return None



print(f"The shape variable is: {shape}, and its type is: {type(shape)}")


model = vxm.networks.VxmDense(
    nb_unet_features=([256] * 4, [256] * 6),
    int_steps=5,
    int_resolution=2,
    svf_resolution=2,
    inshape=shape,
  )
model = tf.keras.Model(model.inputs, model.references.pos_flow)



# "shapes" variant of SynthMorph, which is trained on images synthesized from random shapes only
# model.load_weights('shapes-dice-vel-3-res-8-16-32-256f.h5')

# "brains" variant of SynthMorph, which is trained on images synthesized from brain label maps
model.load_weights('brains-dice-vel-0.5-res-16-256f.h5')


for subj_id in subject_nifti_paths.keys():
    print(f"\n===== Processing subject: {subj_id} =====\n")
    paths = get_subject_paths(subj_id)
    subj_path = paths["t1"]
    out_seg = paths["t1_seg"]
    output_filename = paths["t1_mask"]
    affine_moved = paths["t1_lin_reg"]
    matrix_filepath = paths["t1_trans"]
    seg_affine_moved = paths["t1_seg_lin_reg"]
    def_field = paths["t1_def_field"]
    def_moved = paths["t1_def_reg"]
    seg_def_moved = paths["t1_seg_def_reg"]
    jac_det_path = paths["jac_det"]
    ravens_path = paths["ravens"]
    ravens_temp_path = paths["ravens_temp"]

    # --- Input original NIfTI paths and preprocess (resize/reshape/resample) ---
    orig_subj_nii = subject_nifti_paths[subj_id]
    orig_template_nii = template_nifti_path
    preproc_subj_nii = f"{output_dir}/{subj_id}/init/{subj_id}_t1.nii.gz"
    preproc_subj_skull_stripped = f"{output_dir}/{subj_id}/init/{subj_id}_t1_skull_stripped.nii.gz"
    preproc_template_nii = f"{output_dir}/template/template_t1.nii.gz"

    # Preprocess template image
    if not os.path.exists(preproc_template_nii):
        print("Preprocessing template image ...")
        template_vol = sf.load_volume(orig_template_nii).resize(voxsize=2).reshape(shape).reorient('LPS')
        # template_vol = sf.load_volume(orig_template_nii).reshape(shape).reorient('LPS')
        # template_vol = sf.load_volume(orig_template_nii).reorient('LPS')
        os.makedirs(os.path.dirname(preproc_template_nii), exist_ok=True)
        template_vol.save(preproc_template_nii)
    else:
        print(f"Preprocessed template image exists: {preproc_template_nii}")
    
    # Preprocess subject image
    if not os.path.exists(preproc_subj_nii):
        print(f"Preprocessing subject image for {subj_id} ...")
        # subj_vol = sf.load_volume(orig_subj_nii).resize(voxsize=2).reshape(shape).reorient('LPS')
        # subj_vol = sf.load_volume(orig_subj_nii).reshape(shape).reorient('LPS')

        subj_vol = sf.load_volume(orig_subj_nii).resample_like(template_vol)
        # subj_vol = sf.load_volume(orig_subj_nii).reorient('LPS')
        os.makedirs(os.path.dirname(preproc_subj_nii), exist_ok=True)
        subj_vol.save(preproc_subj_nii)
    else:
        print(f"Preprocessed subject image exists: {preproc_subj_nii}")

    # Skull Stripping with BET
    if skull_strip:
        print(f"Skull Stripping subject image for {subj_id} ...")
        # cmd = f'bet {preproc_subj_nii} {preproc_subj_skull_stripped} -R -f 0.5 -g 0'
        # os.system(cmd)
        # subj_path = preproc_subj_skull_stripped

        import ants
        import antspynet
        import os

        # Load image
        img = ants.image_read(preproc_subj_nii)

        # Perform brain extraction
        brain_mask = antspynet.brain_extraction(img, modality="t1")

        # Apply mask and save
        skull_stripped_img = img * brain_mask
        ants.image_write(skull_stripped_img,preproc_subj_skull_stripped)
        
        subj_path = preproc_subj_skull_stripped

    else:
        subj_path = preproc_subj_nii

    # Update paths for downstream steps
    template_path = preproc_template_nii

    if segmentation == 'fast':
        print(f"Performing Fast segmentation for {subj_id} ...")
        out_seg = os.path.join(f"{output_dir}/{subj_id}/init/", f'{subj_id}_t1_seg.nii.gz')

        if not os.path.exists(out_seg):
            os.makedirs(os.path.dirname(matrix_filepath), exist_ok=True)
            cmd = f'fast --nopve -o {preproc_subj_nii} {subj_path}'
            print(f'About to run: {cmd}')
            os.system(cmd)
        else:
            print(f'Out file exists, skip: {out_seg}')
    else:
        # --- Segmentation with SynthSeg (template and subject) ---
        NUMTHD = 8
        
        if segmentation == 'synthseg_freesurfer':
            synthseg_command = "mri_synthseg"
        elif segmentation == 'synthseg_github':
            synthseg_command = "python ../SynthSeg/scripts/commands/SynthSeg_predict.py"

        # For each subject, segment both the template and the subject's T1 image
        seg_targets = [
            (f"template", template_path, f"{output_dir}/template/"),
            (subj_id, subj_path, f"{output_dir}/{subj_id}/init/")
        ]
        for cur_id, cur_img, out_base in seg_targets:
            # Segment image
            print(f'Segmenting image for {cur_id} ...')
            out_seg = os.path.join(out_base, f'{cur_id}_t1_seg.nii.gz')
            out_qc = os.path.join(out_base, f'{cur_id}_t1_Seg_QC.csv')
            out_vol = os.path.join(out_base, f'{cur_id}_t1_Seg_Vol.csv')
            out_post = os.path.join(out_base, f'{cur_id}_t1_Seg_Post.nii.gz')
            out_resample = os.path.join(out_base, f'{cur_id}_t1_Seg_Resample.nii.gz')
            cmd = f'{freesurfer_prefix} {synthseg_command} --i {cur_img} --o {out_seg} --robust --vol {out_vol} --qc {out_qc} --resample {out_resample} --threads {NUMTHD} --cpu'
            if not os.path.exists(out_seg):
                print(f'About to run: {cmd}')
                os.system(cmd)
            else:
                print(f'Out file exists, skip: {out_seg}')

    # Now use the subject's segmentation output for downstream steps
    out_seg = os.path.join(f"{output_dir}/{subj_id}/init/", f"{subj_id}_t1_seg.nii.gz")
    t1_seg = sf.load_volume(out_seg).reshape(shape).reorient('LPS')
    if SHOW_IMAGES:
        show(t1_seg, title=f'Segmented T1-weighted MRI ({subj_id})')

    # --- Create Binary Mask ---
    target_labels = csf_labels
    seg_data = t1_seg.data
    mask_data = np.isin(seg_data, target_labels).astype(np.uint8)
    affine_matrix = np.asarray(t1_seg.geom.vox2world)
    csf_mask_nii = nib.Nifti1Image(mask_data, affine_matrix)
    nib.save(csf_mask_nii, output_filename)
    print(f"Binary mask saved to: {output_filename}")
    temp1 = sf.load_volume(output_filename).reshape(shape).reorient('LPS')
    if SHOW_IMAGES:
        show(temp1, title=f'Binary mask ({subj_id})')

    # --- Affine Registration ---
    # Options: 'synthmorph_freesurfer', 'flirt', 'itk'
    affine_moving = subj_path
    affine_fixed = template_path

    # Ensure output directory exists for the transform file
    os.makedirs(os.path.dirname(matrix_filepath), exist_ok=True)

    if affine == 'synthmorph_freesurfer':
        # Estimate and save an affine transform trans.lta in FreeSurfer LTA format
        cmd1 = f'{freesurfer_prefix} mri_synthmorph register -m affine -t {matrix_filepath} {affine_moving} {affine_fixed}'
        # Apply an existing transform to an image
        cmd2 = f'{freesurfer_prefix} mri_synthmorph apply {matrix_filepath} {affine_moving} {affine_moved}'
        if not os.path.exists(affine_moved):
            print(f'About to run: {cmd1}')
            os.system(cmd1)
            print(f'About to run: {cmd2}')
            os.system(cmd2)
        else:
            print(f'Out file exists, skip: {affine_moved}')
    elif affine == 'flirt':
        matrix_filepath = paths["t1_trans"].replace('.lta', '.mat')
        cmd = f'flirt -in {affine_moving} -ref {affine_fixed} -out {affine_moved} -omat {matrix_filepath} -dof 12'
        if not os.path.exists(affine_moved):
            print(f'About to run: {cmd}')
            os.system(cmd)
        else:
            print(f'Out file exists, skip: {affine_moved}')
    elif affine == 'itk':
        # Perform ITK-based affine registration
        registered_image, final_transform = itk_linear_register_nifti(
            moving_image_path=affine_moving,
            fixed_image_path=affine_fixed,
            transform_type="affine",
            register_as_image_type=sitk.sitkFloat32,
            interpolator_type=sitk.sitkLinear
        )
        
        # Save the registered image
        sitk.WriteImage(registered_image, affine_moved)
        print(f'ITK registered image saved to: {affine_moved}')
        
        # Save the transform parameters for later use
        transform_params = final_transform.GetParameters()
        transform_matrix = parse_itk_transform(final_transform)
        if transform_matrix is not None:
            # Save transform matrix in a simple text format
            matrix_filepath = paths["t1_trans"].replace('.lta', '_itk.txt')
            np.savetxt(matrix_filepath, transform_matrix, fmt='%.6f')
            print(f'ITK transform matrix saved to: {matrix_filepath}')
        
        # Store transform for later use in segmentation application
        itk_final_transform = final_transform
        itk_transform_matrix = transform_matrix

    # --- Apply the affine step to the segmentation ---
    seg_affine_moving = output_filename
    if affine == 'synthmorph_freesurfer':
        cmd = f'{freesurfer_prefix} mri_synthmorph apply -m nearest {matrix_filepath} {seg_affine_moving} {seg_affine_moved}'
        if not os.path.exists(seg_affine_moved):
            print(f'About to run: {cmd}')
            os.system(cmd)
        else:
            print(f'Out file exists, skip: {seg_affine_moved}')
    elif affine == 'flirt':
        cmd = f'flirt -in {seg_affine_moving} -ref {affine_fixed} -out {seg_affine_moved} -init {matrix_filepath} -applyxfm'
        if not os.path.exists(seg_affine_moved):
            print(f'About to run: {cmd}')
            os.system(cmd)
        else:
            print(f'Out file exists, skip: {seg_affine_moved}')
    elif affine == 'itk':
        # Apply the ITK transform to the segmentation mask
        print(f'Applying ITK transform to segmentation mask...')
        
        # Load the segmentation mask
        seg_mask = sitk.ReadImage(seg_affine_moving, sitk.sitkUInt8)
        
        # Create resampler for the segmentation
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk.ReadImage(affine_fixed))
        resampler.SetTransform(itk_final_transform)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        
        # Apply transform to segmentation
        registered_seg = resampler.Execute(seg_mask)
        
        # Save the transformed segmentation
        sitk.WriteImage(registered_seg, seg_affine_moved)
        print(f'ITK transformed segmentation saved to: {seg_affine_moved}')

    # --- Calculate Scale Factor for the Affine Registration ---
    if affine == "synthmorph_freesurfer":
        lta_matrix = parse_freesurfer_lta_file(matrix_filepath)
        if lta_matrix is not None:
            scale_factor = calculate_volume_change_from_matrix(lta_matrix)
            print(f"📈 SynthMorph Volume Change Factor (k): {scale_factor:.4f}\n")
    elif affine == 'flirt':
        flirt_matrix = parse_flirt_mat_file(matrix_filepath)
        if flirt_matrix is not None:
            scale_factor = calculate_volume_change_from_matrix(flirt_matrix)
            print(f"📈 FLIRT Volume Change Factor (k): {scale_factor:.4f}\n")

            scale_factor = calculate_physical_volume_change(flirt_matrix, subj_path, template_path)
            print(f"📈 FLIRT Volume Change Factor (k): {scale_factor:.4f}\n")
    elif affine == 'itk':
        # The transform returned by ITK maps the fixed space to the moving space.
        # To find the volume change of the moving image, we need the determinant of the INVERSE transform.
        inverse_transform = itk_final_transform.GetInverse()
        inverse_matrix = parse_itk_transform(inverse_transform)
        
        if inverse_matrix is not None:
            scale_factor = calculate_volume_change_from_matrix(inverse_matrix)
            print(f"📈 ITK Volume Change Factor (k): {scale_factor:.4f}\n")
        else:
            print("Warning: Could not calculate ITK scale factor from inverse matrix.")
            scale_factor = 1.0


    # --- Calculate Volumes for Validation ---
    volume_1 = calculate_nifti_volume(seg_affine_moved)
    print(f"\nCalculated Volume 1: {volume_1:.4f} mm³")
    volume_2 = calculate_nifti_volume(output_filename)
    print(f"\nCalculated Volume 2: {volume_2:.4f} mm³")
    actual_scale_factor = volume_1 / volume_2
    print(f"\nCalculated Scale Factor: {actual_scale_factor:.4f} mm³")

    # --- Deformable Registration using the SynthMorph Freesurfer Command Line Interface ---
    if deformable == 'synthmorph_freesurfer':
        # Ensure output directory exists for the transform file
        os.makedirs(os.path.dirname(def_field), exist_ok=True)

        cmd1 = f'{freesurfer_prefix} mri_synthmorph register -m deform -t {def_field} {affine_moved} {template_path}'
        cmd2 = f'{freesurfer_prefix} mri_synthmorph apply {def_field} {affine_moved} {def_moved}'
        cmd3 = f'{freesurfer_prefix} mri_synthmorph apply -m nearest {def_field} {seg_affine_moved} {seg_def_moved}'
        if not os.path.exists(seg_def_moved):
            print(f'About to run: {cmd1}')
            os.system(cmd1)
            print(f'About to run: {cmd2}')
            os.system(cmd2)
            print(f'About to run: {cmd3}')
            os.system(cmd3)
        else:
            print(f'Out file exists, skip: {seg_def_moved}')

        # Open the def_field file and get the data
        def_field_data = nib.load(def_field).get_fdata()
        print(f"Def_field data shape: {def_field_data.shape}")
        print(f"Def_field data type: {def_field_data.dtype}")
        print(f"Def_field data min: {def_field_data.min()}")
        print(f"Def_field data max: {def_field_data.max()}")
        
        # Calculate the Jacobian determinant of the deformation field
        def_field_data = np.squeeze(def_field_data)  # Remove singleton dimensions if present
        


    # --- Deformable Registration using the SynthMorph VoxelMorph Interface ---
    elif deformable == 'synthmorph_voxelmorph':
        print(f'Performing Deformable Registration using VoxelMorph ...')
        # t1_fixed = sf.load_volume(affine_fixed).reshape(shape).reorient('LPS')
        t1_fixed = sf.load_volume(affine_fixed)
        # t1_moving = sf.load_volume(affine_moved).resample_like(t1_fixed)

        t1_moving = sf.load_volume(affine_moved)
        if SHOW_IMAGES:
            show(t1_fixed, title=f'Fixed T1-weighted MRI ({subj_id})')
            show(t1_moving, title=f'Moving T1-weighted MRI ({subj_id})')
        moving = normalize(t1_moving)
        fixed = normalize(t1_fixed)
        trans = model.predict((moving, fixed))
        moved = vxm.layers.SpatialTransformer(fill_value=0)((moving, trans))
        os.makedirs(os.path.dirname(def_field), exist_ok=True)
        t1_fixed.new(trans[0]).save(def_field)
        t1_fixed.new(moved[0]).save(def_moved)
        if SHOW_IMAGES:
            show(t1_moving, title=f'Moving T1-weighted MRI ({subj_id})')
            show(moved, title=f'Moved T1-weighted MRI ({subj_id})')
            show(fixed, title=f'Fixed T1-weighted MRI ({subj_id})')
            show(moved - fixed, title=f'Difference after registration 1 ({subj_id})')

        # --- Deformable Registration of Segmented Image ---
        # Define the template segmentation path based on the segmentation step
        template_seg_path = os.path.join(f"{output_dir}/template/", "template_t1_seg.nii.gz")
        t1_fixed_seg = sf.load_volume(template_seg_path).reshape(shape).reorient('LPS')
        t1_moving_seg = sf.load_volume(seg_affine_moved).resample_like(t1_fixed_seg)
        before = normalize(t1_moving_seg) - normalize(t1_fixed_seg)
        if SHOW_IMAGES:
            show(t1_moving_seg, title=f'Moving Registered T1-weighted MRI ({subj_id})')
            show(t1_fixed_seg, title=f'Fixed Registered T1-weighted MRI ({subj_id})')
            show(before, title=f'Difference before Segmented registration ({subj_id})')

        # --- Apply Deformation Field to Segmented Image ---
        moving_seg = normalize(t1_moving_seg)
        fixed_seg = normalize(t1_fixed_seg)
        moved_seg = vxm.layers.SpatialTransformer(interp_method='nearest', fill_value=0)((moving_seg, trans))
        if SHOW_IMAGES:
            show(moved_seg, title=f'Moved T1-weighted MRI ({subj_id})')
            show(fixed_seg, title=f'Fixed T1-weighted MRI ({subj_id})')
            show(moved_seg - fixed_seg, title=f'Difference after Segmented registration ({subj_id})')
        os.makedirs(os.path.dirname(seg_def_moved), exist_ok=True)
        t1_fixed.new(moved_seg[0]).save(seg_def_moved)

        # Save the deformation field data
        def_field_data = trans[0]
        print(f"Def_field data shape: {def_field_data.shape}")
        print(f"Def_field data type: {def_field_data.dtype}")
        print(f"Def_field data min: {def_field_data.min()}")
        print(f"Def_field data max: {def_field_data.max()}")

    # Calculate the Jacobian determinant of the deformation field
    jacobian_det = vxm.py.utils.jacobian_determinant(def_field_data)
    print(f"Jacobian determinant shape: {jacobian_det.shape}")
    print(f"Jacobian determinant type: {jacobian_det.dtype}")
    print(f"Jacobian determinant min: {jacobian_det.min()}")
    print(f"Jacobian determinant max: {jacobian_det.max()}")

    # Save the Jacobian determinant to a file
    jac_det_nii = nib.Nifti1Image(jacobian_det, affine_matrix)
    nib.save(jac_det_nii, jac_det_path)
    print(f"Jacobian determinant saved to: {jac_det_path}")

    # --- Calculate Ravens Map ---
    def calc_ravens(f_jac, f_seg, labels, f_out):
        nii_jac = nib.load(f_jac)
        img_jac = nii_jac.get_fdata()
        nii_seg = nib.load(f_seg)
        nii_seg_data = nii_seg.get_fdata()
        ravens = img_jac * nii_seg_data
        nii_out = nib.Nifti1Image(ravens, nii_jac.affine)
        # nib.save(nii_out, ravens_temp_path)
        # print(f"Temp RAVENS map saved to: {ravens_temp_path}")
        ravens /= scale_factor
        nii_out = nib.Nifti1Image(ravens, nii_jac.affine)
        nib.save(nii_out, f_out)
        print(f"RAVENS map saved to: {f_out}")

    calc_ravens(jac_det_path, seg_def_moved, target_labels, ravens_path)

    # --- Show RAVENS Maps ---
    if SHOW_IMAGES:
        t1_rav = sf.load_volume(ravens_path)
        show(t1_rav, title=f'RAVENS Map ({subj_id})')

    # --- Print Sums for Debugging ---
    print(f"\nThe volume of the original mask is: {volume_2}")

    nii_file = nib.load(ravens_path)
    data = nii_file.get_fdata()
    total_sum = np.sum(data)
    print(f"The sum of the values in the final RAVENS map is: {total_sum}")

    # nii_file = nib.load(ravens_temp_path)
    # data = nii_file.get_fdata()
    # total_sum = np.sum(data)
    # print(f"The sum of the values in temp RAVENS is: {total_sum}")

    # nii_file = nib.load(out_seg)
    # data = nii_file.get_fdata()
    # voxel_count = np.count_nonzero(np.isin(data, target_labels))
    # print(f"The number of voxels with a value of X in the Original Image is: {voxel_count}")
    # nii_file = nib.load(output_filename)
    # data = nii_file.get_fdata()
    # total_sum = np.sum(data)
    # print(f"The sum of the values in Original MASK is: {total_sum}")
    # nii_file = nib.load(output_filename)
    # data = nii_file.get_fdata()
    # voxel_count = np.count_nonzero(data)
    # print(f"The count of the values in Original mask is: {voxel_count}")

