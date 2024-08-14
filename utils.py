import numpy as np
import nibabel as nib
import os
import pandas as pd
import traceback
import SimpleITK as sitk
import matplotlib.pyplot as plt
import time
import wandb
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_gaussian, create_pairwise_bilateral
from scipy.ndimage import gaussian_filter
from pdb import set_trace as bp


def correct_proba(patient, probabilities, maximum_distance=15):
    """
    The probabilities calculated by the model can be corrected by this function,
    which finds the minimum distance between a voxel and the tumor/cavity, and
    attenuates the probabilities accordingly. The attenuating factor decays more
    or less quickly depending on the parameter <slope> of the hyperbolic tangent.
    The steeper the slope, the faster the factor decreases. With <slope>=1 and
    <maximum_distance>=20, distances of 19mm, 20mm, and 21mm correspond to factors
    of 0.88, 0.5, and 0.12 respectively.

    Input:
        - patient: path to a patient folder. Needed to know the location of each
          peritumor voxel with respect to the tumor/cavity.
        - probabilities: the output of the model to be corrected.
        _ maximum_distance: distance (in mm) beyond which probabilities are
          attenuated. Default value: 20mm.
    Output:
        - probabilities: attenuated probabilities.
    """

    # The voxel's coordinates in 3D space have to be retrieved.
    img_shape = nib.load(os.path.join(patient, "t1ce.nii.gz")).shape
    df = pd.read_parquet(os.path.join(patient, "voxel_features.parquet"))
    index = df.index
    coordinates = np.unravel_index(index, img_shape)
    coordinates = np.array(coordinates).swapaxes(0, 1)

    # The edge of the tumor/cavity are found, no need to check the distance with
    # inner points. For each voxel, the minimum distance is found and the
    # probability corrected.
    slope = 1
    ROI_points = ROI(patient)
    for i, xyz in enumerate(coordinates):
        distance = np.min(np.linalg.norm(xyz - ROI_points, axis=1))
        probabilities[i] *= 0.5 + 0.5 * np.tanh(-slope * (distance - maximum_distance))

    return probabilities


def ROI(patient):
    """
    This function lists the voxels within a ROI (either a tumor or a surgical cavity).

    Input:
        - patient: path to a patient folder.
    Output:
        - ROI: coordinates of the voxels within the patient's ROI.
    """

    path = os.path.join(patient, "tumor.nii.gz")
    try:
        ROI = nib.load(path).get_fdata()
    except FileNotFoundError as e:
        path = os.path.join(patient, "cavity.nii.gz")
        ROI = nib.load(path).get_fdata()

    ROI = np.array(np.where(ROI)).T
    return ROI


def create_3d_image(patient):
    """
    This function creates the Nifti image with the calculated probabilities, placed
    in the correct 3D coordinate system corresponding to the patient's t1ce scan.
    The resulting image is stored in the subfolder "saved_images" within the patient's
    folder.

    Input:
        - patient: path to a patient folder.
    """
    save_path = os.path.join(patient, "saved_images")
    os.makedirs(save_path, exist_ok=True)
    IMG_NIB = nib.load(os.path.join(patient, "t1ce.nii.gz"))
    df = pd.read_parquet(os.path.join(patient, "predictions.parquet"))

    img_shape = IMG_NIB.shape
    feature_image = np.empty(img_shape)
    feature_image[:] = np.nan
    feature_image[np.unravel_index(df.index, img_shape)] = df.probabilities.values
    feature_image = nib.Nifti1Image(feature_image, IMG_NIB.affine, IMG_NIB.header)
    nib.save(feature_image, os.path.join(save_path, "probabilities"))

    return None


def fuse_t1ce_and_proba(patient, proba=None, name=None):

    """
    This function creates the DICOM file with the calculated probabilities fused
    with the patient's t1ce scan. It requires an intermediate step where the same
    fusion is stored as Nifti image.
    The resulting DICOM is stored in the subfolder "saved_images" within the patient's
    folder.

    Input:
        - patient: path to a patient folder.
    """

    # The probabilities are loaded from the Nifti image for convenience,
    # and their location saved in <mask>.The loaded probabilities are
    # grey levels (0 - 255), they have to be scaled back to 0 - 1.

    if proba is None:
        path = os.path.join(patient, "saved_images", "probabilities.nii")
        if not os.path.isfile(path):
            create_3d_image(patient)
        proba = nib.load(path).get_fdata()
    mask = ~np.isnan(proba)
    proba = (proba - np.nanmin(proba)) / (np.nanmax(proba) - np.nanmin(proba))

    # The t1ce scan is loaded and scaled between 0 - 1.
    t1ce_nib = nib.load(os.path.join(patient, "t1ce.nii.gz"))
    array = t1ce_nib.get_fdata()
    array = (array - np.min(array)) / (np.max(array) - np.min(array))

    # The turbo colormap (blue=0, red=1) gives RGBA images, the last channel
    # must be discarded, and the t1ce data repeated to have RGB. Finally,
    # the probabilities are fused into the t1ce scan.
    cmap = plt.get_cmap("turbo")
    proba = np.delete(cmap(proba), 3, -1)
    array = np.repeat(array[..., np.newaxis], 3, 3)
    array[mask, :] = proba[mask, :]

    # Trickery to make the RGB work in DICOM.
    array = (255 * array).astype(np.uint8)
    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    shape_3d = array.shape[0:3]
    new_array = array.copy().view(dtype=rgb_dtype).reshape(shape_3d)

    # The intermediate Nifti and the DICOM (with some added tags) are saved.
    new_nib = nib.Nifti1Image(new_array, t1ce_nib.affine)
    nib.save(new_nib, os.path.join(patient, "saved_images", "intermediate.nii.gz"))
    new_dicom = sitk.ReadImage(
        os.path.join(patient, "saved_images", "intermediate.nii.gz")
    )
    new_tags = [
        ("0008|0031", time.strftime("%H%M%S")),  # Series Time
        ("0008|0021", time.strftime("%Y%m%d")),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        ("0010|0020", os.path.basename(patient)),  # Patient ID
        ("0008|0060", "MR"),  # Modality
    ]
    for tag, value in new_tags:
        new_dicom.SetMetaData(tag, value)
    sitk.WriteImage(
        new_dicom,
        os.path.join(
            patient,
            "saved_images",
            "t1ce_fused_proba" + ["", "_" + name][isinstance(name, str)] + ".dcm",
        ),
    )

    return None


def _dense_gaussian_filtering(probabilities):
    """
    Test
    """
    d = probabilities[..., np.newaxis]
    d = np.concatenate((d, 1.0 - d), axis=3)
    W, H, D = d.shape[0:3]

    CD = dcrf.DenseCRF(W * H * D, 2)
    d[d == 0] = 10e-20
    U = -(np.log(d))
    U = U.transpose(3, 0, 1, 2).reshape((2, -1))
    U = U.copy(order="C")
    CD.setUnaryEnergy(U.astype(np.float32))
    scaling_dim = tuple(wandb.config["scaling_dim"] for i in range(3))
    scaling_p = (wandb.config["scaling_p"],)
    pairwise_energy_gaussian = create_pairwise_gaussian(scaling_dim, (W, H, D))
    CD.addPairwiseEnergy(pairwise_energy_gaussian, compat=wandb.config["compat"])

    # pairwise_energy_bilateral = create_pairwise_bilateral(
    #     sdims=scaling_dim, schan=scaling_p, img=probabilities, chdim=-1
    # )
    # CD.addPairwiseEnergy(pairwise_energy_bilateral, compat=wandb.config["compat"])
    Q = CD.inference(wandb.config["inference"])
    heatmap = np.array(Q, dtype=np.float32)
    heatmap = np.reshape(heatmap[0, ...], (W, H, D))
    return heatmap


if __name__ == "__main__":
    try:
        wandb.init(
            config={"compat": 3, "scaling_dim": 10, "scaling_p": 0.001, "inference": 5}
        )
        probs = nib.load("Patients/ntnu_34/saved_images/probabilities.nii").get_fdata()
        bp()
        probs[np.isnan(probs)] = 0
        filtered_probs = _dense_gaussian_filtering(probs)
        filtered_probs_save = np.copy(filtered_probs)
        filtered_probs_save[np.isnan(probs)] = np.nan
        fuse_t1ce_and_proba("Patients/ntnu_34", filtered_probs_save, "filtered")
        Y = (
            pd.read_parquet("Patients/ntnu_34/voxel_features_with_region.parquet")[
                "region"
            ]
            == "rec"
        )

        from skimage.filters import threshold_otsu
        from sklearn.metrics import roc_auc_score

        filtered_probs = filtered_probs.ravel()[Y.index]
        probs = probs.ravel()[Y.index]
        print(np.mean(np.abs(probs - filtered_probs)))
        auc = roc_auc_score(Y, filtered_probs)
        original_auc = roc_auc_score(Y, probs)
        print(auc - original_auc)
        wandb.log({"gain": auc - original_auc})

        corrected_proba = correct_proba("Patients/ntnu_34", probs)
        print(np.mean(np.abs(probs - corrected_proba)))
        corrected_auc = roc_auc_score(Y, corrected_proba)
        print(corrected_auc - original_auc)

        print(np.mean(np.abs(corrected_proba - filtered_probs)))
    except:
        traceback.print_exc()
        print("\a")

    print("\a")
