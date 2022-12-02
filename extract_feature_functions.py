import radiomics

radiomics.setVerbosity(40)

from radiomics import featureextractor
import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import warnings
import traceback


def create_dataset(DATADIR):
    """
    Function to loop over all the subfolders of DATADIR, where each subfolder is supposed to represent a patient.
    Feature extraction is performed for those patients whose radiomic features have not been already extracted.

    Input:
        - DATADIR: path to the folder with patients to be analysed.
    Output:
        - patients: list of paths to each patient subfolder.
    """

    # List of DATADIR subfolders, files within DATADIR are ignored
    patients = [os.path.join(DATADIR, patient) for patient in next(os.walk(DATADIR))[1]]

    # List of patients whose features were already extracted
    parquets = [
        patient
        for patient in patients
        if "voxel_features.parquet" in os.listdir(patient)
    ]

    # Patients whose features are missing
    missing = list(set(patients) - set(parquets))
    for patient in (pbar := tqdm(missing)):
        pbar.set_description(f"Processing missing patient {os.path.basename(patient)}")
        patient_feature_extraction(patient)

    return patients


def patient_feature_extraction(patient):
    """
    Function to extract features for one patient. Six files are expected in the patient folder:
    [adc.nii.gz, flair.nii.gz, t2.nii.gz, t1.nii.gz, t1ce.nii.gz, peritumor.nii.gz]
    The function then loops over the MRI sequences and creates the feautes for all voxels within the peritumor.
    The features are then stacked in the columns of a Pandas DataFrame, where each row is a peritumor voxel.

    Input:
        - patient: path to a patient folder.
    Output:
        - patient_df: Pandas DataFrame containing the patient's extracted features.
    """

    patient_df = []
    seg_path = os.path.join(patient, "peritumor.nii.gz")

    # Loop MR sequences
    for mri_seq in (pbar2 := tqdm(["adc", "flair", "t2", "t1", "t1ce"], leave=False)):
        pbar2.set_description(f"Processing {mri_seq}")
        load_path = os.path.join(patient, f"{mri_seq}.nii.gz")

        # Extraction of the features for sequence mri_seq
        features_df = extract_features(load_path, seg_path)
        features_df = features_df.add_prefix(mri_seq + "_")
        patient_df.append(features_df)

    # The final DataFrame contains the features from all the MRI sequences
    patient_df = pd.concat(patient_df, axis=1)
    patient_df.to_parquet(os.path.join(patient, "voxel_features.parquet"))

    return patient_df


def extract_features(IMG_PATH, SEG_PATH):
    """
    This function extracts radiomic features for an MRI image IMG_ARRAY, but only for the voxels belonging
    to the segmentation SEG_ARRAY. The feature extractor is configured with "Params.yaml" in the same folder.
    All the features are stacked on the columns axis of a Pandas DataFrame, where each row is a voxel within
    the segmentation.

    Input:
        - IMG_PATH: path to the MRI sequence.
        - SEG_PATH: path to a segmentation which is valid for the same MRI sequence. In this framework
                    it is supposed to be a peritumoral region of a glioblastoma.
    Output:
        - features_df: Pandas DataFrame containing the MRI sequence's extracted features.
    """

    # Input image and segmentation image prepared.
    IMG_ARRAY = nib.load(IMG_PATH).get_fdata()
    SEG_ARRAY = nib.load(SEG_PATH).get_fdata()
    SEG_ARRAY[SEG_ARRAY != 0] = 1
    VOX_SIZE = nib.load(IMG_PATH).header.get_zooms()

    IMG = sitk.GetImageFromArray(IMG_ARRAY)
    IMG.SetSpacing((int(VOX_SIZE[0]), int(VOX_SIZE[1]), int(VOX_SIZE[2])))
    IMG.SetOrigin((1, 1, 1))

    SEG = sitk.GetImageFromArray(SEG_ARRAY)
    SEG.SetSpacing((int(VOX_SIZE[0]), int(VOX_SIZE[1]), int(VOX_SIZE[2])))
    SEG.SetOrigin((1, 1, 1))

    # Instantiate the feature extractor
    PRM_PATH = "Params.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(PRM_PATH)

    # Set the non-calculated voxels to NaN
    extractor.settings["initValue"] = np.nan

    # Extract features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        features = extractor.execute(IMG, SEG, voxelBased=True)

    # Loop over all feature images (sitk.Image) and stack them columnwise
    features_df = []
    for feature_name, feature_image in tqdm(
        features.items(), desc="Features", leave=False
    ):
        # Some variables produced by the extractor are not features (sitk.Image instances),
        # so they are filtered out
        if isinstance(feature_image, sitk.Image):

            # This step ensure that the same 3D dimensions are maintained
            feature_image = sitk.Resample(
                feature_image,
                IMG,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                np.nan,
                feature_image.GetPixelID(),
            )

            # Each feature is a list of values, one per voxel
            new_feature = pd.Series(
                sitk.GetArrayFromImage(feature_image).flatten(), name=feature_name
            )
            features_df.append(new_feature.dropna())

    features_df = pd.concat(features_df, axis=1)
    return features_df


def retrieve_patient_data(patient, scaler):
    """
    Retrieve the patient's radiomic features, impute the missing NaN values with the
    mean over the whole peritumor, and apply the scaler to the features.

    Input:
        - patient: path to a patient folder.
        - scaler: feauter scaler.
    Output:
        - data: voxel features imputed and scaled.
    """

    # Retrieve data, fill NaN with mean values.
    data = pd.read_parquet(os.path.join(patient, "voxel_features.parquet"))
    values = data.values
    col_means = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(col_means, inds[1])

    # Apply the scaler
    index = data.index
    data = scaler.transform(data)
    col_names = scaler.get_feature_names_out()
    data = pd.DataFrame(data, columns=col_names, index=index)

    return data


if __name__ == "__main__":
    try:
        data_path = "Patients"
        dataset = create_dataset(data_path)
    except:
        traceback.print_exc()
        print("\a")

    print("\a")
