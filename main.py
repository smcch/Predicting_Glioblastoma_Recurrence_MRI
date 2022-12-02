import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from extract_feature_functions import create_dataset, retrieve_patient_data
from skimage.filters import threshold_otsu
from utils import correct_proba, fuse_t1ce_and_proba
import traceback


def main(path, maximum_distance=20):
    """
    Main function of the repository that generates the results for each patient,
    where each patient is represented by a subfolder of a given directory.
    First, it creates the dataset by checking if the features for each patient
    have been extracted and, if not, it extracts them for all the patients
    missing them. Then, for every patient:

        - It calculates the voxle probabilities.

        - A maximum distance from the tumor/cavity can be set, beyond which
          the voxel probabilities are strongly attenuated. If set to 0, no
          attenuation is applied.

        - An automatic thresholding method (Otsu) finds an optimal threshold
          and all voxels above (below) it are labelled as recurrence (no recurrence).

        - Finally, the probabilities can be visualised alone (probabilities.nii.gz)
          or fused with the t1ce sequence (t1ce_fused_proba.dcm)

    Input:
        - path: path to a dataset of patients
        - maximum distance (mm): probabilities beyond this distance are attenuated.
          If set to 0 no attenuation is applied.
    """

    dataset = create_dataset(path)

    with open("model.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    for patient in (pbar := tqdm(dataset, leave=False)):
        pbar.set_description(f"Applying to {os.path.basename(patient)}")

        # Retrieve data, obtain probabilities, correct them
        X = retrieve_patient_data(patient, scaler)
        predicted_prob = model.predict_proba(X)[:, 1]
        if maximum_distance:
            predicted_prob = correct_proba(patient, predicted_prob, maximum_distance)

        # Otsu threshold and labelling (1= recurrence, 0= no recurrence)
        threshold = threshold_otsu(predicted_prob)
        predictions = predicted_prob > threshold

        # Save results and create images
        pd.DataFrame(
            {"predictions": predictions, "probabilities": predicted_prob},
            index=X.index,
        ).to_parquet(os.path.join(patient, "predictions.parquet"))

        fuse_t1ce_and_proba(patient)

    return None


if __name__ == "__main__":
    try:
        data_path = "Patients"
        main(data_path)
    except:
        traceback.print_exc()
        print("\a")

    print("\a")
