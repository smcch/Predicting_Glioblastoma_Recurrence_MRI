# Predicting regions of local recurrence in glioblastomas using voxel-based radiomic features of multiparametric postoperative MRI

This repository contains the Python implementation of the paper: Cepeda S, Luppino LT, Pérez-Núñez A, Solheim O, García-García S, Velasco-Casares M, Karlberg A, Eikenes L, Sarabia R, Arrese I, Zamora T, Gonzalez P, Jiménez-Roldán L, Kuttner S. Predicting Regions of Local Recurrence in Glioblastomas Using Voxel-Based Radiomic Features of Multiparametric Postoperative MRI. Cancers. 2023; 15(6):1894. https://doi.org/10.3390/cancers15061894 (https://www.mdpi.com/2072-6694/15/6/1894)

## Overall idea

This model uses as input the voxelwise radiomic features of the non-enhancing peritumoral region of glioblastomas extracted from multiparametric structural MRI. As output, the probability for each voxel of becoming a site of future tumor recurrence is obtained. The probabilities are represented through color-coded maps. In addition, a segmentation of the regions identified as high-risk by the model is generated.

## Prerequisites

Raw MRI sequences need to be pre-processed according to the following pipeline:

1. DICOM to NifTI conversion
2. LPS-RAI reorientation
3. Coregistration
4. Skull stripping
5. Intensity normalization (z-score)

After preprocessing, segmentation of the following structures is mandatory: a) peritumoral region, b) tumor core (enhancing volume + necrosis) or surgical cavity depending on whether it is a preoperative or postoperative study.

We strongly recommend carrying out the preprocessing using CaPTk https://cbica.github.io/CaPTk/, which has tools for conversion to NifTI files and the BraTS pipeline, which includes the aforementioned steps.

![imagen](https://user-images.githubusercontent.com/87584415/206718950-1141f2c9-3501-40c8-a91e-10881642a008.png)
 

The normalized volumes in the output directory should be renamed as follows: t1.nii.gz, t1ce.nii.gz, t2.nii.gz, flair.nii.gz. In addition, the automatic segmentation performed by the software must be corrected manually if necessary. It is essential to separate the labels to create previously mentioned structures (peritumor.nii.gz, tumor or cavity.nii.gz).
The ADC map sequence is not supported in the BraTS pipeline offered by CapTK. Please follow these pre-processing steps after DICOM to NifTI conversion:
- Skull stripping: can be done using the BET function of FSL https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL. Or through the 3D Slicer - Swiss skull strip plugin https://github.com/lassoan/SlicerSwissSkullStripper.
- Co-registration to the t1ce.nii.gz. This function can be executed through CapTK or FSL.

Below is an example of the volumes and segmentation that will be used as input.
 
![imagen](https://user-images.githubusercontent.com/87584415/206718200-76151ded-36a5-4689-9724-f2dc7db9d781.png)


## How to use

### Python

##### Requirements

In order to run this code, Python 3.9.15 or above is required. The Python packages listed in `requirements.txt` are also necessary. One can install them by running
```
pip install -r requirements.txt
```

A folder named `Patients` containing the patients' data to be analysed is expected to be placed in the same folder as the code. `Patients` should contain a subfolder for each of the patients. Each patient's folder must contain:

* t1ce.nii.gz
* t1.nii.gz
* t2.nii.gz
* flair.nii.gz
* adc.nii.gz
* peritumor.nii.gz
* tumor.nii.gz **_OR_** cavity.nii.gz


#### Usage

The main function is in `main.py`, and to run the code over all the patients, one must run 
```
python main.py
```
from within the code folder. Once the script is done, each patient will have their results stored within their respective subfolder.

### Docker

A Docker image for arm64 architectures (Apple M1) was already created with this Dockerfile: `llu025/gbm:repo`<sup>[1](https://hub.docker.com/layers/llu025/gbm/repo/images/sha256-3bdcf2ed2663dcf48cca49c8a34459d3a56f1eba93b9f89cfd313938be7c25dd?context=explore)</sup>.
It can be pulled from [hub.docker.com](hub.docker.com) by running
```
docker pull llu025/gbm:repo
```
For other architectures (amd64), one can create a Docker image by running
```
docker build -t <repo_name>/<image_name>:<tag_name> -f Dockerfile .
```
where `<repo_name>`, `<image_name>`, and `<tag_name>` are specified by the user.

Once the Docker image is ready, either by pulling it from the hub or by building it, one can run
```
docker run --rm -ti -v </path/to/the/patients/data/>:/usr/src/app/Patients <repo_name>/<image_name>:<tag_name> python3 main.py
```
where `<repo_name>/<image_name>:<tag_name>` is the same as above, and `</path/to/the/patients/data/>` can be any path where the data is stored, so in this case the data does not have to be moved or copied into the code folder.
The result is the same as with the pure Python approach: each patient will have their results stored within their respective subfolder.


Mind that `-v /source/folder/:/destination/folder` maps the content of the source folder from the host filesystem into the destination folder of the Docker container, so it does not create copies nor move any data.
Also, mind that `</path/to/the/patients/data/>` has to end with `/`, otherwise the parent folder will be mapped into `/usr/src/app/Patients` instead of the patients' subfolders.

