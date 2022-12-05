# Predicting regions of local recurrence in glioblastomas using voxel-based radiomic features of multiparametric postoperative MRI

This repository contains the Python implementation of the paper [Predicting regions of local recurrence in glioblastomas using voxel-based radiomic features of multiparametric postoperative MRI](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#links)

## Overall idea

Here we explain what the model does, expects in input, spits in output, etc.

## Prerequisites

Here Santiago explains what the prerequisites are: preprocessing, registration, segmentation, etc.

## How to use

### Python

##### Requirements

In order to run this code, Python 3.9.15 or above is required. The Python packages listed in `requirements.txt` are also necessary. One can istall them by running
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

