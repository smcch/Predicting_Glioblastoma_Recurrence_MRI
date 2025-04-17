# 🧠📊 **GlioMap – Radiomics-Based Recurrence Prediction for Glioblastoma** 🔍🧬

### 🚀 **GlioMap v2.0** is now available for academic research use!

An innovative AI-powered tool designed to **predict regions of tumor infiltration and recurrence** in glioblastoma using **radiomic features** extracted from **multiparametric MRI**.

🔍 Supports non-invasive tumor characterization  
📊 Aims to guide personalized follow-up and treatment strategies

👉 Learn more and explore the tool: [**https://geibac.uva.es/**](https://geibac.uva.es/) 🔗

---

🔐 **License**  
**GlioMap v2.0** is distributed under the **Academic Non-Commercial Source Code License Agreement**  
© 2024 **UiT – The Arctic University of Norway** *(Licensor)*

This software is intended exclusively for **research purposes** and **not for clinical decision-making**.

---

🧠 Developed by the [GEIBAC Research Group](https://geibac.uva.es/about) – Biomedical Imaging and Computational Analysis  
🏥 In collaboration with the Neurosurgery Department of **Río Hortega University Hospital** and **UiT – The Arctic University of Norway**

---

💬 **We welcome collaboration!**  
If you're a researcher, clinician, or student interested in tumor modeling, radiomics, or glioblastoma research, join us!

📧 Get in touch: [scepedac@saludcastillayleon.es](scepedac@saludcastillayleon.e)



# Predicting regions of local recurrence in glioblastomas using voxel-based radiomic features of multiparametric MRI

This repository contains the Python implementation of the papers: 
1) Cepeda S, Luppino LT, Pérez-Núñez A, Solheim O, García-García S, Velasco-Casares M, Karlberg A, Eikenes L, Sarabia R, Arrese I, Zamora T, Gonzalez P, Jiménez-Roldán L, Kuttner S. Predicting Regions of Local Recurrence in Glioblastomas Using Voxel-Based Radiomic Features of Multiparametric Postoperative MRI. Cancers. 2023; 15(6):1894. https://doi.org/10.3390/cancers15061894 (https://www.mdpi.com/2072-6694/15/6/1894)
2) Cepeda S, Luppino L, Wodsinki M, Solheim O, Pérez-Núñez A, García-García S, Karlberg A, Eikenes L, Zamora T, Sarabia R, Arrese I, Kuttner S. NIMG-45. EXTERNAL EVALUATION OF A MACHINE LEARNING MODEL EMPLOYING RADIOMICS TO IDENTIFY REGIONS OF LOCAL RECURRENCE IN GLIOBLASTOMA FROM POSTOPERATIVE MRI. Neuro-Oncology, Volume 25, Issue Supplement_5, November 2023, Pages v195–v196, https://doi.org/10.1093/neuonc/noad179.0741
3) Cepeda S, Luppino L, Solheim O, Pérez-Núñez A, García-García S, Karlberg A, Eikenes L, Zamora T, Sarabia R, Arrese I, Kuttner S. Machine Learning-based Identification of Local Recurrence Regions in Glioblastoma using Postoperative MRI: Implications for Survival Prognostication. Brain and Spine Volume 3, Supplement 1, 2023, 101960. https://doi.org/10.1016/j.bas.2023.101960

## Overall idea

This model uses as input the voxelwise radiomic features of the non-enhancing peritumoral region of glioblastomas extracted from multiparametric structural MRI. As output, the probability for each voxel of becoming a site of future tumor recurrence is obtained. The probabilities are represented through color-coded maps. In addition, a segmentation of the regions identified as high-risk by the model is generated.

## Prerequisites

Raw MRI sequences need to be pre-processed according to the following pipeline: https://github.com/smcch/Postoperative-Glioblastoma-Segmentation

After preprocessing, segmentation of the following structures is mandatory: a) peritumoral region, b) tumor core (enhancing volume + necrosis) or surgical cavity depending on whether it is a preoperative or postoperative study.

Below is an example of the volumes and segmentation that will be used as input, and the model's output (recurrence probability map).
 
![mri_seq](https://user-images.githubusercontent.com/87584415/232340667-c31257a7-b5dc-4e88-8808-c87eae353812.jpg)

![probs](https://user-images.githubusercontent.com/87584415/232340674-2867bfeb-de14-4b11-b0a3-3406db851b65.jpg)

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


