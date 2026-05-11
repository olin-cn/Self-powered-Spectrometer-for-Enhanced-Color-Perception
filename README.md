# Photovoltaic Spectrometer for Enhanced Color Perception





This repository provides supporting code and example data associated with our work on a vdWH photovoltaic spectrometer. The repository is intended for academic reference, reproducibility verification, and understanding the computational workflows described in the manuscript and Supplementary Information.



##### \# Repository Overview



The repository contains two main computational workflows:



1. Spectral reconstruction workflow



&#x20;- MATLAB / Python based



&#x20;- Used for monochromatic and broadband spectral reconstruction





2\. CIFAR-10 image classification workflow



&#x20;- Python / TensorFlow based



&#x20;- Used for evaluating spectrometer-derived multispectral representations in Vision Transformer (ViT) image classification.



##### \# Important Clarification



The code provided in this repository should be regarded as a proof-of-concept computational workflow intended to reproduce the main computational procedures and representative results reported in the manuscript.



The “spectrometer-derived” multispectral inputs used in Fig. 5 are not obtained through direct physical imaging using the single-pixel spectrometer device. Instead, they are simulated multispectral representations generated from CIFAR-10 RGB images using the experimentally measured wavelength-dependent response characteristics of the vdWH photovoltaic spectrometer.



This workflow is intended to evaluate whether wavelength-resolved sensing enabled by the experimentally measured device response can improve image-classification performance after conversion into multispectral representations.



##### \# Tested Environment



```text



Python >= 3.9



TensorFlow >= 2.x



MATLAB 



 Repository Structure

.

├── monochromatic\_spectra\_reconstruction/

│   ├── Monochromatic\_spectra\_reconstruction.m

│   ├── ResponseMatrix.xlsx

│   └── narrow360-1000.mat

│

├── broadband\_spectra\_reconstruction/

│   ├── Broadband\_spectra\_reconstruction.py

│   ├── ResponseMatrix.txt

│   ├── MeasuredSignals.txt

│   └── Refspectra.txt

│

├── cifar10\\\_classification/

│   ├── Train.py

│   ├── cifar-10-python.tar.gz

│   ├── results/

│   └── spectral\\\_cache/

│

└── README.txt



##### \# Spectral Reconstruction Workflow



The spectral reconstruction workflow reconstructs incident spectra from experimentally measured photovoltaic response vectors using a response matrix and regularized non-negative optimization.



Both monochromatic and broadband spectral reconstruction use the same overall framework:



\- Measured response vector



\- Response matrix



\- Gaussian spectral basis representation



\- Regularized non-negative optimization



\- Reconstructed spectrum



###### **A. Monochromatic Spectral Reconstruction (MATLAB)**



The MATLAB workflow demonstrates the basic monochromatic reconstruction process using experimentally measured response matrices and spectral basis functions.



**Required Files**



* monochromatic\_spectra\_reconstruction.m



* ResponseMatrix.xlsx



* narrow360-1000.mat



This workflow reconstructs representative monochromatic spectra and compares reconstructed spectra with reference spectra.



###### **B. Broadband Spectral Reconstruction (Python)**



The Python workflow demonstrates broadband spectral reconstruction using experimentally measured photovoltaic response vectors.



**Required Files**



* Broadband\_spectra\_reconstruction.py



* ResponseMatrix.txt



* MeasuredSignals.txt



* Refspectra.txt



where:



\* ResponseMatrix.txt contains the experimentally measured wavelength-dependent response matrix;

\* MeasuredSignals.txt contains broadband photovoltaic response vectors;

\* Refspectra.txt contains the corresponding reference spectra.



The script reconstructs broadband spectra from measured response vectors and compares reconstructed spectra with reference spectra.



##### **# CIFAR-10 Image Classification Workflow**



\- Implemented in Python / TensorFlow.



\- Corresponds to the image classification workflow described in Supplementary Note 6 and the related computational results in Fig. 5.



\- The code supports generating and comparing three types of inputs: normal RGB inputs, spectrometer-derived multispectral inputs, and simulated color-vision-deficiency inputs.



\- All three input types are trained using the same Vision Transformer classification network to ensure a fair comparison.



###### **## Python Environment Requirements**



The image classification code requires the following Python dependencies:



```bash



pip install numpy pandas matplotlib scikit-learn tqdm tensorflow



```



**Recommended environment:**



```text



Python >= 3.9



TensorFlow >= 2.x



NumPy



Pandas



Matplotlib



Scikit-learn



tqdm



```



A TensorFlow environment with GPU support is recommended to accelerate training. The script automatically detects whether a GPU is available. If no GPU is detected, the program will continue running on the CPU.



###### **## Dataset Preparation**



The image classification workflow uses the CIFAR-10 Python version dataset.



Please download the CIFAR-10 Python archive:



```text



cifar-10-python.tar.gz



```



and place it in the same directory as `Train.py`.



The default path read by the script is:



```text



./cifar-10-python.tar.gz



```



The code does not automatically download the CIFAR-10 dataset, so users need to prepare this file in advance.



###### **## Overview of the Image Classification Workflow**



The overall workflow of the Python code is as follows:



1. Read CIFAR-10 images from the local `cifar-10-python.tar.gz` file.



2\. Normalize RGB pixel values to the `\\\[0, 1]` range.



3\. Generate one of three input types according to `CONFIG.MODE`:



&#x20; - normal RGB input;



&#x20; - spectrometer-derived multispectral input;



&#x20; - simulated protanopia input.



4\. Train using the same Vision Transformer model.



5\. Save training logs, model files, confusion matrices, and training curves.



6\. Compare image classification performance under different input representations.



To ensure a fair comparison, the same network architecture and training workflow are used for different modes, except for the different number of input channels.



###### **## Configuration**



The main experimental parameters are defined in the `CONFIG` class in `Train.py`:



```python



class CONFIG:



&#x20;  MODE = 3



&#x20;  IMG\\\_SIZE = 32



&#x20;  PATCH\\\_SIZE = 4



&#x20;  N\\\_SPECTRAL\\\_BANDS = 24



&#x20;  SPECTRAL\\\_WIDTH = 50



&#x20;  PROJECTION\\\_DIM = 192



&#x20;  TRANSFORMER\\\_LAYERS = 6



&#x20;  NUM\\\_HEADS = 8



&#x20;  MLP\\\_RATIO = 4



&#x20;  EPOCHS = 100



&#x20;  BATCH\\\_SIZE = 128



&#x20;  LEARNING\\\_RATE = 3e-4



&#x20;  DROPOUT\\\_RATE = 0.3



&#x20;  WEIGHT\\\_DECAY = 1e-3



&#x20;  LABEL\\\_SMOOTHING = 0.2



&#x20;  SAVE\\\_DIR = "results"



&#x20;  CACHE\\\_DIR = "spectral\\\_cache"



&#x20;  REQUIRE\\\_GPU = True



```

###### 

###### **## Input Modes**



Three input modes are supported:



| Mode       | Input Type                               |



| ---------- | ---------------------------------------- |



| `MODE = 1` | RGB input                                |



| `MODE = 2` | Spectrometer-derived multispectral input |



| `MODE = 3` | Simulated protanopia input               |



For the spectrometer-derived mode:



1\. CIFAR-10 RGB images are normalized to the \\\[0,1] range;

2\. RGB images are converted into wavelength-resolved multispectral representations spanning 400–1000 nm using spectral basis functions derived from the experimentally measured device response;

3\. Photon noise and readout noise are added to emulate realistic sensing conditions;

4\. PCA dimensionality reduction is applied before ViT classification.



The PCA and normalization parameters are fitted only on the training dataset and then applied to the test dataset to ensure consistent training and evaluation procedures.



##### **# Reproducibility Notes**



The Python workflow fixes the random seed:



&#x09;SEED = 42



Minor variations may still occur due to differences in:



\* TensorFlow version

\* CUDA / cuDNN version

\* GPU hardware

\* low-level non-deterministic operators

\* differences caused by using AdamW or Adam.



To facilitate reproducibility, it is recommended to record the following information:



```text



Python version



TensorFlow version



CUDA version



GPU model



CONFIG.MODE



CONFIG.EPOCHS



CONFIG.BATCH\\\_SIZE



CONFIG.LEARNING\\\_RATE



```



##### **# Current Limitations**



The code provided in this repository is intended as a proof-of-concept computational workflow for demonstrating the core reconstruction and multispectral image-processing concepts described in the manuscript. The current implementation is designed primarily for reproducibility, transparency, and methodological illustration rather than for optimized deployment or real-time applications.



Several limitations should be noted:



\- the spectral reconstruction workflow currently uses simplified Gaussian spectral basis functions and regularized non-negative optimization, which may exhibit reduced robustness under high-noise conditions or for highly complex broadband spectra;



\- the broadband reconstruction examples are based on representative experimentally measured response matrices and example input spectra rather than a fully integrated hardware acquisition pipeline;



\- the CIFAR-10 multispectral workflow uses simulated multispectral representations derived from experimentally measured wavelength-dependent response characteristics, rather than direct multispectral imaging acquired by the experimental device;



\- the current machine-learning workflow is intended to evaluate the potential benefits of wavelength-resolved sensing and does not yet represent a fully optimized multispectral computer-vision framework;



\- reconstruction speed, memory usage, and model efficiency have not yet been optimized for embedded or edge-computing applications.



Future improvements may include:



\- physics-informed neural-network-based spectral reconstruction;



\- improved noise-aware reconstruction algorithms;



\- direct integration with experimentally acquired multispectral imaging datasets;



\- hardware-accelerated real-time reconstruction;



\- optimized multispectral machine-learning pipelines for practical sensing applications.



##### **# Code Usage Policy**



Any use, copying, modification, redistribution, or derivative use of this code requires prior permission from the corresponding author.



This repository is intended solely for academic reference. Without explicit written permission, this code may not be used for commercial purposes, public redistribution, or integration into other projects.



##### **# Citation**



If this repository is helpful for your academic research, please cite the paper corresponding to this project.



Formal citation information will be added after the paper is published.



##### **# Contact**



This repository is provided for academic research and reproducibility purposes.



For questions regarding code usage or collaboration, please contact the corresponding author.





