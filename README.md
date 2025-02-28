# X-MARs Net: Retinal Vessel Segmentation with X-Multiple Attention Residual Network

## Introduction

This repository contains the implementation of **X-MARs Net**, a novel deep learning framework for retinal vessel segmentation, as presented in the paper *"X-MARs: X-Multiple Attention Residual Network For Retinal Vessel Segmentation"*. X-MARs Net employs a dual encoder-decoder architecture with dynamic feature compression and symmetric skip connections to enhance the detection of subtle vascular structures. Evaluated on the DRIVE and CHASE-DB1 datasets, it outperforms state-of-the-art models such as U-Net, LadderNet, and Swin-Res-Net in metrics like AUC, Accuracy, F1-score, Sensitivity, Specificity, Precision.

## Dependencies

To run this project, install the following dependencies:

- **Python**: 3.12.8 (as per the paper; 3.6+ compatible with adjustments)
- **PyTorch**: 2.5.1 with CUDA 12.6 and cuDNN 9.0 (paper-specific; 0.4+ for baseline compatibility)
- **configparser**: For parsing `hyper-parameter_CHASEDB.txt` and `hyper-parameter_DRIVE.txt`
- **h5py**: For HDF5 dataset handling
- **NumPy**: For numerical operations
- **Matplotlib**: For plotting ROC and precision-recall curves
- **scikit-image**: For image processing
- **pillow**: For image handling
Install dependencies via pip:

```bash
pip install torch==2.5.1+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install configparser h5py numpy matplotlib scikit-image pillow
```

Note: Adjust PyTorch version based on your CUDA setup if not using CUDA 12.6.

## Running the Project

### Step 1: Prepare the Dataset

1. Check the DRIVE dataset `../DRIVE/` relative to the project root.
2. Check the DRIVE dataset `../CHASEDB1/` relative to the project root.
3. Run the preprocessing script to generate HDF5 files:

```bash
python prepare_datasets_DRIVE.py
python prepare_datasets_CHASEDB1.py
```

The created files are stored in `../DRIVE_datasets/` and `../CHASEDB1_datasets/`.

### Step 2: Configure the Experiment

Edit `hyper-parameter_DRIVE.txt` and `hyper-parameter_CHASEDB.txt` to set own parameters or use prepared value.

`hyper-parameter_DRIVE.txt` shows below:

```
[data paths]
path_local =  ../DRIVE_datasets/
train_imgs_original = DRIVE_ori_train.hdf5
train_groundTruth = DRIVE_groundTruth_train.hdf5
train_border_masks = DRIVE_borderMasks_train.hdf5
test_imgs_original = DRIVE_ori_test.hdf5
test_groundTruth = DRIVE_groundTruth_test.hdf5
test_border_masks = DRIVE_borderMasks_test.hdf5

[experiment name]
name = test_DRIVE

[data attributes]
patch_height = 32
patch_width = 32

[training settings]
N_subimgs = 190000
inside_FOV = False
batch_size = 1024
nohup = True

[testing settings]
best_last = best
full_images_to_test = 20
N_group_visual = 1
average_mode = True
stride_height = 5
stride_width = 5
nohup = False
```

Additional training hyperparameters (e.g., learning rate) are defined in `XMARs_Net_main/DRIVE_TRAIN.py` and `XMARs_Net_main/CHASEDB_TRAIN.py`.

### Step 3: Train the Model

Navigate to `XMARs_Net_main/` and start training:

```bash
cd XMARs_Net_main
python DRIVE_TRAIN.py
python CHASEDB_TRAIN.py
```

**Training Setup**:
- **Hardware**: 13th Gen Intel Core i9-13900K @ 3.00 GHz, 128 GB RAM, NVIDIA GeForce RTX 4090 (CUDA 12.6, driver 561.17), Ubuntu 22.04.1 LTS.
- **Software**: Python 3.12.8, PyTorch 2.5.1 with CUDA 12.6 and cuDNN 9.0.
- **Hyperparameters**: Adam optimizer, initial learning rate 0.001, batch size 1024, 100/120 epochs, etc..

Models are saved in `XMARs_Net_main/checkpoint/`.

### Step 4: Test the Model

Generate predictions stay in `XMARs_Net_main/`:

```bash
python DRIVE_PREDICT.py
python CHASEDB_PREDICT.py
```

Results are stored in `../test_DRIVE/` and `../test_CHASE/`, including output images and visualizations.

## Project Structure

```
/X-MARs_Net/
├── hyper-parameter_CHASEDB.txt          
├── hyper-parameter_DRIVE.txt 
├── prepare_datasets_CHASEDB1.py
├── prepare_datasets_DRIVE.py  
├── XMARs_preprocess/
│   ├── extract_CHASEDB1.py
│   ├── extract_DRIVE.py
│   ├── image_handle.py
│   └──pre_processing.py
├── XMARs_Net_main/
│   ├── CHASEDB_PREDICT.py   
│   ├── CHASEDB_TRAIN.py    
│   ├── DRIVE_PREDICT.py
│   ├── DRIVE_TRAIN.py
│   ├── checkpoint/           
│   ├── XMARs_Net.py         
│   └── loss.py 
├── test_CHASE/                      
│   └── [output images and visualizations] 
├── test_DRIVE/ 
│   └── [output images and visualizations] 
├── DRIVE_datasets/      
│   └── [generated hdf5 files]   
├── CHASEDB1_datasets/      
│   └── [generated hdf5 files]  
└── README.md                 
```

## Results
The performance metrics of DRIVE dataset:

```
Area under the ROC curve: 0.9935486169167707
F1 score: 0.8728609835049293
ACCURACY: 0.9746483628876393
SENSITIVITY: 0.8547704761888275
SPECIFICITY: 0.992627619281837
PRECISION: 0.8616380024351324
```
The performance metrics of CHASE-DB dataset:
```
Area under the ROC curve: 0.9887186145168813
F1 score: 0.8595909887210576
ACCURACY: 0.9741872125928802
SENSITIVITY: 0.8178256874548113
SPECIFICITY: 0.9900563284725797
PRECISION: 0.8827150335687884
```