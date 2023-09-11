# Hololens Registration Model

## Overview
This repository contains a Hololens registration model that uses deep learning techniques to align 3D point clouds. The registration model can be trained on simulated data and tested on actual or simulated point clouds. The provided GUI allows users to easily fine-tune the model and evaluate its performance.

## Getting Started

### Prerequisites
- Python 3.x
- Dependencies listed in `requirements.txt`

### Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/Hololens_registration_model.git
   cd Hololens_registration_model

### Usage

### GUI
Run the following command to start the GUI:
```sh
python Hololens_registration_model/src/GUI.py
```
The GUI operates in two modes: Train and Test.

#### Train Mode
In Train mode, users can fine-tune the model on simulated data. This is useful when you have a new 3D segmented model of a patient and want to adapt the model to this new patient. The following parameters are relevant to the simulated data for model fine-tuning:

Input STL file location: File location of the 3D point cloud file of the 3D preoperative model of the patient.
Number of sampled points: Number of points to sample from the STL file.
Maximum allowable rotation (Rotation Magnitude): Maximum allowed rotation while creating simulated training data.
Maximum allowable translation (Translation Magnitude): Maximum translation allowed for the simulated target point cloud.
Visibility control: Controls the visibility of the simulated target point cloud.
Noise introduction: Simulates noise in the scene, representing the patient lying on a table surrounded by spherical noise.
Maximum number of noise spheres: Maximum number of noise spheres allowed around the simulated target point cloud.
Number of iterations for the RPMNet algorithm: Specify the number of iterations for the RPMNet algorithm.
Model file path for fine-tuning: Path to the model file that you want to fine-tune.

#### Test Mode
The GUI offers two testing modes: users can evaluate the model using actual data or simulate testing with predefined parameters. When using simulated data, the source point cloud undergoes manipulation based on the simulation parameters previously described in the Train mode. Additional options in the test mode include:

Input Target file (3D point cloud file): The 3D point cloud file constructed from any 3D reconstruction method.
Visualize: Enables visualization of the registered point cloud.
Save Mesh: Allows users to choose whether to save the mesh of the registered point cloud scene.
Capture RGB: Saves a rendered image of the 3D registration scene.
Simulated: Switches between using simulated data or actual data.
