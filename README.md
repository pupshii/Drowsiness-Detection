<h1 align="center">Real-time object detection using Nvidia Jetson Xavier</h1>

<div align='center'>
    <a href='https://github.com/pupshii' target='_blank'><strong>Poonyavee Wongwisetsuk</strong></a><sup></sup>&emsp;
</div>

<div align='center'>
    King Mongkut's University of Technology Thonburi
</div>

<div align='center'>
    In collaboration with Kanazawa University 
</div>

## Introduction
This is a code repository containing Pytorch implementation of the paper Real-time object detection using Nvidia Jetson Xavier. 

## Getting Started
### 1. Equipment preparation
Turn on the Nvidia Jetson Xavier and connect a webcam to the system.
Open the terminal and follow the next steps
### 2. Clone the code
```bash
git clone https://github.com/pupshii/Drowsiness-Detection
cd Drowsiness-Detection
```
### 3. Prepare the environment
```bash
# create env using conda
conda create -n Drowsiness-Detection python==3.9.18
conda activate Drowsiness-Detection

# install dependencies with pip
pip install -r requirements.txt
```
### 4. Inference
```bash
python drowsydetect.py
```


