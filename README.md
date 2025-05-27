# Solar-burst-detection-using-CLIP

This project applies OpenAI’s CLIP (Contrastive Language–Image Pre-training) model to detect and classify solar radio bursts from astronomical FITS and PNG image data.

---

## Overview

CLIP is a transformer-based vision-language model trained on over 400 million image-text pairs. We use the `ViT-B/32` variant for its efficiency and performance in image classification tasks.

The objective of this project is to detect solar bursts by processing, denoising, and classifying image data using CLIP.

---

## Environment Setup


```bash
Install the required Python packages:

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Import the necessary libraries:
import numpy as np
import torch
import clip
from tqdm.notebook import tqdm

Check for GPU availability:
torch.cuda.is_available()
If a GPU is not available, the pipeline will run on CPU with reduced performance.

Model Initialization

Load the CLIP model:
model, preprocess = clip.load("ViT-B/32")
Print model specifications:
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", model.visual.input_resolution)
print("Context length:", model.context_length)
print("Vocab size:", model.vocab_size)

Data Acquisition

FITS and PNG files are downloaded using web scraping. The requests and BeautifulSoup libraries are used to collect data from specified URLs.
import os
import requests
from bs4 import BeautifulSoup

def download_file(url, save_path):
    # Download a single file from a URL
    ...

def fetch_files_from_page(base_url, file_extension, output_directory):
    # Fetch all files of a given extension from the page
    ...

def FIT_DOWNLOADER():
    # Main function to handle file downloads
    ...

Data Preparation

Decompression

Downloaded .fit.gz files are decompressed using the gzip module:
import gzip
import shutil

with gzip.open("file.fit.gz", 'rb') as f_in:
    with open("file.fit", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

FITS to PNG Conversion

FITS files are converted to PNG format using astropy and matplotlib.
from astropy.io import fits
import matplotlib.pyplot as plt

def fits_to_png(fits_path, png_path):
    ...

Image Denoising

OpenCV is used to remove noise and clean the image for better classification results.
import cv2

def denoise_image(image_path):
    ...
This step is important to reduce false detections caused by noise in the raw image data.

Classification Using CLIP

We define custom class labels and classify images using the CLIP model.
imagenet_classes = [
    "burst: Sudden, high-intensity signal variations in frequency-time plots.",
    "non burst: Background noise or stable, low-intensity variations."
]

def classify_with_clip(image, model, preprocess):
    ...

Frequency-Time Data Extraction

A utility function is provided to extract frequency-time information from FITS files:
def get_frequency_time_data(fits_file):
    ...


This helps in analyzing signal behavior over time and frequency, complementing the burst classification task.

Notes

    The ViT-B/32 variant of CLIP is used due to its strong performance on image classification tasks.

    The ResNet variant of CLIP is not used, as the task does not require recurrent or sequential modeling.

    The pipeline can run on both CPU and GPU.

    Older FITS datasets may not include PNGs; these are generated programmatically using astropy.

