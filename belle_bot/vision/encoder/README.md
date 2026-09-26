# Vision Encoder

This directory contains the implementation of a Variational Autoencoder (VAE) used to compress camera images into a compact latent representation.

## Overview

The Vision Encoder project provides tools for creating datasets from replays, training a VAE model to reconstruct those images, and running inference to visualize the reconstruction quality.

### Components

- **`config/`**: Pydantic-based configuration classes for dataset creation, model architecture, training hyperparameters, and inference.
- **`dataset/`**: Scripts to fetch data from the Houston replay service and package it into WebDataset shards (TAR files).
- **`training/`**: The VAE model implementation, custom loss function (with KL annealing), and training loop integrated with MLflow.
- **`inference/`**: A script to run a trained model on replay data and generate a video showing original vs. reconstructed frames.

---

## Getting Started

### 1. Dataset Creation

The vision dataset is comprised of the rgb frame and the depth frame. The encoder uses [WebDataset](https://github.com/webdataset/webdataset) format for 
efficient data loading.

During development, it was found that the model would largely overfit to frequent frames. E.g. when belle-bot first 
turns on or is placed down or waiting in a stationary moment. The number of similar frames increases and the model 
then starts overfitting to those frames degrading the performance of the whole dataset. 

To combat this, these frequent frames are downsampled. You take the latest version of the encoder model, run the 
downsample script to create a frequency map which is used to downsample at 1/neighbours. This way, items with lots of 
similar frames will show less frequently and therefore improve the model's overall ability to learn to a low-dimensional
representation.

Step 1:
```bash
python -m belle_bot.vision.encoder.dataset.create_dataset
```
If you have a pretrained model, then run continue. If not, train the model on the generated dataset from above and come 
back here.

Step 2:
```bash
python -m belle_bot.vision.encoder.dataset.dataset_downsampler
```
This will create a frequency_map that will be used to downsample the dataset.

Step 3:  
Repeat step 1 but include the --frequency_map flag.

> For more information on the above commands, you can run them with '-h' or '--help'.

### 2. Training

To start training the VAE:

```bash
python -m belle_bot.vision.encoder.training.train
```

**Key Features:**
- **MLflow Integration**: Metrics like training loss and validation loss are logged to MLflow. Reconstructions are also saved as artifacts during evaluation steps.
- **Hardware Acceleration**: Automatically detects and uses Apple Silicon GPU (MPS), CUDA, or CPU.
- **KL Annealing**: Gradually increases the weight of the KL divergence loss to prevent posterior collapse.
- **Learning Rate Scaling**: Implements a power-law learning rate decay based on the current step.

Checkpoints are saved as `model-<step>.pt` files in the current directory.

### 3. Inference & Visualization

To verify the quality of a trained model, you can generate a side-by-side comparison video:

```bash
python -m belle_bot.vision.encoder.inference.run
```

This script:
1. Loads a specific model checkpoint (edit `run.py` to point to your `.pt` file).
2. Reads frames from local replay files.
3. Performs inference to encode and decode each frame.
4. Overlays a visualization of the latent vector (1024-dim reshaped to 32x32) on the reconstructed frame.
5. Saves the output as `belle-bot-vision-encoder.mp4`.

---

## Configuration

Configuration is managed via Pydantic models in the `config/` directory. You can override defaults using CLI arguments thanks to the `clpy` utility.

Example:
```bash
python -m belle_bot.vision.encoder.training.train --mini_batch_size 32 --learning_rate 0.0005
```

- **`VisionEncoderModelConfig`**: Controls the `embedding_size` (default: 1024).
- **`VisionEncoderTrainingConfig`**: Controls batch size, learning rate, max steps, and evaluation frequency.
- **`VisionEncoderDatasetCreationConfig`**: Controls Houston connection settings and output directories.

## Internal Structure

- `training/ml_model.py`: Contains the `VAE` and `OptimizedVAE` architectures (CNN-based).
- `training/loss.py`: Implements `OptimizedVaeLoss` which combines MSE reconstruction loss and KL divergence.
- `training/data_loader.py`: Handles WebDataset loading and image preprocessing (resizing, normalization, and jitter for training).
