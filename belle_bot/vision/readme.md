# Belle-bot Vision

## Overview
The purpose of the vision module is to understand and perceive the world.  Since most downstream components use ML models, the primary purpose of the vision component is to produce an embedding that downstream components can use in their ML processes. Other information can be exported but the primary objective is to produce embeddings which other systems can use whilst belle-bot is running.

## Multi-task
Whilst simply training a VAE is generally sufficient, I found there to be problems with how it would capture certain information. Often omitting distance objects like people or other information that some components may need - e.g. distant people. The bellm component would be insufficient at learning to see things if some of this information wasn't captured and therefore the richer and more accurate the embedding, the better.

## Tasks
#### Encoder
The encoder model takes sensor data (currently this is the camera and perceived depth estimation, though in future other data could be added here) and feeds it through a bottle-neck for doing multi-task modelling.

#### Decoder
The decoder model task the embedding from the encoder and projects it back to an image which allows the model to basically operate as a normal VAE so we can visualise what data is represented in the embedding.

#### Object Detection
The object detection model is trained on pre-labelled YOLO images via knowledge distillation. This allows us to add bounding boxes to our unlabelled datasets and have the model train against them creating bounding box detections from the image embedding.

#### Segmentation


#### Language Projection

The tasks
...

so we have an encoder model which takes whatever sensor information we want to include. currently camera and depth (we could include lidar in future tho). this gets encoded through a bottle neck. then we have a series of takes and decoders which do different things. object detection, face detection, image reconstruction, language modelling etc. this way we create and embedding that can be useful for multiple purposes and is more rich than just an embedding for one task 

By training a model for these tasks, the embedding captures far more rich information that belle-bot can use to understand the world.

from here we can have a series of very light weight models which share this embedding and can infer upon for all other downstream services in belle bot


faqs
- why multi task
    - to improve the models embedding for multiple tasks 
- why not just bespoke lightweight open models
    - running multiple models will increase the load in the system. even lightweight models will reuse a lot of similar low level operations which is not ideal. so instead we have one shared backbone that the model learns and reuses for all tasks
todo add section for adding taks
add section for adding input modalities

### How to add tasks
### how to fine tune a head
### Serving
### Training
### Datasets

### Todo list
- [ ] Add more metrics during training that can be used to monitor training performance.
- [ ] V3
  - [ ] Custom yolo classes for bb / segmentation so we learn and concentrate only on information we want.
  - [ ] Perform ablations on the vision stack
  - [ ] Allow for larger batch size and distributed training so we can train on cloud


The vision pipeline for belle-bot provides various real-time image processing capabilities including object detection, pose estimation, facial recognition, and image segmentation. All services communicate via the `Fabric` messaging system.


## Communication Flow

1.  **Ingestion**: Services listen to `sensors/camera` (defined in `belle_bot.sensors.cameras.config.FABRIC_ID`) for incoming RGB frames.
2.  **Preprocessing**: Frames are typically parsed using `parse_camera_stream` and resized/padded using `letterbox_image` to fit model requirements (usually 640x640).
3.  **Inference**: Models process the frame to produce detections.
4.  **Postprocessing**: Detections are rescaled back to the original camera dimensions.
5.  **Publication**: Results are published to their respective Fabric topics as JSON-encoded data.

## Getting Started

### Running a Service

Each service can be started by running its `service.py` file. For example:

```bash
python -m belle_bot.vision.object_detection.service
```

### Visualizing Outputs

Some modules include a `watcher.py` script that provides a local CV2 window to preview the detections overlaid on the camera stream.

```bash
python -m belle_bot.vision.object_detection.watcher
```



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
