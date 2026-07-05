# Object Detection

This module takes the camera frames and outputs bounding boxes for those frames to try and workout what objects within the frame are. These can then be combined at a later stage in the pipeline to understand local items in the scene.

Currently, this takes the raw camera frame and runs it through the yolo26n object detector. This is not a good detector but it's small and fast meaning that it can be used for basic object detection which we may need at a later date. This means we'll be able to detect humans. This likely will be superceeded by pose estimation but it would take minimal effort to swap over too.

### todo 
- embeddings for vision
- todo explain the trainer: https://huggingface.co/Ultralytics/YOLO26