# YOLOv7-TensorRT + DeepSORT
A modified DeepSORT for conveyor applications. Works on very low FPS. Assumptions made: Objects move linearly at near constant speed. Objects don't change shape or size during runtime, which are reasonable assumptions. The modification must be made by YOU directly to the [DeepSORT-realtime package](deep-sort-realtime). The modifications are found in the Modified Kalman folder.
## Modifications
- Modified Kalman filter to update only in the y direction
- Since the conveyor only move in 1 direction, lineary. The assignment policy has been modified to not allow matching of previous objects.
- Changed the matching algorithm to euclidean cost instead of iou_cost since iou is useless in low FPS conditions

## Sequence Generator
You can create an image sequence by supplying the function with a step, a small image, a large image. The small image will be slided on the large image with the step size supplied.

## TRT files
The DeepSORT module works seperately from your object detection pipeline. You drop your TRT file in the engine_path. The output bounding boxes, classes and confidences will be supplied to the DeepSORT tracker for assigment

![Demo](https://user-images.githubusercontent.com/20887245/225369296-3f8ed1a7-9eb5-43cd-b4d3-d1e24e4caf57.gif)