# Lane Detection 
This repository attempts to build a lane detection system using both OpenCV and deep learning techniques with the goal of deploying onto an Autonomous Mobile Robot (AMR) via a Raspberry Pi. Currently, this repo contains the OpenCV methods adapted from [this GitHub solution](https://github.com/CesarTrevisan/Finding-Lane-Lines-on-the-Road/blob/master/CarND-LaneLines-P1-master/P1.ipynb).

## Steps:
1. Grayscale the image
2. Apply Gaussian blur
3. Use Canny edge detection
4. Define vertices for area of interest and apply mask
5. Use Hough Lines Transform to find lines
6. Merge detected lines and final image
7. Create pipeline

## Requirements
- OpenCV
- Matplotlib
- Numpy

## Future Work
1. To use deep learning techniques to train a model for lane detection instead of using image processing methods.
2. To attempt on different embedded systems like Jetson Nano for deployment.
