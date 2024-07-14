# Vehicle Cut In Detection

This project utilizes the YOLO (You Only Look Once) object detection model and SORT (Simple Online and Realtime Tracking) to detect and track vehicles in a video stream. It calculates the speed of detected vehicles and estimates the time to collision with the vehicle containing the camera.

## Table of Contents

- [Overview](#overview)
- [Code Explanation Video Link](#VideoLink)
- [PPT And Documentation Link](#Documentation)
- [DataSetUsed](#Dataset)
- [Models And Video Link](#ModelsAndTestVideoLink)
- [Features](#features)
- [Customization](#customization)
- [Dependencies](#dependencies)
- [Contributing](#contributing)


## Overview

This project aims to:
- Detect and track vehicles in a video stream.
- Calculate the speed of each detected vehicle.
- Estimate the time to collision with the vehicle containing the camera, based on relative speeds.


## VideoLink

-  https://drive.google.com/file/d/1hsKE7QM_RN9RVO8ZoNgjFyknU_gnYe52/view?usp=sharing


##Documentation

- https://drive.google.com/drive/folders/1jALwAqojlCw14Y_GQ0i7lpgSrmFEdb9M?usp=sharing
  

## Dataset

- IDD temporal dataset used for indian road


## ModelsAndTestVideoLink

- https://drive.google.com/drive/folders/14_9QmOtJKXwsCpRpDVUAHp2Bgz_NSzP0?usp=sharing


## Features

- **Real-Time Vehicle Detection and Tracking:** Uses YOLOv8 for object detection and SORT for tracking.
- **Speed Calculation:** Computes the speed of each detected vehicle.
- **Collision Estimation:** Calculates the time to collision based on the relative speeds of the vehicles.


## Customization

- **Change the video source:** Update the `vdo` variable to point to your desired video file.
- **Adjust your car's speed:** Modify the `v1` variable to match your vehicle's speed form the sensors used or API calls for the camera fit car.
- **Calibration:** Adjust the `real_distance_meters` calculation factor to match the scale of your video(depends on the camera used).

## Dependencies

- OpenCV
- YOLOv8
- cvzone
- numpy
- sort
- math

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push to your branch.
5. Create a pull request.

