# Vehicle Detection and Tracking
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this project is to build a pipeline for detecting vehicles in images
and videos of roads. We first establish a method to extract features from a
vehicle/non-vehicle image and then build a model for classifying these images.
After this, we apply a sliding window search
with this classification model to road images for detecting vehicles.
To see the performance of our pipeline, vehicle detection and tracking is carried out
on a video in the end.

The final video  `project_output.mp4` obtained by applying our pipeline
for vehicle detection to `project_video.mp4` provided
by [Udacity](https://github.com/udacity/CarND-Vehicle-Detection)
is available in the `videos` folder (one can also watch it [here](https://youtu.be/QgNBR6WHYkY)).

The followings are the contents of this repository (please also refer to
the [repository by Udacity for this project](https://github.com/udacity/CarND-Vehicle-Detection)):

- `Vehicle_Detection.ipynb`: this jupyter notebook contains the python codes
for this project.
-  `Vehicle_Detection.html`: this is the html export of `Vehicle_Detection.ipynb`.
- `writeup.md`: this markdown file summarizes and explains the analysis we carried out
in `Vehicle_Detection.ipynb`.
- `README.md`: this markdown is what you are reading now.
- `test_images`: this folder contains images of car roads which we use for checking
our pipeline for detecting vehicles.
- `output_images`: this folder contains the images that we use in `writeup.md`.
- `videos`: this folder contains `project_output.mp4` which we obtained by
applying our pipeline to `project_video.mp4` provided by
[Udacity](https://github.com/udacity/CarND-Vehicle-Detection).




<!--
In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You can submit your writeup in markdown or use another method and submit a pdf instead.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
 -->
