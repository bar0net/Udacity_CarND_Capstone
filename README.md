# Udacity Self-Driving Car Nanodegree Capstone Project

## Introduction
The objective of this project is to implement the core set of functionalities that would allow an autonomous vehicle drive arround a track with obstacles. These include a path planning system, a control system to manage the signals sent to the actuators on the car and a detection and classification system to identify obstacles (in this case traffic light status).

## Implemented by:
#### Jordi Tudela Alcacer
- jordi.tudela.alcacer@gmail.com

## Systems
### Waypoint Updater
The Waypoint updater checks what are the desired speeds and the positions for the next positions the car should reach. If the car is about to reach a traffic light, it checks if it must stop and when the car gets within a reasonable distance from the crossroad, it applies a linear deceleration.

### Twist Control
Implemented a PID control to try to smooth the changes of velocity and direction. Implemented a reset policy for the filter to avoid undesired unstable behaviours of the error's value.

### Traffic Light Detector & Classifier
For the task of detecting and classifying the state of traffic lights, I have implemented a fast-rcnn network with (a reduced) 50 regions and also limiting the output to 4 classes (Green, Yellow, Red, Off). I started with a pretrained model from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) trained on the COCO dataset.

To train the model, I initially chose the [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) because it is an accessible, reasonably extensive dataset but, not having access to a cuda capable computer and having to train it using Google ML-engine, it seemed apparent that this was not the most resource-efficient strategy. In the end, I opted to train the model with a mixture of images both from the real world and also taken from the simulator.



------------

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
  * [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
