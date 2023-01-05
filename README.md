# Airlab Optitrack+ROS Setup

This package contains configuration and utilities to work with the OptiTrack motion capture system. 

## requiremets:

clone and build the following ROS driver from source in your catkin workspace:

```bash
git clone https://github.com/L2S-lab/natnet_ros_cpp
catkin build
```

> Needed to communicate using the NatNet protocol used by the OptiTrack motion capture system. 
