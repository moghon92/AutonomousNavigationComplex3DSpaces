# Navigation in Complex 3D Spaces

This project presents a state-of-the-art approach for using depth sensing cameras for mobile robot navigation and indoor terrain exploration in challenging and physically inaccessible areas. It involves object detection, path planning, and obstacle avoidance, showing promise in the field of depth sensors for navigation and setting the foundation for future development.

## 1.1 Objective

The objective is to enable quadcopter to navigate through unknown environments. It examines a variety of navigation strategies and their practical implementation.

Depth sensing cameras, with their ability to process captured data to extract and locate obstacles in unknown environments, are considered an effective and efficient method to navigate complex 3D spaces.

This study presents the technique of using range-sensing cameras for path planning and navigation in complex 3D spaces. Until now, most mobile robot implementations relied on 2D sensors; depth cameras aren't common among researchers. This research showcases the potential of 3D sensing and depth vision as an efficient solution for environment exploration and navigation. It serves as a basis for further development and innovation in various research areas.


## 1.2 Scope of Work and Limitations

The research focuses on object detection, path planning, and obstacle avoidance for unmanned aerial vehicles, with an emphasis on range finding sensors, primarily the Kinect depth sensor. The research faced limitations, including the vision capabilities of the Kinect depth sensor and its incompatibility with the current hardware setup of FALTER quadrocopter. This incompatibility prevented us from real-life testing of our software on the copter control unit.

## 1.3 report Outline

- **Chapter 2**: Provides the basic background needed to understand subsequent discussions.
- **Chapter 3**: Overviews the current state of art developments related to our topic and summarizes techniques developed to solve path planning and navigation problems.
- **Chapter 4**: Explains the current FALTER hardware and software architecture, emphasizing the data acquired from Kinect and how it's used in our research.
- **Chapter 5**: Discusses five different object detection techniques and concludes which technique would best fit our application.
- **Chapter 6**: Presents the technique used for path planning and obstacle avoidance using depth data from Kinect.
- **Chapter 7**: Explains how the algorithm developed in Chapter 6 was integrated with FALTER's virtual reality on the Matlab Simulink simulation.
- **Chapter 8**: Proposes a technique to further extend the algorithm presented in Chapter 6.
- **Chapter 9**: Summarizes the thesis, provides conclusions based on testing results, and outlines potential directions for future work on FALTER.
