# Move Over Law Compliance Analysis
### This repository was created as a part of Responder and Other Roadside Worker Safety Research project
---


### Introduction

Beginning in September 2020, FHWA issued an award to Battelle Memorial Institute, the University of Maryland's Center for Advanced Transportation Technology (UMD CATT), and the Volpe National Transportation System Center to conduct a comprehensive, data-driven assessment of existing Move Over Laws in the United States. This effort included large dataset analytics covering multiple states and regions as well as automated incident scene observations using state-of-the-art Object Detection and Tracking (ODT) technologies already developed and deployed by UMD CATT. The ODT solution has been modified to focus on tracking vehicles as they travel by first responders (transportation, Fire/EMS, and or law enforcement) who are stopped on the highway with alert lights engaged.
This repository containes the modified ODT code and an application example.

### TO DO 
- share models 
- share example video

### Installation
1) Download this repository
2) It is highly recommended to create a separate environment. I tested the compatibility with Python3.7 and 3.11. You can create and activate the environment with
```
conda create -n "MoveOver" python=3.11 ipython
conda activate MoveOver
```

2) Install the requirements with
```
pip install -r requirements.txt
```
3) Download the models from `LINK REQUIRED` to the models/ folder


### Quick Start
A quick start code is provided in the `MoveOver/example` folder. To use it you should download the video file from `LINK REQUIRED` to the `videos/` folder.
Then yom may start running notebooks.
- `01-PreparConfigs.ipynb` - helps to create a config file for each video
- `02-CreateDetections.ipynb` - performs object detection
- `03-Detect_Lanes.ipynb` - performs object tracking and lane detection
- `04-Lane_analysis.ipynb` - performs Move Over analysis 
- `05-OutputProcess.ipynb` - post-processes the output file to the final format


### Acknowledgements
This work would not have been possible without the sponsorship of the U.S. Department of Transportation. The views expressed are those of the authors and do not reflect the official policy or position of the US Department of Transportation or the US Government.

The Deep SORT implementation is a variation of [Nicolai Wojke Deep Sort repository](https://github.com/nwojke/deep_sort).
