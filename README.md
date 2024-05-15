# YOLOv8 Mask generation
Program that integrates the YOLOv8 computer vision model to identify and generate masks for furniture items in an input image

### Program's functionalities:

**1. Filtering certain furniture from the image**
 - detectection all objects from the image
 - providing masks of only certain furniture (chair, couch, bed and dining table)

**2. Mask Generation**
 - generation of masks for all the filtered furniture objects.
    
**3. Saving to the local File**
 - saving the masks into a local folder as PNG files with a name of the detected object.

## Getting strated
To ensure that the program runs on your device first install unltralytics. Run this line of code on your terminal\
`pip install ultralytics`

or install the ultralytics package from GitHub by running this line on your terminal\
`pip install git+https://github.com/ultralytics/ultralytics.git@main`

To clone this repository run this line on your terminal:
`https://github.com/ameliadrozdowska22/YOLOv8-mask-generation.git`

## Imports
`from ultralytics import YOLO
from ultralytics.engine.results import Results, Masks
from PIL import Image
from typing import List, Dict, Any
import os
`

## Uploading the image
After running the program you will be asked to provide the path to the image you wish to use.\
Paste the path in the console and press enter.\
If the path to the file is faulty or does not exist you will be informed and asked to provide a correct path.\

## Saving the masks
After the image has been processed by the program, you will be asked to provide the path to the folder, in which you wish to save generated masks.\
Paste the path to the foulder in the console and press enter.\
If the path to the foulder is faulty or does not exist you will be informed and asked to provide a correct path.\

## Usage examples
This program can be used to generates masks for certain furniture in a room.

### Visualization of usage example:


