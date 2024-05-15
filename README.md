# YOLOv8 Mask generatiiom
Program that integrates the YOLOv8 computer vision model to identify and generate masks for furniture items in an input image

**1. Filtering certain furniture from the image**

The program: 
 - detectects all objects from the image
 - provides masks of only certain furniture (chair, couch, bed and dining table)

**2. Mask Generation**

The program:
 - generates masks for all the filtered furniture objects.
    
**3. Saving to the local File**

The program:
 - saves the masks into a local folder as PNG files with a name of the detected object.

## Imports
`from ultralytics import YOLO
from ultralytics.engine.results import Results, Masks
from PIL import Image
from typing import List, Dict, Any
import os
`

