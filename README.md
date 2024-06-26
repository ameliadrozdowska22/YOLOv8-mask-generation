# YOLOv8 Mask generation
Program that integrates the YOLOv8 computer vision model to identify and generate masks for furniture items in an input image

### Program's functionalities:

**1. Filtering specific furniture from the image**
 - detection of all objects from the image
 - providing masks of only specific furniture (chair, couch, bed and dining table)

**2. Mask Generation**
 - generation of masks for all the filtered furniture objects.
    
**3. Saving to the local file**
 - saving the masks into a local folder as PNG files with a name of the detected object.

## Getting started
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

## Approach explanation
Each task of the program is managed by a separate function.\
All functions are then called by the `main()` function to ensure integrated flow\
Tasks of the program include:
 - loading the model, managed by the `load_model()` function
 - obtaining the image from user input, managed by the `image_user_input()` function
 - obtaining the results of model's predictions, managed by the `get_results()` function
 - filtering masks of certain pieces of furniture, managed by the `filter_masks()` function
 - creating images of the masks, managed by the `generate_mask_images()` function
 - saving the images as PNG files on the desired folder with the names corresponding to the masked object, managed by the `save_masks()` function

## Visualization
**1. Uploaded picture:**

![exemplary_image](https://github.com/ameliadrozdowska22/YOLOv8-mask-generation/assets/95606503/3e247ebc-bf37-41cc-afea-8a54c0ded555)

**2. Predictions generated by the YOLOv8 model:**

![boxes](https://github.com/ameliadrozdowska22/YOLOv8-mask-generation/assets/95606503/7e27dfb7-0fef-44ea-9764-f4f4f69ebe18)

**3. Onle of the masks generated by the program:**\
The mask correcponds to the detected dining table

![mask6_dining table](https://github.com/ameliadrozdowska22/YOLOv8-mask-generation/assets/95606503/ef543918-eae7-4e1a-a8a9-34d099780f58)

**4. All the filtered masks generated from the examplary image and their names:**

<img width="368" alt="Generated_masks_examplary" src="https://github.com/ameliadrozdowska22/YOLOv8-mask-generation/assets/95606503/4b2519d8-bbc3-42db-9951-666da011750c">

## Usage examples
This program generates masks for certain furniture in a room.\
Masks of detected objects are used for object segmantation.\
Generated masks can be used to:
1. isolate the piece of furniture from the background.
2. locate the piece of furniture in the room
3. identify the number of certain furniture in the image of the room for example the number of chairs
