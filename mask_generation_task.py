from ultralytics import YOLO
from ultralytics.engine.results import Results, Masks
from PIL import Image
from typing import List, Dict, Any
import os


def image_user_input():
    """
    This function gets the path to the image from user's input and checks if it is valid.  
    param: None
    return: string (part to the file) or an error
    """
    # Getting the path to image from user's input
    while True:
        file_path: str = input("Plese privide the path to your image: ")

        # Checking if the file exists
        try:
            with open(file_path, 'rb'):
                pass
        except FileNotFoundError:
            print("File not found. Please enter a valid file path.")
        else:
            return file_path
        
        return
    
        
def load_model() -> YOLO: 
    """
    This function initializes the YOLO object which is our model throughout the program 
    param: None
    return: YOLO model 
    """
    # Innitializing the YOLO object which is our model throughout the program 
    model: YOLO = YOLO("yolov8m-seg.pt")

    return model

        
def get_results(model: YOLO) -> Results:
    """
    This function gets the results predicted by the model from the image
    param: None
    return: Results 
    """
    # Accesing the path to image
    file_path: str = image_user_input()

    # Accesing the list of results predicted by the model from the image
    results: List[Results] = model.predict(file_path)

    # Getting the first result object from the list,
    # because we use only one input image at a time
    result: Results = results[0]

    return result


def filter_masks(result: Results) -> Dict[str, Masks]:
    """
    This function filters masks of certain furniture (chair, couch, bed and dining table) from masks of all detected objects from a picture
    param: Results
    return: Dictionary with the class names of detected furniture as keys and the filtered masks as values
    """

    masks: Masks | None = result.masks
    filtered_furniture_masks_dict : Dict[str, Masks] = {}

    # Creating a list of values corresponding to class names of object detected on the image
    class_values_list: List | Any = result.boxes.cls.tolist()

    # Creating a dictionary with keys corresponding to the keys in original file with class names of furniture (coco-labels-2014_2017.txt)
    class_name_dict: Dict[int, str] ={56: 'chair', 57: 'couch', 59: 'bed', 60: 'dining table'} 

    for i, mask in enumerate(masks):
        
        # Getting the name of the object from the mask 
        class_index = class_values_list[i]  
        class_name: str = class_name_dict.get(class_index) 

        # Checking if the the object from the mask is a certain piece of furniture (chair, couch, bed or dining table)
        if class_name in class_name_dict.values():

            # Adding a number to the key to avoid overwritting, if the key (class name) already exists in the dictionary
            key: str = class_name
            counter: int = 2
            while key in filtered_furniture_masks_dict: 
                key = f"{class_name}_{counter}" 
                counter += 1

            filtered_furniture_masks_dict[key] = mask

    return filtered_furniture_masks_dict


def generate_mask_images(filtered_furniture_masks: Dict[str, Masks]) -> Dict[str, Any]:
    """
    This function creates images of filtered masks
    param: Dictionary with the class names of detected furniture as keys and the filtered masks as values
    return: Dictionary with the class names of detected furniture as keys and the images of masks as values
    """
    mask_img_dict: Dict[str, Any] = {}

    # Creating images of masks
    for class_name, mask in filtered_furniture_masks.items():
        mask_data: Any = mask.data[0].numpy()
        mask_img: Image = Image.fromarray(mask_data, "I")
        mask_img_dict[class_name] = mask_img

    return mask_img_dict

   
def save_masks(mask_img_dict: Dict[str, Any]) -> None:
    """
    This function takes the path to the forder from user's input and checks if it is valid.   
    It saves the images of the masks with corresponding class names of masked object in the folder

    param:  Dictionary with the class names of detected objects as keys and the images of masks as values
    return: None
    """
    # Getting the path to the forder from user's input
    to_save_path: str = input("Please provide the path to the folder where you would like to save the masks: ")

    # Checking if the folder exists
    if os.path.isdir(to_save_path):

        # Creating unique names for files with images
        counter: int = 0
        for class_name, mask_img in mask_img_dict.items():
            counter += 1
            file_name: str= f'mask{counter}_{class_name}'

            # Saving the images as PNG files
            mask_img.save(f'{to_save_path}/{file_name}.png')

    else:
        print("Folder not found.")
        save_masks(mask_img_dict)

    return


def main() -> None:
    """
    This function calls other functions of the program connecting their outputs
    It assures integrataion throughout the program

    param:  Dictionary with the class names of detected objects as keys and the images of masks as values
    return: None
    """
    model: YOLO = load_model() 
    result: Results =  get_results(model)
    filtered_furniture_masks: Dict[str, Masks] = filter_masks(result)
    mask_img_dict: Dict[str, Any] = generate_mask_images(filtered_furniture_masks)
    save_masks(mask_img_dict) 

    return


if __name__ == "__main__":
    main()

