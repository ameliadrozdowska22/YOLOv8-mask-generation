from ultralytics import YOLO
from PIL import Image
from typing import List


def image_user_input():
    """
    This function takes the path to the image from user's input and checks if it is valid.  
    param: none
    return: string (part to the file) or an errora
    """
    while True:
        file_path = input("Plese privide the path to your image: ")

        # Check if the file exists
        try:
            with open(file_path, 'rb'):
                pass
        except FileNotFoundError:
            print("File not found. Please enter a valid file path.")
        else:
            return file_path
        
        return
        
def load_model(): 
    model = YOLO("yolov8m-seg.pt")

    return model
        
def get_results(model):
    file_path = image_user_input()
    results = model.predict(file_path)
    result = results[0]

    return result

def filter_masks(result):

    masks = result.masks
    filtered_furniture_masks_dict = {}
    clss = result.boxes.cls.cpu().tolist()
    class_name_dict={56: 'chair', 57: 'couch', 59: 'bed', 60: 'dining table'}
    for i, mask in enumerate(masks):
        class_index = clss[i]
        class_name = class_name_dict.get(class_index) 
        if class_name in class_name_dict.values():
            key = class_name
            counter = 2
            while key in filtered_furniture_masks_dict:  # Check if key already exists
                key = f"{class_name}_{counter}"  # Append counter to class name
                counter += 1
            filtered_furniture_masks_dict[key] = mask

    return filtered_furniture_masks_dict


def generate_mask_images(filtered_furniture_masks):
    mask_img_dict = {}
    for class_name, mask in filtered_furniture_masks.items():
        mask_data = mask.data[0].numpy()
        mask_img = Image.fromarray(mask_data, "I")
        mask_img_dict[class_name] = mask_img

    return mask_img_dict

   
def save_masks(mask_img_dict):
    to_save_path = input("Please provide the path to the folder where you would like to save the masks: ")
    counter = 0
    for class_name, mask_img in mask_img_dict.items():
        counter += 1
        file_name = f'mask{counter}_{class_name}'
        mask_img.save(f'{to_save_path}/{file_name}.png')

    return



def main():
    model = load_model()
    result =  get_results(model)
    filtered_furniture_masks = filter_masks(result)
    mask_img_dict = generate_mask_images(filtered_furniture_masks)
    save_masks(mask_img_dict) 
    # result.show()

    return

if __name__ == "__main__":
    main()



#TO DO:
#code comments and doc strings
#github repository
#documentation 