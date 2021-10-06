# The Dataloader is used to extract the labels from the .json files
# that are created from the module_1 - Label extractions


from utils import extract_labels
import os
import tensorflow as tf
import glob
import cv2
import json
import numpy as np

def Datasets(image_path, label_path):
    """
    The function is used to read the results from the data extraction module.
    :param image_path : Location of the image files
    :param label_path : Location of the location json file

    returns all the image paths, respective pick locations, corresponding pick object label.
    """

    all_img_path = []
    all_labels = []
    all_goal_label = []

    json_file = os.path.join(label_path,'frame.json')
    with open(json_file,'r') as file:
        data = json.load(file)
    #print(data[0]['image_0'])


   
    for img_path in glob.glob(image_path + "/*.png"):
        img_num = img_path.split(".")[0]
        frame_name = img_num.split("/")[-1]
        frame_number = int(frame_name.split("_")[-1])

        goal_location = data[frame_number][frame_name]
        goal_label = data[frame_number]['label']
        
        all_img_path.append(img_path)
        all_labels.append(goal_location)
        all_goal_label.append(goal_label)
    
    return all_img_path, all_labels, all_goal_label
    

class Dataloader_location_label(tf.keras.utils.Sequence):

    def __init__(self, image_path, label_path):
        """
        The dataloader accepts image path and output from module 1 for each demonstration and links with its corresponding label location.
        :param image_path : Location of the image files
        :param label_path : Location of the location json file
        """
        self.image_path = image_path
        self.label_path = label_path
        

        self.x, self.y, self.z = Datasets(self.image_path, self.label_path)
         
    def __len__(self):
        return len(self.x) 

    def __getitem__(self, idx):


        img_path = self.x[idx]
        goal_location = self.y[idx]
        goal_label = self.z[idx]


        return np.array(cv2.imread(img_path)), np.array(goal_location), goal_label

class Dataloader_location_label_batch(tf.keras.utils.Sequence):

    def __init__(self, image_path, label_path, batch_size):
        """
        The dataloader introduces batch size for the above function.
        :param image_path : Location of the image files
        :param label_path : Location of the location json file
        :param batch_size : Batch size of the images to be trained
        """
        self.image_path = image_path
        self.label_path = label_path
        self.batch_size = batch_size

        self.x, self.y, self.z = Datasets(self.image_path, self.label_path)
         
    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):

        images = []
        location_all = []
        label_all = []
        initi = idx * self.batch_size
        for i in range(initi, initi + self.batch_size):
            if i >= len(self.x):
                break
            img_path = self.x[i]
            goal_location = self.y[i]
            goal_label = self.z[i]
            images.append(np.array(cv2.imread(img_path)))
            location_all.append(np.array(goal_location[1]))
            label_all.append(np.array(goal_label))


        return np.array(images), np.array(location_all), np.array(label_all)
