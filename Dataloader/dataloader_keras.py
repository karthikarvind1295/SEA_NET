
"""
The dataloader is used to extract images and labels for a simulation dataset in a ROBOT.
"""



import os
import tensorflow as tf
import glob
import cv2
import json
import numpy as np

def extract_labels(label_path):
    """
    The function extracts pick locations from the .json file
    : param label_path: location of the file containing the pick locations for each frames
    """
    
    with open(label_path,'r') as file:
        data = json.load(file)

    label = data['shapes']
    points = []
    for i in range(len(label)):
        points.append(np.array(label[i]['points']))
    #print(points)

    return points

def indices_extraction(indice_location, name):
    """
    The function extracts the label of each sequence
    :param indice_location: location to the file containing the indices where the change in sequence takes place.
    """

    json_file = os.path.join(indice_location, name,'frame.json')
    with open(json_file,'r') as file:
        data = json.load(file)

    return np.array(data[0]['indices'])

def Datasets(image_path, demo_names):

    all_img_path = []
    all_locations = []
    all_labels = []
    for name in demo_names:
        indices = indices_extraction(image_path, name)
        path = os.path.join(image_path, name)
        for img_path in glob.glob(path + "/*.png"):
            img_num = img_path.split("_")[-1]
            frame_number = int(img_num.split(".")[0])
            label_path = os.path.join(path, 'label', 'image_%d.json'%frame_number)

            location = extract_labels(label_path)

            if frame_number <= indices[0]:
                label = 0
            elif frame_number > indices[0] and frame_number <= indices[1]:
                label = 1
            elif frame_number > indices[1] and frame_number <= indices[2]:
                label = 2
            else:
                label = 3
            
            all_img_path.append(img_path)
            all_locations.append(location)
            all_labels.append(np.round(label))
    
    return all_img_path, all_locations, all_labels
    

class CustomDataset_Keras(tf.keras.utils.Sequence):

    def __init__(self, demo_path, demo_names):
        """
        The dataloader accepts the demonstrations location and name of each demonstrations inside it
        :param demo_path : Location of the demonstration files
        :param demo_names : The list of file names inside image_path
        """
        self.image_path = demo_path
        self.demo_names = demo_names


        self.x, self.y, self.z = Datasets(self.image_path, self.demo_names)
         
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):



        img_path = self.x[idx]
        goal_location = self.y[idx]
        label = self.z[idx]

        return np.array(cv2.imread(img_path)), np.array(goal_location), np.array(label)
