
import os
import tensorflow as tf
import glob
import cv2
import json
import numpy as np

def numerical_sort(img_path):
    img_num = img_path.split("_")[-1]
    frame_number = int(img_num.split(".")[0])
    return frame_number


def Datasets(image_path, demo_names):
    """
    This function is used to extract the image path, locations of the object and the labels of the image for the
    location and classification of the images
    :param image_path : Location of the image
    :demo_names : Files inside the image_path

    returns : All the image_path, locations of each images, labels of each images
    """

    all_img_path = []
    all_labels = []
    all_goal_label = []
 
    #Iterating through all the demonstrations in the folder
    for name in demo_names:

        #Reading the frame and indices json files. Indices has the segmented frame numbers.
        label_path = os.path.join(image_path, name)

        with open(os.path.join(label_path,'frame.json'),'r') as file:
            data = json.load(file)
        with open(os.path.join(label_path,'indices.json'), 'r') as f:
            indices = json.load(f)

        #Allocating labels based on the segmented sequences
        num_goal_labels = range(0, len(indices))

        #Iterating through all the images in the folder
        for img_path in sorted(glob.glob(label_path + "/*.png"), key= numerical_sort):

            #Extracting the frame name and number from the image name
            img_num = img_path.split(".")[0]
            frame_name = img_num.split("/")[-1]
            frame_number = int(frame_name.split("_")[-1])

            #Extracting goal_images
            goal_location = data[frame_number][frame_name]

            #Allocating labels based on the frame_number
            if frame_number <= indices[0]:
                goal_label = num_goal_labels[0]
            elif frame_number > indices[0] and frame_number <= indices[1]:
                goal_label = num_goal_labels[1]
            elif frame_number > indices[1] and frame_number <= indices[2]:
                goal_label = num_goal_labels[2]
            else:
                goal_label = num_goal_labels[3]
            
            #Returning all the files
            all_img_path.append(img_path)
            all_labels.append(goal_location)
            all_goal_label.append(goal_label)
    
    return all_img_path, all_labels, all_goal_label
    

class Dataloader_location_label_sequence_test(tf.keras.utils.Sequence):

    def __init__(self, image_path, demo_names):
        """
        The dataloader accepts frames of each demonstration and links with its corresponding label location.
        :param image_path : Location of the image files
        :param demo_names : The list of file names inside image_path
        """
        self.image_path = image_path
        #self.label_path = label_path
        self.demo_names = demo_names

        self.x, self.y, self.z = Datasets(self.image_path, self.demo_names)
         
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        img_path = self.x[idx]
        goal_location = self.y[idx]
        goal_label = self.z[idx]

        return np.array(cv2.imread(img_path)), np.array(goal_location), int(goal_label)
