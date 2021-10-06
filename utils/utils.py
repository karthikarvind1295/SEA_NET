
#import torch
#from torch import nn
import os
#import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import json



# class CoordinateUtils(object):
#     @staticmethod
#     def get_image_coordinates(h, w, normalise):
#         x_range = torch.arange(w, dtype=torch.float32)
#         y_range = torch.arange(h, dtype=torch.float32)
#         if normalise:
#             x_range = (x_range / (w - 1)) * 2 - 1
#             y_range = (y_range / (h - 1)) * 2 - 1
#         image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
#         image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        
#         return image_x, image_y

# class convert_rgb_gray(object):
#     def transform(images, crop_size):

#         images = nn.functional.interpolate(images, crop_size)
#         images = [transforms.ToPILImage()(x) for x in images]
#         images = [transforms.Grayscale()(x) for x in images]
#         images = [transforms.ToTensor()(x) for x in images]
#         images = torch.stack(images)

#         return images

# def save_model(savedir, save_variables, filename):
#     model_path = os.path.join(savedir, filename)
#     torch.save(save_variables, model_path)

# def draw_spatial_features(numpy_image, features, image_size=(240, 240)):
#     image_size_x, image_size_y = image_size

#     numpy_image = np.swapaxes(np.swapaxes(numpy_image,0,2), 0, 1)
#     plt.imshow(numpy_image)

#     for sp in features:
#         x, y = sp
#         #attend_x_pix = int((x + 1) * (image_size_x - 1) / 2)
#         #attend_y_pix = int((y + 1) * (image_size_y - 1) / 2)
#         #numpy_image[int(x), int(y)] = [0, 0, 255]
#         plt.scatter(int(x), int(y), marker="x", color="red", s=200)
#         #plt.scatter(attend_x_pix, attend_y_pix, marker="x", color="red", s=200)
#     plt.show()


# def draw_figure(num_images_to_draw, spatial_features_to_draw, images_to_draw, reconstructed_images_to_draw):
#     f, axarr = plt.subplots(num_images_to_draw, 2, figsize=(10, 15), dpi=100)
#     plt.tight_layout()
#     for idx, im in enumerate(reconstructed_images_to_draw[:num_images_to_draw]):
#         # original image
#         og_image = (images_to_draw[:num_images_to_draw][idx] + 1) / 2
#         og_image_res = og_image.numpy()
#         cv2.imshow("original_image",og_image_res)
#         #og_im_res = np.repeat(og_image.numpy().reshape(60, 60, 1), 3, axis=2)
#         # print(spatial_features_to_draw[idx].shape)
#         #draw_spatial_features(og_image_res, spatial_features_to_draw[idx])
#         # axarr[idx, 0].imshow(og_image_res)
#         # reconstructed image
#         # scaled_image = (im + 1) / 2
#         # axarr[idx, 1].imshow(scaled_image.numpy().reshape(60, 60), cmap="gray")

#     #plt.savefig(filename)
#     # plt.show()
#     # plt.close()

def extract_labels(label_path, indice_location, name):
    
    name_split = os.path.splitext(name)[0]
    indices = indices_extraction(indice_location, name_split)
    json_file = os.path.join(label_path,name_split+'.json')
    with open(json_file,'r') as file:
        data = json.load(file)

    label = data['shapes']
    points = []
    for i in range(len(label)):
        points.append(np.array(label[i]['points']))
    #print(points)

    return points, indices

def indices_extraction(indice_location, name):
    json_file = indice_location+'frame.json'
    with open(json_file,'r') as file:
        data = json.load(file)

    return np.array(data[name][0]['indices'])

def location(points, indices, frame_length):
    goal_location = np.zeros((frame_length, 2))
    
    if len(indices) > 4:
        indices = indices[0:4]
    #print(len(indices))
    # for i in range(frame_length):
    #     if i in indices:
    #         k += 1
    #     print(k)
    #     goal_location.append(points[k])
    previous_index = 0
    point_index = 0
    for index in indices:
        goal_location[previous_index:index] = points[point_index]
        previous_index = index
        point_index += 1
    #print(goal_location)

    assert len(goal_location) == frame_length

    return goal_location

def frame_capture(video_path, video_name):
    frames = []
    for name in video_name:
        vidObj = cv2.VideoCapture(os.path.join(video_path,name)) 
        count = 0
        fail = 0
        # checks whether frames were extracted 
        success = 1
        
        while success: 
            # OpenCV Uses BGR Colormap
            success, image = vidObj.read() 
            if success:
                RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #RGBimage_tens = self.transform(RGBimage)
                #RGBimage_tens = RGBimage_tens.permute(1,2,0)

                #if count%10 == 0:            #Sample 1 frame per 10 frames
                frames.append(RGBimage)
                count += 1
            else:
                fail += 1
        vidObj.release()
        #frames = torch.stack(frames)
    return frames

def preprocess(img):
  """Pre-process input (subtract mean, divide by std)."""
  color_mean = 0.18877631
  #depth_mean = 0.00509261
  color_std = 0.07276466
  #depth_std = 0.00903967
  img[:, :, :3] = (img[:, :, :3] / 255 - color_mean) / color_std
  #img[:, :, 3:] = (img[:, :, 3:] - depth_mean) / depth_std
  
  return img

def preprocess_batch(img):
  """Pre-process input (subtract mean, divide by std)."""
  color_mean = 0.18877631
  #depth_mean = 0.00509261
  color_std = 0.07276466
  #depth_std = 0.00903967
  img[:, :, :, :3] = (img[:, :, :, :3] / 255 - color_mean) / color_std
  #img[:, :, 3:] = (img[:, :, 3:] - depth_mean) / depth_std
  
  return img


