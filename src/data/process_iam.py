import os
import pandas as pd
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

raw_data_path = 'data/raw/iam' 
processed_data_path = 'data/iam_dataset'
os.makedirs(processed_data_path, exist_ok=True)

def png_to_mat(path): 
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    
    if img is None:
        return None 
    img = img / 255.0 
    print("max min", img.max(), img.min()) 
    
    return img 

def resize_img(img, target_height, target_width): 
    res = np.ones((target_height, target_width)) 
    res[:img.shape[0], :img.shape[1]] = img 
    return res 
# go through all folders 

data = [] 
img_id = 0 
max_size = 10000 
for root, dirs, files in os.walk(raw_data_path):
    dirs.sort() 
    files.sort() 
    for file in files:
        if not file.endswith('.png'):
            continue 
        source = os.path.join(root, file) 
        print(source) 
        img = png_to_mat(source) 
        
        if (
            not(
            img is None or 
            img.shape[0] > 100 or 
            img.shape[1] > 100
            )
        ):  
            data.append(
                [img_id,img]
            )
        
        img_id += 1 
        if len(data) > max_size: 
            break
    if len(data) > max_size: 
        break 

print("Images loaded") 

world_label = {}
words_file_path = "data/raw/iam/words.txt" 
word_id = 0 

with open(words_file_path, "r") as f:
    last_word_id = data[-1][0] 
    for line in f:
        if word_id > last_word_id:
            break 
        # Skip comments or empty lines
        if line.startswith("#") or line.strip() == "":
            continue
        # Split the line into parts
        parts = line.split()
        world_label[word_id] = parts[-1] 
        word_id += 1

print("Resizing images") 
      
#resize all imgaes. 
max_height = max([img.shape[0] for _,img in data])  
max_width = max([img.shape[1] for _,img in data]) 
labels = []
images = [] 
for i, (img_id, img) in enumerate(data): 
    data[i] = resize_img(img, max_height, max_width)
    labels.append(world_label[img_id]) 
    images.append(data[i]) 
    
print("Start saving data") 
# Save the data 
images = np.array(images) 
labels = np.array(labels) 
np.save(os.path.join(processed_data_path, "iam_images.npy"), images) 
np.save(os.path.join(processed_data_path, "iam_labels.npy"), labels) 

print("Data saved")