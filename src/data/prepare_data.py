import pandas as pd 
import random 
import numpy as np 
import argparse

DEFAULT_HEIGHT = 90 
DEFAULT_WIDTH = 400 

CHESS_READER_SPLIT = 0.5 
ONLINE_SPLIT = 0.95
CUSTOM_SPLIT = 0.95 


# Paths to datasets
chess_reader_data_path = "data/chess_reader_dataset"
online_data_path = "data/online_dataset"
custom_data_path = "data/custom_dataset"

# Function to load images and labels
def load_images_labels(path):
    
    df = pd.read_csv(path + "/labels.txt", delimiter=',')
    
    return [
        (path + "/" + name , label) 
        for name, label in zip(
            list(df['id']), list(df['prediction'])
        ) 
    ]
    
     
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Prepare data for training") 
    parser.add_argument("--size_custom", type=int, default=-1, help="Size of custom dataset")
    args = parser.parse_args() 
    
    
    chess_reader_dataset = load_images_labels(chess_reader_data_path) 
    online_dataset = load_images_labels(online_data_path)
    custom_dataset = load_images_labels(custom_data_path)
    
    if (args.size_custom != -1): 
        custom_dataset = custom_dataset[:args.size_custom] 

        
    print(f"Chess reader dataset: {len(chess_reader_dataset)}")
    print(f"Online dataset: {len(online_dataset)}")
    print(f"Custom dataset: {len(custom_dataset)}")
    
    random.shuffle(chess_reader_dataset) 
    random.shuffle(online_dataset)
    random.shuffle(custom_dataset)
    
    # TEST 
    split = int(len(chess_reader_dataset) * CHESS_READER_SPLIT) 
    np.save("data/testset.npy", chess_reader_dataset[:split]) 
    print(f"Testset: {len(chess_reader_dataset[:split])}") 
    # VALIDATION 
    valdataset = chess_reader_dataset[split:] 
    split = int(len(online_dataset) * ONLINE_SPLIT)
    valdataset += online_dataset[split:] 
    split = int(len(custom_dataset) * CUSTOM_SPLIT)
    valdataset += custom_dataset[split:]
    random.shuffle(valdataset)
    print(f"Valset: {len(valdataset)}") 
    np.save("data/valset.npy", valdataset) 
    
    # TRAIN 
    train_dataset = custom_dataset[:split] 
    split = int(len(online_dataset) * ONLINE_SPLIT) 
    train_dataset += online_dataset[:split] 
    
    random.shuffle(train_dataset) 
    np.save("data/trainset.npy", train_dataset) 
    
    print(f"Trainset: {len(train_dataset)}")
    
    
    
