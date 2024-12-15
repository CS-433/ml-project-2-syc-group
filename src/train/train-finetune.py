import os
import torch.optim as optim
from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from src.utils import CERMetricShortCut, WERMetricShortCut 
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage
from src.augmentations import RandomRotateFillWithMedian 
from src.models.cnn_bilstm import CNNBILSTM
from src.models.trocr import TrOCRWrapper, ModelTrOCR, TrOCRLabelIndexer
from argparse import ArgumentParser 
import yaml 
import numpy as np 
from types import SimpleNamespace 
from src.models.htr_net import HTRNet
from src.models.trocr import CERMetricTrOCR 

# LOAD CONFIGS 
parser = ArgumentParser() 
parser.add_argument("--config", type=str, required=True) 
args = parser.parse_args() 
config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
configs = SimpleNamespace(**config) 

assert configs.model is not None, "Please specify a model architecture in the configs file"


# LOAD THE DATASET. 
train_dataset = np.load("data/trainset.npy") 
train_dataset = [(name, label) for name, label in train_dataset] 
val_dataset = np.load("data/valset.npy") 
val_dataset = [(name, label) for name, label in val_dataset]



# Create our model 
# NOTE to test another model architecture. add a filed "model" to the configs file 
# and add if-else logic here to create the model specified in the configs file.



network = TrOCRWrapper(model_path="microsoft/trocr-small-handwritten", device=configs.device)

optimizer = optim.AdamW(network.parameters(), lr=configs.learning_rate)

# put on cuda device if available
network = network.to(configs.device)


print(f"Len of train dataset: {len(train_dataset)}") 


# Create a data provider for the dataset
train_dataProvider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        #ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        TrOCRLabelIndexer(network.processor), 
        #LabelPadding(max_word_length=configs.max_text_length, padding_value=-100)#TODO 
        ],
    use_cache=True,
)
val_dataProvider = DataProvider(
    dataset = val_dataset,
    skip_validation=train_dataProvider._skip_validation,
    batch_size=train_dataProvider._batch_size,
    data_preprocessors=train_dataProvider._data_preprocessors,
    transformers=train_dataProvider._transformers,
    use_cache=train_dataProvider._use_cache,
)


# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotateFillWithMedian(angle=10), 
]




# create callbacks
# used to track important metrics. 
earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)


# create model object that will handle training and testing of the network
# NOTE : FOR REASONS I DONT KNOW (YET), DATA IS PASSED TO THE MODEL WITH SHAPE 
# (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS) 


model = ModelTrOCR(network, optimizer, loss = None , metrics=[CERMetricTrOCR(network.processor)])
model.fit(
    train_dataProvider, 
    val_dataProvider, 
    epochs=configs.train_epochs, 
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr]
)


# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))