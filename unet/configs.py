"""
 Implementation based on:
 - https://youtu.be/IHq1t7NxS8k?si=d9dofGF9n96192R8
 - https://youtu.be/HS3Q_90hnDg?si=6BFVv_jLfQLhuA5i
 - https://d2l.ai/chapter_convolutional-modern/batch-norm.html
 - https://www.youtube.com/watch?v=oLvmLJkmXuc
"""

import torch.nn as nn
import os

# Model Configuration
IN_CHANNELS: int = 3
OUT_CHANNELS: int = 1

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 15
NUM_WORKERS = 2
# BCEWithLogitsLoss - "This loss combines a Sigmoid layer and the BCELoss in one single class. 
#   This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, 
#   we take advantage of the log-sum-exp trick for numerical stability." @https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
LOSS_FN = nn.BCEWithLogitsLoss()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" -> If I ever want to learn/use CUDA.
DEVICE: str = "cpu"

# Model Loading/Saving
LOAD_MODEL = True
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER: str = "model"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FOLDER)
MODEL_NAME = "model_checkpoint"

# Image Configuration
# It has to be divisible by 16 because we are doing 4 down pool convolutions, which means 2^4 = 16.
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
DATA_FOLDER: str = "data"
DATA_DIR = os.path.join(BASE_DIR, DATA_FOLDER)
ORIGINAL_FOLDER: str = "original"
SKIN_MASKS_FOLDER: str = "masks"
DATASETS = [
            "Dataset1_HGR"
            ,"Dataset2_TDSD"
            # ,"Dataset3_Schmugge"
            ,"Dataset4_Pratheepan"
            ,"Dataset5_VDM"
            # ,"Dataset6_SFA"
            ,"Dataset7_FSD"
            # "Dataset8_Abdomen" UNSTABLE
            ]