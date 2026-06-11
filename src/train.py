import pandas as pd 
import torch 

from torch.utils.data import DataLoader

from dataset import FlickrDataset
from model import load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 4
batch_size = 16
lr = 3e-4 


