import pandas as pd 
import torch 

from torch.utils.data import DataLoader

from dataset import FlickrDataset
from model import load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 4
batch_size = 16
lr = 3e-4 

def train():
    model, feature_extractor, tokenizer = load_model()
    
    model.to(device)
    
    dataset = FlickrDataset(
                img_file='/home/hornet/dataset_folders/archive/Images/flickr30k_images',
                csv_file='/home/hornet/dataset_folders/archive/context.txt',
                tokenizer=tokenizer,
                feature_extractor=feature_extractor
    )
    
    


