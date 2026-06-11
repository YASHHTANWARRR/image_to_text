import numpy as np 
import pandas as pd 
import PIL

import torch
from torch.utils.data import dataset

def _init_(self,img_dir,csv_file,tokenizer,feature_extractor,
            maxlength=64):
    self.df = pd.read_csv(csv_file)
    
    self.image_dir = img_dir 
    
    self.tokenizer = tokenizer
    
    self.feature_extractor = feature_extractor
    
    self.max_length = maxlength

def _len_(self):
    return len(self,df)

def _get_(self,idx):
    image_path= (
        f"{self.img_dir}/"
        f"{self.df.iloc[idx]["image"]}"
    )
    caption = self.df.iloc[idx][caption]
    
    image = Image.open(image_path).conver("RGB")
    
    pixel_values= feature_extractor(images=image,
                                    return_tensors="pt").pixel_values.squeeze(0)
    
    labels = self.tokenizer(
        caption,
        padding="max_Length",
        truncation= True,
        max_Length = self.max_length,
        return_tensors="pt"
    ).imput_ids.squeeze(0)
    
    return {
        "labels": labels,
        "pixel_values": pixel_values
    }
    