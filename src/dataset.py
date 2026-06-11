import os 
import numpy as np 
import pandas as pd 
import PIL

import torch
from torch.utils.data import dataset

from sklearn.model_selection import train_test_split

class FlickrDataset(dataset):
    
    def _init_(self,dataframe,img_dir,tokenizer,feature_extractor,
                maxlength=64):
        self.df = dataframe.reset_index(drop=True)
        
        self.image_dir = img_dir 
        
        self.tokenizer = tokenizer
        
        self.feature_extractor = feature_extractor
        
        self.max_length = maxlength

    def _len_(self):
        return len(self,df)

    def _get_(self,idx):
        image_name = self.df.iloc[idx]["image"]
        
        caption = self.df.iloc[idx]["caption"]
        
        image_path = os.path.join(self.img_dir,
                                    image_name)
        
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
        
        labels [labels==self.tokenizer.pad_token_id,
                ]=-100
        
        return {
            "labels": labels,
            "pixel_values": pixel_values
        }

def load_dataframe(caption_file):
    
    df = pd.read_csv(caption_file,
                        skipinitialspace=True)
    
    df.columns=["images",
                "captions"]
    
    df["captions"] = (df["caption"]
                        .astype(str)
                        .str.strip()
                        )
    
    