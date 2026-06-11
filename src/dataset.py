import os 
import numpy as np 
import pandas as pd 
import PIL

from PIL import Image

import torch
from torch.utils.data import Dataset,Dataloader

from sklearn.model_selection import train_test_split

#dataset class being called in train.py
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
    
    return df

#splitting of dataset 
def create_splits(
    df,
    train_splits=0.8,
    val_splits=0.1,
    random_states=42
):
    unique_images = (df["images"]
                        .unique())
    
    train_imgs, temp_imgs=(
        train_test_split(
            temp_imgs,
            train_size=train_splits,
            val_splits=val_splits,
            random_states=random_states
        )
    )
    
    val_imgs, test_imgs=(
        train_test_split(
            temp_imgs,
            test_size=0.5,
            random_state=random_states
            
        )
    )
    
    train_df=(
        df[df["images"].isin(train_imgs)]
            .reset_index(drop=True)
    )
    
    val_df=(
        df[df["images"].isin(val_imgs)]
        .reset_index(drop=True)
    )
    
    test_df=(
        df[df["images"].isin(test_imgs)]
        .reset_index(drop=True)
    )
    
    return (train_df,val_df,test_df)


#data loaders  
def create_dataLoaders(
    caption_file,
    img_dir,
    processor,
    tokenizer,
    batch_size=6,
    maxlength=64,
    num_workers=4
):
    
    df = load_dataframe(caption_file)
    
    (train_df,
        val_df,
        test_df)=create_splits(df)
    
    train_dataset=FlickrDataset(
        train_df,
        img_dir,
        tokenizer,
        processor,
        maxlength
    )
    
    val_dataset= FlickrDataset(
        val_df,
        img_dir,
        tokenizer,
        processor,
        maxlength
    )
    
    test_dataset = FlickrDataset(
        test_df,
        img_dir,
        tokenizer,
        processor,
        maxlength
    )
    
    train_loader = Dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = Dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = Dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return(train_loader,val_loader,test_loader) 