import os
import pandas as pd

from PIL import Image

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.model_selection import (
    train_test_split
)


class FlickrDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        processor,
        max_length=32
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.image_dir = image_dir

        self.processor = processor

        self.max_length = max_length

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        image_name = (
            self.df.iloc[idx]["image"]
        )

        caption = (
            self.df.iloc[idx]["caption"]
        )

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = (
            Image.open(image_path)
            .convert("RGB")
        )

        encoding = (
            self.processor(
                images=image,
                text=caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        )

        return {

            "pixel_values":
            encoding[
                "pixel_values"
            ].squeeze(0),

            "input_ids":
            encoding[
                "input_ids"
            ].squeeze(0),

            "attention_mask":
            encoding[
                "attention_mask"
            ].squeeze(0)
        }


def load_dataframe(
    caption_file
):

    df = pd.read_csv(
        caption_file,
        skipinitialspace=True
    )

    df.columns = [
        "image",
        "caption"
    ]

    df["caption"] = (
        df["caption"]
        .astype(str)
        .str.strip()
    )

    return df


def create_splits(
    df,
    train_ratio=0.8,
    val_ratio=0.1,
    random_state=42
):

    unique_images = (
        df["image"]
        .unique()
    )

    train_imgs, temp_imgs = (
        train_test_split(
            unique_images,
            train_size=train_ratio,
            random_state=random_state
        )
    )

    val_imgs, test_imgs = (
        train_test_split(
            temp_imgs,
            test_size=0.5,
            random_state=random_state
        )
    )

    train_df = (
        df[
            df["image"]
            .isin(train_imgs)
        ]
        .reset_index(drop=True)
    )

    val_df = (
        df[
            df["image"]
            .isin(val_imgs)
        ]
        .reset_index(drop=True)
    )

    test_df = (
        df[
            df["image"]
            .isin(test_imgs)
        ]
        .reset_index(drop=True)
    )

    return (
        train_df,
        val_df,
        test_df
    )


def create_dataloaders(
    caption_file,
    image_dir,
    processor,
    batch_size=4,
    max_length=32,
    num_workers=2
):

    df = load_dataframe(
        caption_file
    )

    (
        train_df,
        val_df,
        test_df
    ) = create_splits(df)

    train_dataset = FlickrDataset(
        train_df,
        image_dir,
        processor,
        max_length
    )

    val_dataset = FlickrDataset(
        val_df,
        image_dir,
        processor,
        max_length
    )

    test_dataset = FlickrDataset(
        test_df,
        image_dir,
        processor,
        max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(
        f"Train Samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation Samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test Samples: "
        f"{len(test_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )