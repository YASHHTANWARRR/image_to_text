import os
import torch

from tqdm import tqdm

from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from transformers import (
    get_linear_schedule_with_warmup
)

from dataset import create_dataloaders
from model import ImageCaptionModel


CONFIG = {

    "caption_file":"/home/hornet/dataset_folders/archive/captions.txt",
    "image_dir":"/home/hornet/dataset_folders/archive/Images/flickr30k_images",
    "batch_size": 1,
    "epochs": 5,
    "lr": 1e-5,
    "max_length": 32,
    "num_workers": 0,
    "patience": 3,
    "checkpoint_dir":
    "checkpoints"
}


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class Trainer:

    def __init__(self):

        os.makedirs(
            CONFIG["checkpoint_dir"],
            exist_ok=True
        )

        self.caption_model = (
            ImageCaptionModel()
        )

        (
            self.model,
            self.processor
        ) = (
            self.caption_model
            .get_components()
        )

        self.model.to(DEVICE)

        self.caption_model.freeze_all_except_lm_head()
        
        print("Only LM Head Trainable")

        self.model.gradient_checkpointing_enable()

        trainable = (
            self.caption_model
            .count_trainable_parameters()
        )

        total = sum(
            p.numel()
            for p in self.model.parameters()
        )

        print(
            f"Trainable Parameters: "
            f"{trainable:,}"
        )

        print(
            f"Total Parameters: "
            f"{total:,}"
        )
    

        (
            self.train_loader,
            self.val_loader,
            self.test_loader
        ) = create_dataloaders(
            CONFIG["caption_file"],
            CONFIG["image_dir"],
            self.processor,
            batch_size=CONFIG["batch_size"],
            max_length=CONFIG["max_length"],
            num_workers=CONFIG["num_workers"]
        )
        
        self.optimizer = AdamW(
            filter(
                lambda p: p.requires_grad,
                self.model.parameters()
            ),
            lr=CONFIG["lr"]
        )

        total_steps = (
            len(self.train_loader)
            * CONFIG["epochs"]
        )

        self.scheduler = (
            get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
        )

        self.scaler = GradScaler("cuda")

        self.best_loss = float("inf")

    def train_epoch(self):

        self.model.train()

        running_loss = 0

        progress = tqdm(
            self.train_loader,
            desc="Training"
        )

        for batch in progress:

            pixel_values = (
                batch["pixel_values"]
                .to(DEVICE)
            )

            input_ids = (
                batch["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(DEVICE)
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with autocast("cuda"):

                outputs = self.model(

                    pixel_values=
                    pixel_values,

                    input_ids=
                    input_ids,

                    attention_mask=
                    attention_mask,

                    labels=
                    input_ids
                )

                loss = outputs.loss

            self.scaler.scale(
                loss
            ).backward()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            self.scheduler.step()

            running_loss += (
                loss.item()
            )

            progress.set_postfix(
                loss=loss.item()
            )

        return (
            running_loss /
            len(self.train_loader)
        )

    def validate_epoch(self):

        self.model.eval()

        running_loss = 0

        with torch.no_grad():

            progress = tqdm(
                self.val_loader,
                desc="Validation"
            )

            for batch in progress:

                pixel_values = (
                    batch["pixel_values"]
                    .to(DEVICE)
                )

                input_ids = (
                    batch["input_ids"]
                    .to(DEVICE)
                )

                attention_mask = (
                    batch["attention_mask"]
                    .to(DEVICE)
                )

                with autocast("cuda"):

                    outputs = self.model(

                        pixel_values=
                        pixel_values,

                        input_ids=
                        input_ids,

                        attention_mask=
                        attention_mask,

                        labels=
                        input_ids
                    )

                    loss = (
                        outputs.loss
                    )

                running_loss += (
                    loss.item()
                )

        return (
            running_loss /
            len(self.val_loader)
        )

    def save_best_model(
        self,
        epoch,
        val_loss
    ):

        checkpoint_path = (
            f"{CONFIG['checkpoint_dir']}"
            f"/best_model.pt"
        )

        self.caption_model.save_checkpoint(
            checkpoint_path,
            self.optimizer,
            epoch,
            val_loss
        )

    def fit(self):

        patience_counter = 0

        for epoch in range(
            CONFIG["epochs"]
        ):

            print(
                f"\nEpoch "
                f"{epoch+1}/"
                f"{CONFIG['epochs']}"
            )

            train_loss = (
                self.train_epoch()
            )

            val_loss = (
                self.validate_epoch()
            )

            print(
                f"Train Loss:"
                f" {train_loss:.4f}"
            )

            print(
                f"Val Loss:"
                f" {val_loss:.4f}"
            )

            if (
                val_loss
                <
                self.best_loss
            ):

                self.best_loss = (
                    val_loss
                )

                patience_counter = 0

                self.save_best_model(
                    epoch,
                    val_loss
                )

                print(
                    "Best model saved."
                )

            else:

                patience_counter += 1

                print(
                    f"Patience:"
                    f" {patience_counter}"
                )

                if (
                    patience_counter
                    >=
                    CONFIG["patience"]
                ):

                    print(
                        "Early stopping."
                    )

                    break


def main():

    trainer = Trainer()

    trainer.fit()


if __name__ == "__main__":

    main()