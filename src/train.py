import os
import torch

from tqdm import tqdm

from torch.optim import AdamW

from transformers import (
    get_linear_schedule_with_warmup
)

from torch.cuda.amp import (
    autocast,
    GradScaler
)

from dataset import create_dataloaders
from model import ImageCaptionModel


CONFIG = {

    "caption_file":
        "data/captions.txt",

    "image_dir":
        "data/images",

    "batch_size":
        4,

    "epochs":
        10,

    "lr":
        5e-5,

    "max_length":
        64,

    "num_workers":
        4,

    "patience":
        3,

    "checkpoint_dir":
        "checkpoints",

    "resume":
        False
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
            self.processor,
            self.tokenizer
        ) = (
            self.caption_model
            .get_components()
        )

        self.model.to(DEVICE)

        (
            self.train_loader,
            self.val_loader,
            self.test_loader
        ) = create_dataloaders(
            CONFIG["caption_file"],
            CONFIG["image_dir"],
            self.processor,
            self.tokenizer,
            CONFIG["batch_size"],
            CONFIG["max_length"],
            CONFIG["num_workers"]
        )

        self.optimizer = AdamW(
            self.model.parameters(),
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

        self.scaler = GradScaler()

        self.best_loss = float("inf")

        self.start_epoch = 0

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

            labels = (
                batch["labels"]
                .to(DEVICE)
            )

            self.optimizer.zero_grad()

            with autocast():

                outputs = self.model(
                    pixel_values=
                    pixel_values,

                    labels=
                    labels
                )

                loss = outputs.loss

            self.scaler.scale(
                loss
            ).backward()

            self.scaler.unscale_(
                self.optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0
            )

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

                labels = (
                    batch["labels"]
                    .to(DEVICE)
                )

                with autocast():

                    outputs = self.model(
                        pixel_values=
                        pixel_values,

                        labels=
                        labels
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
            self.scheduler,
            epoch,
            val_loss
        )

    def fit(self):

        patience_counter = 0

        for epoch in range(
            self.start_epoch,
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