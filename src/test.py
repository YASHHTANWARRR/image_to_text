import json
import torch

from tqdm import tqdm

from model import ImageCaptionModel
from dataset import create_dataloaders


CONFIG = {

    "caption_file":
    "/home/hornet/dataset_folders/archive/captions.txt",

    "image_dir":
    "/home/hornet/dataset_folders/archive/Images/flickr30k_images",

    "batch_size": 4,

    "max_length": 32,

    "num_workers": 2
}


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class Evaluator:

    def __init__(self):

        caption_model = (
            ImageCaptionModel()
        )

        (
            self.model,
            self.processor
        ) = (
            caption_model
            .get_components()
        )

        checkpoint = torch.load(
            "checkpoints/best_model.pt",
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(
            DEVICE
        )

        self.model.eval()

        (
            _,
            _,
            self.test_loader
        ) = create_dataloaders(
            CONFIG["caption_file"],
            CONFIG["image_dir"],
            self.processor,
            batch_size=
            CONFIG["batch_size"],
            max_length=
            CONFIG["max_length"],
            num_workers=
            CONFIG["num_workers"]
        )

    def evaluate(self):

        predictions = []

        with torch.no_grad():

            progress = tqdm(
                self.test_loader,
                desc="Testing"
            )

            for batch in progress:

                pixel_values = (
                    batch["pixel_values"]
                    .to(DEVICE)
                )

                generated_ids = (
                    self.model.generate(
                        pixel_values=
                        pixel_values,

                        max_length=30,

                        num_beams=4
                    )
                )

                generated_text = (
                    self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )
                )

                predictions.extend(
                    generated_text
                )

        return predictions

    def save_predictions(
        self,
        predictions
    ):

        import os

        os.makedirs(
            "results",
            exist_ok=True
        )

        with open(
            "results/predictions.json",
            "w"
        ) as f:

            json.dump(
                predictions,
                f,
                indent=4
            )


def main():

    evaluator = (
        Evaluator()
    )

    predictions = (
        evaluator.evaluate()
    )

    evaluator.save_predictions(
        predictions
    )

    print(
        "\nSaved predictions to:"
    )

    print(
        "results/predictions.json"
    )


if __name__ == "__main__":

    main()