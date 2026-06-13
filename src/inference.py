import torch

from PIL import Image

from model import ImageCaptionModel


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class CaptionGenerator:

    def __init__(
        self,
        checkpoint_path
    ):

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
            checkpoint_path,
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

    def predict(
        self,
        image_path,
        max_length=30,
        num_beams=4
    ):

        image = (
            Image.open(image_path)
            .convert("RGB")
        )

        inputs = (
            self.processor(
                image,
                return_tensors="pt"
            )
            .to(DEVICE)
        )

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams
                )
            )

        caption = (
            self.processor.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
        )

        return caption


def main():

    generator = (
        CaptionGenerator(
            "checkpoints/best_model.pt"
        )
    )

    while True:

        image_path = input(
            "\nImage Path (q to quit): "
        )

        if image_path.lower() == "q":
            break

        caption = (
            generator.predict(
                image_path
            )
        )

        print(
            f"\nCaption:\n{caption}"
        )


if __name__ == "__main__":

    main()