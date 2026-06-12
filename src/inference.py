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

        self.load_checkpoint(
            checkpoint_path
        )

        self.model.eval()

    def load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            f"Loaded checkpoint:"
            f" {checkpoint_path}"
        )

    def predict(
        self,
        image_path,
        max_length=32,
        num_beams=4
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        pixel_values = (
            self.processor(
                images=image,
                return_tensors="pt"
            )
            .pixel_values
            .to(DEVICE)
        )

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            )

        caption = (
            self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
        )

        return caption

    def predict_topk(
        self,
        image_path,
        max_length=32,
        top_k=50
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        pixel_values = (
            self.processor(
                images=image,
                return_tensors="pt"
            )
            .pixel_values
            .to(DEVICE)
        )

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    do_sample=True,
                    top_k=top_k
                )
            )

        caption = (
            self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
        )

        return caption

    def predict_topp(
        self,
        image_path,
        max_length=32,
        top_p=0.9
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        pixel_values = (
            self.processor(
                images=image,
                return_tensors="pt"
            )
            .pixel_values
            .to(DEVICE)
        )

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    do_sample=True,
                    top_p=top_p
                )
            )

        caption = (
            self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
        )

        return caption

    def batch_predict(
        self,
        image_paths
    ):

        predictions = []

        for image_path in image_paths:

            caption = self.predict(
                image_path
            )

            predictions.append(
                {
                    "image":
                    image_path,

                    "caption":
                    caption
                }
            )

        return predictions


def main():

    generator = CaptionGenerator(
        checkpoint_path=
        "checkpoints/best_model.pt"
    )

    image_path = input(
        "Enter image path: "
    )

    caption = (
        generator.predict(
            image_path
        )
    )

    print(
        "\nGenerated Caption:"
    )

    print(caption)


if __name__ == "__main__":

    main()