import torch

from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)


MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


class ImageCaptionModel:

    def __init__(self):

        self.model = (
            VisionEncoderDecoderModel
            .from_pretrained(MODEL_NAME)
        )

        self.processor = (
            ViTImageProcessor
            .from_pretrained(MODEL_NAME)
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(MODEL_NAME)
        )

        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )

        self.model.config.pad_token_id = (
            self.tokenizer.pad_token_id
        )

        self.model.config.decoder_start_token_id = (
            self.tokenizer.bos_token_id
        )

        self.model.config.eos_token_id = (
            self.tokenizer.eos_token_id
        )

        self.model.config.max_length = 32

        self.model.config.num_beams = 4

        self.model.config.early_stopping = True

        self.model.config.no_repeat_ngram_size = 2

        self.model.config.length_penalty = 1.0

    def get_components(self):

        return (
            self.model,
            self.processor,
            self.tokenizer
        )

    def save_checkpoint(
        self,
        checkpoint_path,
        optimizer=None,
        scheduler=None,
        epoch=None,
        best_loss=None
    ):

        checkpoint = {
            "epoch": epoch,
            "best_loss": best_loss,
            "model_state_dict":
                self.model.state_dict()
        }

        if optimizer is not None:
            checkpoint[
                "optimizer_state_dict"
            ] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint[
                "scheduler_state_dict"
            ] = scheduler.state_dict()

        torch.save(
            checkpoint,
            checkpoint_path
        )

    def load_checkpoint(
        self,
        checkpoint_path,
        optimizer=None,
        scheduler=None,
        device="cpu"
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            optimizer is not None and
            "optimizer_state_dict"
            in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        if (
            scheduler is not None and
            "scheduler_state_dict"
            in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        return (
            checkpoint.get("epoch", 0),
            checkpoint.get(
                "best_loss",
                float("inf")
            )
        )

    def generate_caption(
        self,
        image,
        device,
        max_length=32,
        num_beams=4
    ):

        self.model.eval()

        pixel_values = (
            self.processor(
                images=image,
                return_tensors="pt"
            )
            .pixel_values
            .to(device)
        )

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=num_beams
                )
            )

        caption = (
            self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
        )

        return caption

    def count_parameters(self):

        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

    def freeze_encoder(self):

        for param in (
            self.model.encoder
            .parameters()
        ):
            param.requires_grad = False

    def unfreeze_encoder(self):

        for param in (
            self.model.encoder
            .parameters()
        ):
            param.requires_grad = True

    def freeze_decoder(self):

        for param in (
            self.model.decoder
            .parameters()
        ):
            param.requires_grad = False

    def unfreeze_decoder(self):

        for param in (
            self.model.decoder
            .parameters()
        ):
            param.requires_grad = True