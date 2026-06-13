import torch

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

MODEL_NAME = "Salesforce/blip-image-captioning-base"


class ImageCaptionModel:

    def __init__(self):

        self.processor = (
            BlipProcessor
            .from_pretrained(MODEL_NAME)
        )

        self.model = (
            BlipForConditionalGeneration
            .from_pretrained(MODEL_NAME)
        )

    def get_components(self):

        return (
            self.model,
            self.processor
        )

    def freeze_all_except_lm_head(self):

        for param in self.model.parameters():
            param.requires_grad = False

        for param in (
            self.model
            .text_decoder
            .cls
            .parameters()
        ):
            param.requires_grad = True

        print("Only LM Head Trainable")

    def count_trainable_parameters(self):

        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

    def save_checkpoint(
        self,
        checkpoint_path,
        optimizer=None,
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

        torch.save(
            checkpoint,
            checkpoint_path
        )