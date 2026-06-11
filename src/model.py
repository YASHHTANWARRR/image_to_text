from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer)
import torch
import PIL

model_name = "nlpconnect/vit-gpt2-image-cpationing"


def load_model():
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    return model,feature_extractor,tokenizer