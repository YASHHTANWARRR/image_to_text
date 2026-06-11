from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer)
import torch
import PIL

model_name = "nlpconnect/vit-gpt2-image-cpationing"

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer.pad_token = tokenizer.eos_token

return 
