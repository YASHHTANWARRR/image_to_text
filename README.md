# Image Caption Generation using BLIP

## Overview

This project implements an Image Caption Generation system using the BLIP (Bootstrapping Language-Image Pre-training) architecture from Salesforce. The model generates natural language descriptions for input images by combining computer vision and natural language processing techniques.

The project is trained on the Flickr30k dataset and supports:

* Training
* Validation
* Testing
* Inference
* Model checkpointing
* Automatic caption generation

---

## Project Structure

```text
image_to_text/
│
├── checkpoints/
│   └── best_model.pt
│
├── results/
│   └── predictions.json
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── test.py
│
└── README.md
```

---

## Dataset

Dataset Used:

Flickr30k Dataset

Dataset Format:

```text
image,caption
1000092795.jpg,Two young guys with shaggy hair look at their hands while hanging out in the yard.
1000092795.jpg,Two young White males are outside near many bushes.
...
```

Image Directory:

```text
Images/
├── 1000092795.jpg
├── 10002456.jpg
├── ...
```

---

## Model Architecture

Model:

```text
Salesforce/blip-image-captioning-base
```

Components:

* Vision Transformer (ViT) Image Encoder
* Transformer Text Decoder
* BLIP Processor

Training Strategy:

* Vision encoder frozen
* Language modeling head fine-tuned
* Mixed precision training
* Gradient checkpointing enabled

---

## Requirements

Install dependencies:

```bash
pip install torch torchvision
pip install transformers
pip install accelerate
pip install pillow
pip install pandas
pip install scikit-learn
pip install tqdm
```

Verify installation:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Configuration

Edit configuration inside:

```text
src/train.py
```

Example:

```python
CONFIG = {
    "caption_file": "/path/to/captions.txt",
    "image_dir": "/path/to/images",
    "batch_size": 1,
    "epochs": 5,
    "lr": 1e-5,
    "max_length": 32,
    "num_workers": 0,
    "checkpoint_dir": "checkpoints"
}
```

---

## Training

Start training:

```bash
python src/train.py
```

During training:

* Training loss is displayed
* Validation loss is computed
* Best model is automatically saved

Saved checkpoint:

```text
checkpoints/best_model.pt
```

---

## Inference

Generate captions for individual images:

```bash
python src/inference.py
```

Example:

```text
Image Path:
/path/to/image.jpg

Generated Caption:
A man riding a bicycle on a city street.
```

---

## Testing

Run evaluation on the test set:

```bash
python src/test.py
```

Output:

```text
results/predictions.json
```

Example:

```json
[
    "A man riding a bicycle.",
    "A dog running through grass.",
    "Children playing in a park."
]
```

---

## Hardware Used

Training Environment:

* NVIDIA RTX 3050 Ti Laptop GPU
* 4 GB VRAM
* 16 GB RAM
* Ubuntu Linux
* Python 3.10

Optimization Techniques:

* Mixed Precision Training
* Gradient Checkpointing
* Selective Parameter Fine-Tuning
* AdamW Optimizer

---

## Applications

* Assistive Technologies
* Image Search Systems
* Visual Question Answering Pipelines
* Social Media Caption Generation
* Content Management Systems
* Automated Dataset Annotation

---

## Future Improvements

* BLEU Score Evaluation
* ROUGE Score Evaluation
* CIDEr Metric Evaluation
* LoRA Fine-Tuning
* Quantized Training
* Web Interface Deployment
* ONNX Export
* Real-Time Video Captioning

---

## Author

Yash Tanwar

Department of Computer Science and Engineering

Image Caption Generation using BLIP and Flickr30k Dataset.
