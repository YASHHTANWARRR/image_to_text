# Image Caption Generation using ViT-GPT2

## Overview

This project implements an end-to-end Image Caption Generation system using a Vision Transformer (ViT) encoder and GPT-2 decoder. The model learns to generate natural language descriptions for images by combining computer vision and natural language processing techniques.

The system is trained on the Flickr30k dataset and fine-tuned using Hugging Face's VisionEncoderDecoder architecture. Mixed Precision Training (AMP) is used to reduce GPU memory consumption and improve training speed.

---

## Features

* ViT (Vision Transformer) image encoder
* GPT-2 text decoder
* Flickr30k dataset support
* Automatic train/validation/test splitting
* Mixed Precision Training (AMP)
* Beam Search caption generation
* BLEU, METEOR, and ROUGE evaluation
* Checkpoint saving and loading
* Early stopping
* Gradient clipping
* Learning rate scheduling
* GPU acceleration using CUDA

---

## Project Structure

```text
IMAGE_TO_TEXT/
│
├── data/
│   ├── captions.txt
│   └── flickr30k_images/
│
├── checkpoints/
│   └── best_model.pt
│
├── results/
│   ├── metrics.json
│   └── predictions.json
│
├── dataset.py
├── model.py
├── train.py
├── inference.py
├── test.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

### Flickr30k

The Flickr30k dataset contains approximately:

* 31,000 images
* 155,000 captions
* 5 captions per image

Example:

```text
1000092795.jpg, Two young guys with shaggy hair look at their hands while hanging out in the yard .
1000092795.jpg, Two young White males are outside near many bushes .
1000092795.jpg, Two men in green shirts are standing in a yard .
1000092795.jpg, A man in a blue shirt standing in a garden .
1000092795.jpg, Two friends enjoy time spent together .
```

Dataset format:

```text
image,caption
image_name.jpg,caption text
```

---

## Model Architecture

```text
Input Image
      │
      ▼
Vision Transformer (ViT)
      │
      ▼
Image Embeddings
      │
      ▼
GPT-2 Decoder
      │
      ▼
Generated Caption
```

### Encoder

* Vision Transformer (ViT)
* Extracts high-level image features

### Decoder

* GPT-2
* Generates captions autoregressively

### Pretrained Model

```python
nlpconnect/vit-gpt2-image-captioning
```

---

## Training Configuration

```python
CONFIG = {
    "batch_size": 2,
    "epochs": 10,
    "lr": 5e-5,
    "max_length": 32,
    "num_workers": 2,
    "patience": 3
}
```

### Hardware

Training was designed for:

* GPU: NVIDIA RTX 3050 Ti (4GB VRAM)
* RAM: 16GB
* OS: Ubuntu Linux

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision
pip install transformers
pip install evaluate
pip install nltk
pip install pandas
pip install pillow
pip install scikit-learn
pip install tqdm
pip install accelerate
```

Or:

```bash
pip install -r requirements.txt
```

---

## Training

Update dataset paths in `train.py`:

```python
CONFIG = {
    "caption_file": "/path/to/captions.txt",
    "image_dir": "/path/to/flickr30k_images"
}
```

Start training:

```bash
python train.py
```

During training:

* Training loss is displayed
* Validation loss is monitored
* Best model is automatically saved
* Early stopping prevents overfitting

Saved checkpoint:

```text
checkpoints/best_model.pt
```

---

## Inference

Generate captions for a new image:

```bash
python inference.py
```

Input:

```text
data/images/example.jpg
```

Example Output:

```text
Generated Caption:
Two young men standing in a grassy field.
```

---

## Evaluation

Run evaluation on the test set:

```bash
python test.py
```

Metrics computed:

### BLEU

Measures n-gram overlap between generated and reference captions.

### METEOR

Evaluates semantic similarity using stemming and synonym matching.

### ROUGE-L

Measures longest common subsequence overlap.

Example:

```json
{
    "BLEU": 0.42,
    "METEOR": 0.31,
    "ROUGE-L": 0.55
}
```

Results are stored in:

```text
results/metrics.json
results/predictions.json
```

---

## Optimization Techniques

### Mixed Precision Training (AMP)

Reduces memory consumption and improves training speed.

### Beam Search

Improves caption quality during inference.

```python
num_beams = 4
```

### Gradient Clipping

Prevents exploding gradients.

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0
)
```

### Early Stopping

Stops training when validation loss no longer improves.

---

## Future Improvements

* BLIP fine-tuning
* Attention visualization
* CIDEr metric implementation
* Reinforcement Learning based caption optimization
* Transformer-based decoder improvements
* Multilingual caption generation
* Web deployment using FastAPI
* ONNX model optimization

---

## Results

The model successfully generates natural language descriptions from images by leveraging pretrained vision-language representations.

Sample prediction:

**Image:** Two people standing in a garden.

**Generated Caption:**

```text
Two young men standing together in a grassy yard.
```

---

## References

* Vision Transformer (ViT)
* GPT-2
* Hugging Face Transformers
* Flickr30k Dataset
* VisionEncoderDecoderModel
* PyTorch
* NVIDIA CUDA
