# Kabyle ASR with OpenAI Whisper

![Python](https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00.svg?logo=googlecolab&logoColor=white)
![Nvidia](https://img.shields.io/badge/Hardware-NVIDIA%20A100-76B900.svg?logo=nvidia&logoColor=white)

This project fine-tunes the **OpenAI Whisper** model (`whisper-small`) for **Automatic Speech Recognition (ASR)** specifically targeting the **Kabyle language** (Taqbaylit).

The goal is to provide a reliable pipeline that takes Kabyle audio input and transcribes it into accurate text representations using the modern deep learning stack from Hugging Face.

---

## Dataset

The model is trained on the [`TutlaytAI/kabyle_asr`](https://huggingface.co/datasets/TutlaytAI/kabyle_asr) dataset hosted on Hugging Face. All audio inputs are resampled to **16 kHz** as required by the Whisper feature extractor.

## Project Structure

The project is structured into two main phases, executed via Jupiter Notebook primarily utilizing the power of **NVIDIA A100 GPUs** for accelerated processing:

#### 1. Training (Fine-Tuning)

* **Preprocessing:** Audio resampling and feature extraction using `WhisperProcessor`. Label padding and formatting using a custom Data Collator.
* **Hyperparameters:**
  * Epochs: 2
  * Learning Rate: 2e-5
  * Batch Size: 36 (Train/Eval)
  * Metric: Word Error Rate (WER)
* **Results:** Reached a final Validation **WER of ~35.42%**.
* **Export:** Checkpoints are exported directly to Google Drive for persistent storage.

#### 2. Inference

* Restores the fine-tuned model weights from the saved checkpoint.
* Leverages Hugging Face's `pipeline` API for easy "Audio-to-Text" execution.

---

## Installation & Requirements

If you want to run or reproduce this environment locally, install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: The notebook was originally designed to run in Google Colab, leveraging GPU acceleration and Google Drive for storage. If running locally, you must remove the `google.colab` imports and update the file payload paths accordingly.*

## Usage (Inference)

Here is a quick example of how the fine-tuned model handles inference on a sample `.wav` file:

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

# 1. Load trained model & processor
checkpoint_path = "./path/to/your/checkpoint"
model_name      = "openai/whisper-small"

model     = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
processor = WhisperProcessor.from_pretrained(model_name)
device    = 0 if torch.cuda.is_available() else -1

# 2. Initialize Pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

# 3. Transcribe
audio_file = "A001_D01_4.wav"
transcription = pipe(audio_file)["text"]

print(transcription)
```

## Evaluation Examples

**Actual Transcription:**
> *"Anda iruḥ baba-s ad yeddu"*

**Model Prediction:**
> *"Anta iṛuḥ baba sredi-d-tuh!"*

*(Note: The model demonstrates an understanding of the phonetic flow but requires further epochs/data to achieve perfect spelling accuracy.)*
