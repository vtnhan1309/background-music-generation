# AudioLDM2 for Zalo AI Challenge 2023

This is an solution for background music generation task in [Zalo AI Challenge 2023](https://challenge.zalo.ai/portal/background-music-generation). By using latent diffusion on audio domain, we can generate audio/music from a prompt. We chose a checkpoint in [AudioLDM2](https://github.com/haoheliu/AudioLDM2) and finetuned on the challenge dataset. Training and evaluation script are provided for reproduction. You can try our submited model now by [HuggingFace](https://huggingface.co/vtrungnhan9/audioldm2-music-zac2023) or [Colab notebook](https://colab.research.google.com/drive/1x8Lz4iWWI65ExEhYpQ9ts6Vf0O43Rbl_?usp=sharing)

## Getting started

### Install packages

```
    pip install --upgrade diffusers transformers accelerate
```

### Text to audio

```
from diffusers import AudioLDM2Pipeline
import torch
import scipy

repo_id = "vtrungnhan9/audioldm2-music-zac2023"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "This music is instrumental. The tempo is medium with synthesiser arrangements, digital drums and electronic music. The music is upbeat, pulsating, youthful, buoyant, exciting, punchy, psychedelic and has propulsive beats with a dance groove. This music is Techno Pop/EDM."
neg_prompt = "bad quality"
audio = pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=200, audio_length_in_s=10.0, guidance_scale=10).audios[0]
scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
```

## Training

```
export CUDA_VISIBLE_DEVICES=0
accelerate launch train.py \
--train_file="/code/data/train_main.json" \
--validation_file="/code/data/val_main.json" \
--freeze_text_encoder \
--gradient_accumulation_steps 1 --per_device_train_batch_size=32 --per_device_eval_batch_size=4 \
--learning_rate=3e-5 --num_train_epochs 200 --snr_gamma 5 \
--text_column captions --audio_column location --checkpointing_steps="best" \
--seed 123 \
--save_every 25
```
