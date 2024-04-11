import time
from diffusers import AudioLDM2Pipeline, DDPMScheduler
import torch
import scipy
import pydub
import numpy as np

repo_id = "/zaloai_submission/main"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda:3")
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
steps = 70
audio_length_in_s = 10.0
guidance_scale = 7
seed = 789
generator = torch.manual_seed(seed)


def export_mp3(path, audio, sr=16000, channels=1):
    y = np.int16(audio * 2 ** 15)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(path, format="mp3", bitrate="24k")


prompts = [
    'The recording features a widely spread electric piano melody, followed by synth pad chords. It sounds emotional and passionate.'
]

start = time.time()
audio = pipe(prompt=prompts, 
            num_inference_steps=steps, 
            audio_length_in_s=audio_length_in_s,
            guidance_scale=guidance_scale,
            generator=generator,
            ).audios[0]

export_mp3('test_main_3.mp3', audio)
print(f'Duration: {time.time() - start}')


scipy.io.wavfile.write('test_main_3.wav', rate=16000, data=audio)
