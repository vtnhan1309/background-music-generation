from diffusers import AudioLDM2Pipeline, DDPMScheduler
import torch
import pydub
import numpy as np
import os
import json
from tqdm import tqdm


repo_id = "/code/main"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda:0")
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
steps = 70
audio_length_in_s = 10.0
guidance_scale = 7


def export_mp3(path, audio, sr=16000, channels=1):
    y = np.int16(audio * 2 ** 15)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(path, format="mp3", bitrate="24k")


def inference(pipeline, input_file, output_dir, generator):
    with open(input_file, 'r') as file_obj:
        samples = json.load(file_obj)

    for k, v in tqdm(samples.items()):
        output_path = f'{output_dir}/{k}'
        audio = pipe(prompt=v, 
                    num_inference_steps=steps, 
                    audio_length_in_s=audio_length_in_s,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    ).audios[0]
        export_mp3(output_path, audio)


if __name__ == '__main__':
    seeds = [789, 987]
    generators = [
        torch.Generator().manual_seed(seed) for seed in seeds
    ]

    input_file = '/private/private.json'
    output_dir_1 = '/results/submission1'
    output_dir_2 = '/results/submission2'
    if not os.path.exists(output_dir_1):
        os.makedirs(output_dir_1)
    
    if not os.path.exists(output_dir_2):
        os.makedirs(output_dir_2)
    
    print(f'Inference for {output_dir_1} ...')
    inference(pipe, input_file, output_dir_1, generators[0])
    print(f'Inference for {output_dir_2} ...')
    inference(pipe, input_file, output_dir_2, generators[1])
    print('End inference .')
