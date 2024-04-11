import torch
from torch import nn
import torch.nn.functional as F
from diffusers import AudioLDM2Pipeline, DDPMScheduler, AudioLDMPipeline
from diffusers.models.vae import DiagonalGaussianDistribution
from stft import TacotronSTFT


def disable_grad_module(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class AudioLDMWrapper(nn.Module):
    def __init__(self, model:AudioLDM2Pipeline) -> None:
        super().__init__()
        self.model = model
        modules = [
            self.model.vae,
            self.model.text_encoder,
            self.model.text_encoder_2,
            self.model.projection_model,
            self.model.language_model,
        ]
        for module in modules:
            disable_grad_module(module)

        self.model.unet.train()
        self.noise_scheduler = DDPMScheduler.from_config(self.model.scheduler.config)
        self.noise_scheduler = self.model.scheduler
        self.snr_gamma = 5

    
    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr


    def forward(self, latents, prompt, validation_mode=False):
        if type(prompt) is tuple:
            prompt = list(prompt)
        device = self.model.text_encoder.device
        num_train_timesteps = self.model.scheduler.num_train_timesteps
        self.model.scheduler.set_timesteps(num_train_timesteps, device=device)

        with torch.no_grad():
            prompt_embeds, attention_mask, generated_prompt_embeds = self.model.encode_prompt(
                prompt,
                device,
                1,
                False,
            )

        bsz = latents.shape[0]

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")


        model_pred = self.model.unet(
            noisy_latents, timesteps, 
            encoder_hidden_states=generated_prompt_embeds,
            encoder_hidden_states_1=prompt_embeds,
            encoder_attention_mask_1=attention_mask
        ).sample
        
        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
        snr = self.compute_snr(timesteps)
        mse_loss_weights = (
            torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

        return loss
    
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.model.vae.encode(x).latent_dist
    
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.model.vae.scaling_factor * z
    

    def get_device(self):
        return next(self.model.text_encoder.parameters()).device
    
    def get_unet(self):
        return self.model.unet
    
    def get_feature_extractor(self):
        return self.model.feature_extractor
    
    def to_device(self, device):
        self.model.to(device)

    
    def train(self):
        self.model.unet.train()

    
    def eval(self):
        self.model.unet.eval()

    def state_dict(self):
        return self.model.unet.state_dict()
    

def get_stft(name):
    config = default_audioldm_config(name)
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )
    fn_STFT.eval()
    return fn_STFT

def default_audioldm_config(model_name="audioldm-s-full"):    
    basic_config = {
        "wave_file_save_path": "./output",
        "id": {
            "version": "v1",
            "name": "default",
            "root": "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/AudioLDM-python/config/default/latent_diffusion.yaml",
        },
        "preprocessing": {
            "audio": {"sampling_rate": 16000, "max_wav_value": 32768},
            "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
            "mel": {
                "n_mel_channels": 64,
                "mel_fmin": 0,
                "mel_fmax": 8000,
                "freqm": 0,
                "timem": 0,
                "blur": False,
                "mean": -4.63,
                "std": 2.74,
                "target_length": 1024,
            },
        },
        "model": {
            "device": "cuda",
            "target": "audioldm.pipline.LatentDiffusion",
            "params": {
                "base_learning_rate": 5e-06,
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "first_stage_key": "fbank",
                "cond_stage_key": "waveform",
                "latent_t_size": 256,
                "latent_f_size": 16,
                "channels": 8,
                "cond_stage_trainable": True,
                "conditioning_key": "film",
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "audioldm.latent_diffusion.openaimodel.UNetModel",
                    "params": {
                        "image_size": 64,
                        "extra_film_condition_dim": 512,
                        "extra_film_use_concat": True,
                        "in_channels": 8,
                        "out_channels": 8,
                        "model_channels": 128,
                        "attention_resolutions": [8, 4, 2],
                        "num_res_blocks": 2,
                        "channel_mult": [1, 2, 3, 5],
                        "num_head_channels": 32,
                        "use_spatial_transformer": True,
                    },
                },
                "first_stage_config": {
                    "base_learning_rate": 4.5e-05,
                    "target": "audioldm.variational_autoencoder.autoencoder.AutoencoderKL",
                    "params": {
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 8,
                        "time_shuffle": 1,
                        "ddconfig": {
                            "double_z": True,
                            "z_channels": 8,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0.0,
                        },
                    },
                },
                "cond_stage_config": {
                    "target": "audioldm.clap.encoders.CLAPAudioEmbeddingClassifierFreev2",
                    "params": {
                        "key": "waveform",
                        "sampling_rate": 16000,
                        "embed_mode": "audio",
                        "unconditional_prob": 0.1,
                    },
                },
            },
        },
    }
    
    if("-l-" in model_name):
        basic_config["model"]["params"]["unet_config"]["params"]["model_channels"] = 256
        basic_config["model"]["params"]["unet_config"]["params"]["num_head_channels"] = 64
    elif("-m-" in model_name):
        basic_config["model"]["params"]["unet_config"]["params"]["model_channels"] = 192
        basic_config["model"]["params"]["cond_stage_config"]["params"]["amodel"] = "HTSAT-base" # This model use a larger HTAST
        
    return basic_config

class AudioLDMV1Wrapper(nn.Module):
    def __init__(self, model:AudioLDMPipeline) -> None:
        super().__init__()
        self.model = model
        modules = [
            self.model.vae,
            self.model.text_encoder,
        ]
        for module in modules:
            disable_grad_module(module)

        self.model.unet.train()
        self.noise_scheduler = DDPMScheduler.from_config(self.model.scheduler.config)
        self.noise_scheduler = self.model.scheduler
        self.snr_gamma = 5

    
    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr


    def forward(self, latents, prompt, validation_mode=False):
        if type(prompt) is tuple:
            prompt = list(prompt)
        device = self.model.text_encoder.device
        num_train_timesteps = self.model.scheduler.num_train_timesteps
        self.model.scheduler.set_timesteps(num_train_timesteps, device=device)

        with torch.no_grad():
            prompt_embeds = self.model._encode_prompt(
                prompt,
                device,
                1,
                False,
            )

        bsz = latents.shape[0]

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")


        model_pred = self.model.unet(
            noisy_latents, timesteps, 
            encoder_hidden_states=None,
            class_labels=prompt_embeds,
        ).sample
        
        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
        snr = self.compute_snr(timesteps)
        mse_loss_weights = (
            torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

        return loss
    
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.model.vae.encode(x).latent_dist
    
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.model.vae.scaling_factor * z
    

    def get_device(self):
        return next(self.model.text_encoder.parameters()).device
    
    def get_unet(self):
        return self.model.unet
    
    def get_feature_extractor(self):
        return self.model.feature_extractor
    
    def to_device(self, device):
        self.model.to(device)

    
    def train(self):
        self.model.unet.train()

    
    def eval(self):
        self.model.unet.eval()

    def state_dict(self):
        return self.model.unet.state_dict()
