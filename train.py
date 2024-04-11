import time
import argparse
import json
import logging
import math
import os
# from tqdm import tqdm

import datasets
import numpy as np
import wandb
import torch
import torchaudio
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffusers
import transformers
from transformers import SchedulerType, get_scheduler
from diffusers import AudioLDM2Pipeline
from model import AudioLDMWrapper
from dataset import Text2AudioDataset

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a diffusion model for text to audio generation task.")
    parser.add_argument(
        "--train_file", type=str, default="data/train_audiocaps.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default="data/valid_audiocaps.json",
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="A csv or a json file containing the test data for generation."
    )
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="How many examples to use for training and validation.",
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=False,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--uncondition", action="store_true", default=False,
        help="10% uncondition for training.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=2,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )
    parser.add_argument(
        "--save_every", type=int, default=5,
        help="Save model after every how many epochs when checkpointing_steps is set to best."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()
    return args


# class Text2AudioDataset(Dataset):
#     def __init__(self, dataset, prefix, text_column, audio_column, num_examples=-1):

#         inputs = list(dataset[text_column])
#         self.inputs = [prefix + inp for inp in inputs]
#         self.audios = list(dataset[audio_column])
#         self.indices = list(range(len(self.inputs)))

#         self.mapper = {}
#         for index, audio, text in zip(self.indices, self.audios, inputs):
#             self.mapper[index] = [audio, text]

#         if num_examples != -1:
#             self.inputs, self.audios = self.inputs[:num_examples], self.audios[:num_examples]
#             self.indices = self.indices[:num_examples]

#     def __len__(self):
#         return len(self.inputs)

#     def get_num_instances(self):
#         return len(self.inputs)

#     def __getitem__(self, index):
#         s1, s2, s3 = self.inputs[index], self.audios[index], self.indices[index]
#         return s1, s2, s3

#     def collate_fn(self, data):
#         dat = pd.DataFrame(data)
#         return [dat[i].tolist() for i in dat]
    

def get_batch_audios(paths, feature_extractor, fe_sr=48000):
    wavs = []
    for path in paths:
        waveform, sr = torchaudio.load(path)
        waveform = waveform[0].numpy()
        wavs.append(waveform)

    input_features = feature_extractor(wavs, return_tensors="pt")
    mels = input_features['input_features']
    return mels


def get_mels(paths):
    mels = []
    for path in paths:
        mel = torch.load(path)
        mels.append(mel)

    return torch.vstack(mels)


def main():
    args = parse_args()
    accelerator_log_kwargs = {}


    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/" + str(int(time.time()))
            
            if not os.path.exists("saved"):
                os.makedirs("saved")
                
            os.makedirs(args.output_dir, exist_ok=True)
            
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(project="Text to Audio Diffusion V5")

    accelerator.wait_for_everyone()

    # Initialize models
    repo_id = 'cvssp/audioldm2-music'
    pipeline = AudioLDM2Pipeline.from_pretrained(repo_id)

    model = AudioLDMWrapper(pipeline)
    device = accelerator.device
    model.to_device(device)


    with accelerator.main_process_first():
        # train_dataset = MelDataset(
        #     file=args.train_file,
        #     shuffle=True,
        #     device=device
        # )
        # eval_dataset = MelDataset(
        #     file=args.validation_file,
        #     shuffle=False,
        #     device=device
        # )
        train_dataset = Text2AudioDataset(
            file=args.train_file,
            shuffle=True,
        )
        eval_dataset = Text2AudioDataset(
            file=args.validation_file,
            shuffle=False
        )
        accelerator.print("Num instances in train: {}, validation: {}".format(train_dataset.get_num_instances(), eval_dataset.get_num_instances()))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size)


    # Optimizer
    optimizer_parameters = model.get_unet().parameters()
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint)
            # path = os.path.basename(args.resume_from_checkpoint)
            accelerator.print(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            
    # Duration of the audio clips in seconds
    duration, best_loss = 10, np.inf

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                text, log_mels = batch
                with torch.no_grad():
                    log_mels = log_mels.to(device)
                    true_latent = model.get_first_stage_encoding(model.encode_first_stage(log_mels))

                loss = model(true_latent, text, validation_mode=False)
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        model.uncondition = False

        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                text, log_mels = batch
                log_mels = log_mels.to(device)
                true_latent = model.get_first_stage_encoding(model.encode_first_stage(log_mels))

                val_loss = model(true_latent, text, validation_mode=True)
                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        model.uncondition = args.uncondition

        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1,
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
            result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)

            wandb.log(result)

            result_string = "Epoch: {}, Loss Train: {}, Val: {}\n".format(epoch, result["train_loss"], result["val_loss"])
            
            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["val_loss"] < best_loss:
                best_loss = result["val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False

        if args.with_tracking:
            accelerator.log(result, step=completed_steps)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(args.output_dir, "best"))
                
            if (epoch + 1) % args.save_every == 0:
                accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))
            
            
if __name__ == "__main__":
    main()
