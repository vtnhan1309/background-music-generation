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

accelerate launch train.py \
--train_file="/code/data/train_temp.json" \
--validation_file="/code/data/val_temp.json" \
--freeze_text_encoder \
--gradient_accumulation_steps 1 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 \
--learning_rate=3e-5 --num_train_epochs 200 --snr_gamma 5 \
--text_column captions --audio_column location --checkpointing_steps="best" \
--seed 123 \
--save_every 1
