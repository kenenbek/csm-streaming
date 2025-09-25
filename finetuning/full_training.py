import os
import logging
import yaml

import numpy as np
import torch
from transformers import CsmForConditionalGeneration, Trainer, TrainingArguments, CsmProcessor

from custom_dataset import parse_file_and_create_text_audio_pairs, SimpleDataset

import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)
logger = logging.getLogger(__name__)

config_file = "config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

MANIFEST = config["MANIFEST"]
MAX_AUDIO_FILES = config["MAX_AUDIO_FILES"]
SORT = config["SORT"]
REVERSE = config["REVERSE"]
OUTPUT_DIR = config["OUTPUT_DIR"]
KEEP_LAST_N_CHECKPOINTS = config["KEEP_LAST_N_CHECKPOINTS"]
LOGGING_STEPS = config["LOGGING_STEPS"]
SAVE_STEPS = config["SAVE_STEPS"]
NUM_EPOCHS = config["NUM_EPOCHS"]
BATCH_SIZE = config["BATCH_SIZE"]
GRADIENT_ACCUMULATION_STEPS = config["GRADIENT_ACCUMULATION_STEPS"]
GRADIENT_CHECKPOINTING = config["GRADIENT_CHECKPOINTING"]
LEARNING_RATE = float(config["LEARNING_RATE"])
SEED = config["SEED"]
MODEL_NAME = config["MODEL_NAME"]


# If you want to target a specific GPU (e.g., GPU 3), prefer setting this before launching:
#   CUDA_VISIBLE_DEVICES=3 python finetuning/full_training.py
# Trainer will then use cuda:0 (which is your selected physical GPU).


def prepare_csm_model_for_training():
    logger.info(f"Loading CSM model: {MODEL_NAME}")

    processor = CsmProcessor.from_pretrained(MODEL_NAME)
    # Load on CPU; Trainer/Accelerate will move to the right single device (cuda:0) automatically.
    model = CsmForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    return model, processor


def data_collator(audio_text_pairs, processor):
    conversations = []

    for pair in audio_text_pairs:
        text = pair.text

        conversations.append(
            [
                {
                "role": f"{500 + pair.speaker_id}",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "audio": pair.audio_path}
                ]
                }
            ]
        )

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        output_labels=True,
    )

    return inputs




def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.backends.cuda.enable_flash_sdp(True)
    # Enable cuDNN benchmark on any CUDA device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model, processor = prepare_csm_model_for_training()
    audio_text_pairs = parse_file_and_create_text_audio_pairs(MANIFEST, MAX_AUDIO_FILES=MAX_AUDIO_FILES)
    if not audio_text_pairs:
        logger.error(f"No audio files found or transcribed in {MANIFEST}")
        return

    dataset = SimpleDataset(
        audio_text_pairs
    )

    logger.info(f"Dataset created with {len(dataset)} samples")
    wandb.init(
        project="full-csm-finetune",
        config={
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        reinit=True
    )

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=LOGGING_STEPS,
        output_dir=f"./{OUTPUT_DIR}",
        report_to="wandb",
        save_steps=SAVE_STEPS,
        save_total_limit=KEEP_LAST_N_CHECKPOINTS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: data_collator(batch, processor),
    )

    trainer.train()

if __name__ == "__main__":
    main()