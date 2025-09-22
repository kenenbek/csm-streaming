import os
import logging
import yaml

import numpy as np
import torch
from transformers import CsmForConditionalGeneration, Trainer, TrainingArguments, AutoProcessor, BitsAndBytesConfig
from custom_trainer import NoShuffleTrainer
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import prepare_model_for_kbit_training
from bitsandbytes.optim import PagedAdamW8bit, PagedLion8bit

from custom_dataset import parse_file_and_create_text_audio_pairs, ConversationDataset

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

PARENT_DIR = config["PARENT_DIR"]
SHORT_META_FILES = config["SHORT_META_FILES"]
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
LEARNING_RATE = config["LEARNING_RATE"]
SEED = config["SEED"]
MODEL_NAME = config["MODEL_NAME"]

LORA_LR = float(config["LORA_LR"])
MODULES_TO_SAVE_LR = float(config["MODULES_TO_SAVE_LR"])
R = config["R"]
ALPHA = config["ALPHA"]
LORA_DROPOUT = config["LORA_DROPOUT"]

TARGET_MODULES = config["TARGET_MODULES"]
MODULES_TO_SAVE = config["MODULES_TO_SAVE"]

META_FILES = [os.path.join(PARENT_DIR, meta) for meta in SHORT_META_FILES]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def split_trainable_params(model):
    lora_params, mts_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        else:
            mts_params.append(p)
    return lora_params, mts_params

def build_optimizer(model):
    lora_params, mts_params = split_trainable_params(model)
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": LORA_LR})
    if mts_params:
        param_groups.append({"params": mts_params, "lr": MODULES_TO_SAVE_LR})

    # Use AdamW (or bitsandbytes' PagedAdamW8bit if preferred)
    optimizer = PagedAdamW8bit(param_groups, betas=(0.9, 0.999), weight_decay=0.01)
    return optimizer


def prepare_csm_model_for_training():
    logger.info(f"Loading CSM model: {MODEL_NAME}")

    # Choose compute dtype based on hardware: bf16 on Ampere+/CUDA if available, else fp16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        compute_dtype = torch.float32

    # Load model in 4-bit with NF4 quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = CsmForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
        dtype=compute_dtype,
    )
    logger.info(f"Model loaded with 4-bit: {getattr(model, 'is_loaded_in_4bit', False)}; dtype: {compute_dtype}")

    logger.info("Applying LoRA to model using PEFT...")
    peft_config = LoraConfig(
        r=R,
        lora_alpha=ALPHA,
        target_modules=TARGET_MODULES,
        modules_to_save=MODULES_TO_SAVE,
        lora_dropout=LORA_DROPOUT,
        bias="all",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.backends.cuda.enable_flash_sdp(True)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    model, processor = prepare_csm_model_for_training()
    audio_text_pairs = parse_file_and_create_text_audio_pairs(metafile_paths=META_FILES, MAX_AUDIO_FILES=MAX_AUDIO_FILES)
    if not audio_text_pairs:
        logger.error(f"No audio files found or transcribed in {META_FILES}")
        return

    dataset = ConversationDataset(
        audio_text_pairs,
        processor=processor,
        sort=SORT,
        reverse=REVERSE,
    )

    logger.info(f"Dataset created with {len(dataset)} samples")
    wandb.init(
        project="csm-finetune",
        config={
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lora_rank": R,
            "lora_alpha": ALPHA,
            "lora_dropout": LORA_DROPOUT,
        },
        reinit=True,
    )

    # Precision flags aligned with quantization compute dtype
    precision_kwargs = {}
    if DEVICE == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            precision_kwargs["bf16"] = True
        else:
            precision_kwargs["fp16"] = True

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
        **precision_kwargs,
    )

    optimizer = build_optimizer(model)

    trainer = NoShuffleTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        optimizers=(optimizer, None),
    )

    trainer.train()

if __name__ == "__main__":
    main()