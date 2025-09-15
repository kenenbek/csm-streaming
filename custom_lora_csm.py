import os

import torch
import logging
import numpy as np
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from tqdm import tqdm
import wandb
from models import Model
from moshi.models import loaders
from huggingface_hub import hf_hub_download
from tokenizers.processors import TemplateProcessing

from peft import get_peft_model, LoraConfig, TaskType
from lora import transcribe_audio_files, CSMDataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)
logger = logging.getLogger(__name__)

PARENT_DIR = "/home/ubuntu/TTS-data-preparator/metadata"
SHORT_META_FILES = [
        "Aiganysh-neutral.txt",
        "Aiganysh-strict.txt",
        "Timur-neutral.txt",
        "Timur-strict.txt"
    ]
META_FILES = [os.path.join(PARENT_DIR, meta) for meta in SHORT_META_FILES]
OUTPUT_DIR = "finetuned_model"
KEEP_LAST_N_CHECKPOINTS = 5
NUM_EPOCHS = 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "sesame/csm-1b"
MAX_AUDIO_FILES = 0

R = 64
ALPHA = 64
LORA_DROPOUT = 0.05

def load_llama3_tokenizer():
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(bos, tokenizer.bos_token_id), (eos, tokenizer.eos_token_id)],
    )
    return tokenizer


def prepare_csm_model_for_training():
    logger.info(f"Loading CSM model: {MODEL_NAME}")
    model = Model.from_pretrained(MODEL_NAME).to(DEVICE)

    # Some fallback logic for config
    if not hasattr(model.config, 'get'):
        def get_method(self, key, default=None):
            if hasattr(self, key):
                return getattr(self, key)
            return default

        model.config.__class__.get = get_method

    text_tokenizer = load_llama3_tokenizer()
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=DEVICE)
    mimi.set_num_codebooks(32)
    audio_tokenizer = mimi

    logger.info("Applying LoRA to model using PEFT...")

    # Define the LoRA configuration using LoraConfig
    peft_config = LoraConfig(
        r=R,
        lora_alpha=ALPHA,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'output_proj', "w1", "w2", "w3"],
        lora_dropout=LORA_DROPOUT,
        bias="all",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )

    # Create the PeftModel
    model = get_peft_model(model, peft_config)

    # You can print the trainable parameters to verify
    model.print_trainable_parameters()

    return model, text_tokenizer, audio_tokenizer


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.backends.cuda.enable_flash_sdp(True)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    model, text_tokenizer, audio_tokenizer = prepare_csm_model_for_training()
    audio_text_pairs = transcribe_audio_files(metafile_paths=META_FILES)
    if not audio_text_pairs:
        logger.error(f"No audio files found or transcribed in {META_FILES}")
        return

    dataset = CSMDataset(
        audio_text_pairs,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        device=DEVICE
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

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=5,
        bf16=True,
        output_dir=f"./{OUTPUT_DIR}",
        report_to="wandb",
        save_steps=500,
        save_total_limit=KEEP_LAST_N_CHECKPOINTS,
        learning_rate=LEARNING_RATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()