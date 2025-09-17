import os

import torch
import logging
import numpy as np
from transformers import CsmForConditionalGeneration, Trainer, TrainingArguments, AutoProcessor, BitsAndBytesConfig
from tqdm import tqdm
import wandb
from models import Model
from moshi.models import loaders
from huggingface_hub import hf_hub_download
from tokenizers.processors import TemplateProcessing

from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import prepare_model_for_kbit_training
from lora import transcribe_audio_files
from torch.utils.data import Dataset, DataLoader
from bitsandbytes.optim import PagedAdamW8bit


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
GRADIENT_ACCUMULATION_STEPS = 32
LEARNING_RATE = 1e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "sesame/csm-1b"
MAX_AUDIO_FILES = 0

# Only keep these layers trainable
TRAINABLE_LAYERS = ["depth_decoder", "lm_head"]


class ConversationDataset(Dataset):
    def __init__(self, audio_text_pairs, processor):
        self.pairs = audio_text_pairs
        self.processor = processor
        self.sample_rate = processor.feature_extractor.sampling_rate

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        audio = item.load_audio(self.sample_rate)

        inputs = self.processor(
            text=f"<|begin_of_text|>[{item.speaker_id}]{item.text}<|end_of_text|><|AUDIO|><|audio_eos|>",
            audio=audio,
            output_labels=True,
            text_kwargs={"padding": True},
            audio_kwargs={"sampling_rate": self.sample_rate},
            common_kwargs={"return_tensors": "pt"},
        )
        cleaned = {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                   for k, v in inputs.items() if torch.is_tensor(v)}
        return cleaned


def _freeze_all_but_layers(model: torch.nn.Module, allowed_layer_names):
    """Freeze all params except those belonging to submodules whose name ends with any of allowed_layer_names.
    Matching is done against module names from model.named_modules(); if a submodule's qualified name ends with
    one of the allowed names (e.g., '...l1' or '...l2'), all of its parameters are set trainable.
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Thaw allowed submodules by module-name match
    allowed_set = set(allowed_layer_names)
    matched_modules = []
    for mod_name, module in model.named_modules():
        short = mod_name.split('.')[-1] if mod_name else mod_name
        if short in allowed_set or mod_name in allowed_set:
            for p in module.parameters(recurse=True):
                p.requires_grad = True
            matched_modules.append(mod_name)

    # As a fallback, also match by parameter-name substring contains '.l1.' or '.l2.' etc.
    for name, p in model.named_parameters():
        if any(f".{n}." in name or name.endswith(f".{n}.weight") or name.endswith(f".{n}.bias") for n in allowed_set):
            p.requires_grad = True

    # Log the trainable parameters summary
    total = sum(p.numel() for p in model.parameters())
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = sum(n for _, n in trainable)
    logger.info(f"Trainable modules matched: {matched_modules}")
    logger.info(f"Trainable params: {trainable_params:,} / {total:,} ({trainable_params / max(total,1):.2%})")


def prepare_csm_model_for_training():
    logger.info(f"Loading CSM model: {MODEL_NAME}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = CsmForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
    )

    # Freeze all but selected layers
    _freeze_all_but_layers(model, TRAINABLE_LAYERS)

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
    audio_text_pairs = transcribe_audio_files(metafile_paths=META_FILES)
    if not audio_text_pairs:
        logger.error(f"No audio files found or transcribed in {META_FILES}")
        return

    dataset = ConversationDataset(
        audio_text_pairs,
        processor=processor,
    )

    logger.info(f"Dataset created with {len(dataset)} samples")
    wandb.init(
        project="csm-finetune",
        config={
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "trainable_layers": TRAINABLE_LAYERS,
        },
        reinit=True,
    )

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=10,
        output_dir=f"./{OUTPUT_DIR}",
        report_to="wandb",
        save_steps=50,
        save_total_limit=KEEP_LAST_N_CHECKPOINTS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()