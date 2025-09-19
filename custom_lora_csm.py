import os

import torch
import logging
import numpy as np
from transformers import CsmForConditionalGeneration, Trainer, TrainingArguments, AutoProcessor, BitsAndBytesConfig
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import prepare_model_for_kbit_training
from lora import transcribe_audio_files
from torch.utils.data import Dataset
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
GRADIENT_CHECKPOINTING = True
LEARNING_RATE = 5e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "sesame/csm-1b"
MAX_AUDIO_FILES = 0

LORA_LR = 1e-4
MODULES_TO_SAVE_LR = 1e-5
R = 32
ALPHA = 64
LORA_DROPOUT = 0.05

TARGET_MODULES = [
    # Backbone model attention layers
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",

    # Backbone model MLP layers
    "gate_proj",
    "up_proj",
    "down_proj"
]

MODULES_TO_SAVE = ["embed_text_tokens",]
                   # "embed_tokens",
                   # "inputs_embeds_projector",
                   # "codebooks_head"]

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
        torch_dtype=compute_dtype,
    )
    logger.info(f"Model loaded with 4-bit: {getattr(model, 'is_loaded_in_4bit', False)}; dtype: {compute_dtype}")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

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
        logging_steps=10,
        output_dir=f"./{OUTPUT_DIR}",
        report_to="wandb",
        save_steps=50,
        save_total_limit=KEEP_LAST_N_CHECKPOINTS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        **precision_kwargs,
    )

    optimizer = build_optimizer(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        optimizers=(optimizer, None),
    )

    trainer.train()

if __name__ == "__main__":
    main()