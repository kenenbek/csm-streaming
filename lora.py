import json
import os
import glob
from pathlib import Path
import shutil

import torch
import torchaudio
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from safetensors.torch import save_file
import csv
from models import Model
from moshi.models import loaders
from huggingface_hub import hf_hub_download
from tokenizers.processors import TemplateProcessing
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import torch.nn as nn

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
NUM_EPOCHS = 50
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 0.1
NUM_CYCLES = 1.0
USE_WANDB = True
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = True
WARMUP_STEPS = 50
MODEL_NAME = "sesame/csm-1b"
TRANSCRIPTION_MODEL = "openai/whisper-large-v3-turbo"
MAX_AUDIO_FILES = 0
R = 128
APLHA = 128


def prune_old_checkpoints(output_dir: str, keep: int, pattern: str = "checkpoint-epoch-") -> None:
    """Keep only the newest `keep` checkpoint directories inside `output_dir`.

    A checkpoint directory is recognized by starting with `pattern` and ending
    with an integer epoch number (e.g. checkpoint-epoch-3).
    If `keep` <= 0 nothing is removed.
    Silently returns if directory does not exist or there are not enough checkpoints.
    """
    try:
        if keep is None or keep <= 0:
            return
        if not os.path.isdir(output_dir):
            return
        entries = []
        for name in os.listdir(output_dir):
            full = os.path.join(output_dir, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith(pattern):
                continue
            # Extract numeric suffix
            try:
                epoch_str = name.split(pattern, 1)[1]
                epoch_num = int(epoch_str)
            except (IndexError, ValueError):
                continue
            entries.append((epoch_num, full))
        if len(entries) <= keep:
            return
        # Sort descending by epoch so newest first
        entries.sort(key=lambda x: x[0], reverse=True)
        to_delete = entries[keep:]
        for epoch_num, path in to_delete:
            try:
                shutil.rmtree(path)
                logger.info(f"Pruned old checkpoint (epoch {epoch_num}): {path}")
            except Exception as e:
                logger.warning(f"Failed pruning checkpoint {path}: {e}")
    except Exception as e:
        logger.warning(f"prune_old_checkpoints error: {e}")


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=32, alpha=64, dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # The base linear (frozen).
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=bias)

        # LoRA trainable matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normal forward with frozen weight
        result = F.linear(x, self.weight, self.bias)

        # LoRA forward with trainable A and B
        lora_out = F.linear(self.dropout(x), self.lora_A)  # [*, r]
        lora_out = F.linear(lora_out, self.lora_B)  # [*, out_features]
        return result + self.scaling * lora_out


def replace_linear_with_lora(module: nn.Module,
                             r=R,
                             alpha=APLHA,
                             dropout=0.0,
                             target_linear_names=None):
    """
    Recursively replace Linear layers that match the given target_linear_names
    with LoRALinear. If target_linear_names is None, it will replace all nn.Linear.
    Return the modified module.
    """
    for name, child in list(module.named_children()):
        # Recursively apply to children
        replaced_child = replace_linear_with_lora(
            child, r=r, alpha=alpha, dropout=dropout, target_linear_names=target_linear_names
        )
        setattr(module, name, replaced_child)

    # If this is a top-level Linear, check if we should replace it
    if isinstance(module, nn.Linear):
        # If no target names provided, replace every linear
        # Otherwise, replace only if the name is in target_linear_names
        if (target_linear_names is None) or any(
                t in module._get_name().lower() for t in target_linear_names
        ):
            # Gather info
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            # Create LoRALinear
            lora_linear = LoRALinear(
                in_features=in_features,
                out_features=out_features,
                r=r,
                alpha=alpha,
                dropout=dropout,
                bias=False,
            )

            # Copy the original weights
            with torch.no_grad():
                lora_linear.weight.copy_(module.weight.data)
                if bias:
                    lora_linear.bias.copy_(module.bias.data)

            return lora_linear
    return module


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


@dataclass
class AudioTextPair:
    audio_path: str
    text: str
    speaker_id: int

    def load_audio(self, sample_rate=24000) -> torch.Tensor:
        waveform, sr = torchaudio.load(self.audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        processed_audio = waveform.squeeze(0)
        return processed_audio


class CSMDataset(Dataset):
    def __init__(self, data_items, text_tokenizer, audio_tokenizer, device):
        self.data_items = data_items
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.sample_rate = audio_tokenizer.sample_rate

    def __len__(self):
        return len(self.data_items)

    def tokenize_text_segment(self, text: str, speaker: int):
        text_tokens = self.text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        return text_frame, text_frame_mask

    def tokenize_audio(self, audio: torch.Tensor):
        assert audio.ndim == 1, "Audio must be single channel"
        audio_device = next(self.audio_tokenizer.parameters()).device
        audio = audio.to(audio_device)

        try:
            audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
            eos_frame = torch.zeros(audio_tokens.size(0), 1, device=audio_device)
            audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

            audio_frame = torch.zeros(audio_tokens.size(1), 33, device=audio_device).long()
            audio_frame_mask = torch.zeros(audio_tokens.size(1), 33, device=audio_device).bool()
            audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
            audio_frame_mask[:, :-1] = True
        except RuntimeError as e:
            logger.warning(f"Error encoding audio: {e}, using empty frames")
            audio_frame = torch.zeros(1, 33, device=audio_device).long()
            audio_frame_mask = torch.zeros(1, 33, device=audio_device).bool()

        return audio_frame, audio_frame_mask

    def __getitem__(self, idx: int):
        item = self.data_items[idx]
        audio = item.load_audio(self.sample_rate)

        text_tokens, text_masks = self.tokenize_text_segment(item.text, item.speaker_id)
        audio_tokens, audio_masks = self.tokenize_audio(audio)

        device = audio_tokens.device
        text_tokens = text_tokens.to(device)
        text_masks = text_masks.to(device)

        input_tokens = text_tokens
        input_masks = text_masks

        target_tokens = torch.cat([text_tokens, audio_tokens], dim=0)
        target_masks = torch.cat([text_masks, audio_masks], dim=0)

        if device != self.device:
            input_tokens = input_tokens.to(self.device)
            input_masks = input_masks.to(self.device)
            target_tokens = target_tokens.to(self.device)
            target_masks = target_masks.to(self.device)

        return {
            "input_tokens": input_tokens,
            "input_masks": input_masks,
            "target_tokens": target_tokens,
            "target_masks": target_masks,
        }


def collate_fn(batch):
    max_seq_len = 1024
    device = batch[0]["input_tokens"].device

    max_input_len = min(max(item["input_tokens"].size(0) for item in batch), max_seq_len)
    max_target_len = min(max(item["target_tokens"].size(0) for item in batch), max_seq_len)

    batch_input_tokens = []
    batch_input_masks = []
    batch_target_tokens = []
    batch_target_masks = []

    for item in batch:
        input_tokens = item["input_tokens"][:max_input_len]
        input_masks = item["input_masks"][:max_input_len]
        target_tokens = item["target_tokens"][:max_target_len]
        target_masks = item["target_masks"][:max_target_len]

        input_tokens = F.pad(input_tokens, (0, 0, 0, max_input_len - input_tokens.size(0)), "constant", 0)
        input_masks = F.pad(input_masks, (0, 0, 0, max_input_len - input_masks.size(0)), "constant", False)

        target_tokens = F.pad(target_tokens, (0, 0, 0, max_target_len - target_tokens.size(0)), "constant", 0)
        target_masks = F.pad(target_masks, (0, 0, 0, max_target_len - target_masks.size(0)), "constant", False)

        batch_input_tokens.append(input_tokens)
        batch_input_masks.append(input_masks)
        batch_target_tokens.append(target_tokens)
        batch_target_masks.append(target_masks)

    return {
        "input_tokens": torch.stack(batch_input_tokens),
        "input_masks": torch.stack(batch_input_masks),
        "target_tokens": torch.stack(batch_target_tokens),
        "target_masks": torch.stack(batch_target_masks),
        "positions": torch.arange(0, max_target_len).unsqueeze(0).repeat(len(batch), 1).to(device)
    }


def get_speaker_name(path):
    p = Path(path)
    parts = p.parts  # tuple of all path components
    # find the date folder (pattern: DD.MM.YYYY)
    for i, part in enumerate(parts):
        if len(part.split(".")) == 3:  # crude date detection
            if i + 1 < len(parts):
                return parts[i + 1]  # the folder right after the date
    return None

import pandas as pd
def transcribe_audio_files(metafile_paths: str = None):
    audio_text_pairs = []

    # Metafile mode
    for metafile_path in metafile_paths:
        meta_df = pd.read_csv(metafile_path, sep="|", header=None)

        # Iterate over rows
        for _, row in meta_df.iterrows():
            local_path = row[0]
            transcription = row[1]

            # Get parent directory
            speaker_name = get_speaker_name(local_path)  # Output: Айганыш

            if "Тимур" == speaker_name:
                speaker_id = 0
            elif "Айганыш" == speaker_name or "Айганыш" == speaker_name:
                speaker_id = 1
            else:
                print(speaker_name)
                print(local_path)
                raise ValueError()

            if "neutral".lower() in metafile_path.lower():
                tone = "<neutral>"
            elif "strict".lower() in metafile_path.lower():
                tone = "<strict>"
            else:
                raise ValueError()

            audio_text_pairs.append(AudioTextPair(audio_path=local_path,
                                                  text=tone + " " + transcription,
                                                  speaker_id=speaker_id))

            if MAX_AUDIO_FILES > 0 and len(audio_text_pairs) >= MAX_AUDIO_FILES:
                logger.info(f"Reached MAX_AUDIO_FILES limit ({MAX_AUDIO_FILES}) while reading metafile.")
                break

    return audio_text_pairs


def prepare_csm_model_for_training():
    logger.info(f"Loading CSM model: {MODEL_NAME}")
    model = Model.from_pretrained(MODEL_NAME).to(DEVICE)

    text_tokenizer = load_llama3_tokenizer()
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=DEVICE)
    mimi.set_num_codebooks(32)
    audio_tokenizer = mimi
    try:

        codebook_0_centroids = mimi.quantizer.rvq_first.layers[0].codebook.weight.data

        num_codebook_0_tokens, embedding_dim = codebook_0_centroids.shape
        model.codebook_embedding = nn.Embedding(num_codebook_0_tokens, embedding_dim).to(DEVICE)
        model.codebook_embedding.weight.data.copy_(codebook_0_centroids)
        logger.info(f"Successfully initialized codebook_embedding with shape: {codebook_0_centroids.shape}")

    except AttributeError:
        num_codebook_0_tokens, embedding_dim = 1024, 1024
        model.codebook_embedding = nn.Embedding(num_codebook_0_tokens, embedding_dim).to(DEVICE)
        nn.init.xavier_uniform_(model.codebook_embedding.weight)

    except Exception as e:
        num_codebook_0_tokens, embedding_dim = 1024, 1024
        model.codebook_embedding = nn.Embedding(num_codebook_0_tokens, embedding_dim).to(DEVICE)
        nn.init.xavier_uniform_(model.codebook_embedding.weight)

    # Some fallback logic for config
    if not hasattr(model.config, 'get'):
        def get_method(self, key, default=None):
            if hasattr(self, key):
                return getattr(self, key)
            return default

        model.config.__class__.get = get_method
    if not hasattr(model.config, 'tie_word_embeddings'):
        model.config.tie_word_embeddings = False
    target_layers = ['q_proj',
                     'k_proj',
                     'v_proj',
                     'output_proj',
                     "w1",
                     "w2",
                     "w3"]
    logger.info("Applying LoRA to model...")
    model = replace_linear_with_lora(
        model,
        r=R,
        alpha=APLHA,
        dropout=0.01,
        target_linear_names=target_layers
    )
    model.cuda()

    # First, freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False

    # Then, unfreeze only the newly added LoRA parameters.
    # It is also common practice to train the bias parameters.
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "bias" in name:
            param.requires_grad = True

    return model, text_tokenizer, audio_tokenizer


def setup_model_caches(model, batch_size):
    try:
        with torch.no_grad():
            model.reset_caches()
            model.backbone.reset_caches()
            model.decoder.reset_caches()
    except Exception as e:
        logger.debug(f"No caches to reset or error: {e}")
    return True


class BridgingModule(nn.Module):
    """For a 2048->1024 bridging if needed."""

    def __init__(self, in_dim=2048, out_dim=1024):
        super().__init__()
        self.bridge = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.bridge.weight)

    def forward(self, x):
        return self.bridge(x)


def compute_loss_for_codebooks_single_pass(
        backbone_out,  # [b, seq_len, 2048]
        decoder_out,  # [b, seq_len, 1024]
        model,
        target_tokens,  # [b, seq_len, codebooks]
        target_masks,  # [b, seq_len, codebooks bool]
        device
):
    bsz, seq_len = target_tokens.size()[:2]
    num_codebooks = model.config.audio_num_codebooks

    c0_logits = model.codebook0_head(backbone_out)
    audio_positions = target_masks[..., :-1].any(dim=-1)  # [b, seq_len] for audio

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    # codebook0
    for b in range(bsz):
        for s in range(seq_len):
            if audio_positions[b, s]:
                token_logits = c0_logits[b, s]
                target_token = target_tokens[b, s, 0]
                if target_token > 0:
                    ce = F.cross_entropy(token_logits.unsqueeze(0), target_token.unsqueeze(0), reduction='sum')
                    total_loss += ce
                    count += 1

    # codebooks [1..N-1] from decoder_out
    for i in range(1, num_codebooks):
        weight_i = model.audio_head[i - 1]
        flat_dec = decoder_out.reshape(bsz * seq_len, -1)
        token_logits_all = flat_dec.mm(weight_i)

        for b in range(bsz):
            for s in range(seq_len):
                if audio_positions[b, s]:
                    target_token = target_tokens[b, s, i]
                    if target_token > 0:
                        row_idx = b * seq_len + s
                        row_logits = token_logits_all[row_idx]
                        ce = F.cross_entropy(row_logits.unsqueeze(0), target_token.unsqueeze(0), reduction='sum')
                        total_loss += ce
                        count += 1

    if count > 0:
        total_loss = total_loss / count
    return total_loss


def single_pass_forward(model, bridging_module, target_tokens, target_masks, positions):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    embed = model._embed_tokens(target_tokens)
    masked_embed = embed * target_masks.unsqueeze(-1)
    h = masked_embed.sum(dim=2)

    backbone_out = model.backbone(h, input_pos=positions, mask=None).to(dtype)
    bridging_out = bridging_module(backbone_out)

    codebook0_logits = model.codebook0_head(backbone_out)
    codebook0_tokens = torch.argmax(codebook0_logits, dim=-1).clamp(0, model.codebook_embedding.num_embeddings - 1)
    c0_embed = model.codebook_embedding(codebook0_tokens)

    # Get the last hidden state from bridging module
    last_h = bridging_out[:, -1, :].unsqueeze(1)

    # Concatenate the last hidden state with the codebook embeddings
    decoder_input = torch.cat([last_h, c0_embed], dim=1)

    # Process decoder inputs in parallel
    B, S, D = decoder_input.shape  # Batch, Sequence length, Dimension

    # Reshape to (B*S, D) to process all tokens in parallel
    decoder_input_flat = decoder_input.reshape(-1, D).unsqueeze(1)  # [B*S, 1, D] safer than view

    # Run decoder on all inputs in parallel
    decoder_out_flat = model.decoder(decoder_input_flat).to(dtype)  # [B*S, 1, output_dim]

    # Reshape back to original batch and sequence dimensions
    decoder_out = decoder_out_flat.reshape(B, S, -1)  # [B, S, output_dim]

    # Remove the first token (corresponding to last_h) as in original code
    decoder_out = decoder_out[:, 1:, :]  # [B, T, 1024]

    # Safety check: handle empty sequences
    if decoder_out.size(1) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    loss = compute_loss_for_codebooks_single_pass(
        backbone_out=backbone_out,
        decoder_out=decoder_out,
        model=model,
        target_tokens=target_tokens[..., 1:],  # Drop codebook 0
        target_masks=target_masks[..., 1:],
        device=device
    )

    return loss


def strip_bias_keys(state_dict: dict) -> dict:
    new_sd = {}
    for k, v in state_dict.items():
        if k == "codebook_embedding.weight":
            print(f"Stripping {k} from checkpoint (training-only layer)")
            continue
        if not k.endswith(".bias"):
            new_sd[k] = v
        else:
            print(f"Stripping {k} from checkpoint")
    return new_sd


def remove_lora_modules(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        new_child = remove_lora_modules(child)
        setattr(module, name, new_child)

    if isinstance(module, LoRALinear):
        out_features, in_features = module.out_features, module.in_features

        # Determine if we actually need a bias
        has_bias = (module.bias is not None)
        new_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias
        )

        # Copy over the merged weight
        new_linear.weight.data.copy_(module.weight.data)

        # If we had a bias in LoRALinear, copy it too
        if has_bias:
            new_linear.bias.data.copy_(module.bias.data)

        return new_linear

    return module


def merge_lora_layer(lora_module: LoRALinear):
    """
    Merge the LoRA params (lora_A, lora_B) into the base weight in-place.
    This transforms the LoRALinear into a standard Linear equivalent.
    """
    # W = W + (alpha/r) * (lora_B @ lora_A)
    merged_delta = lora_module.scaling * (lora_module.lora_B @ lora_module.lora_A)
    lora_module.weight.data += merged_delta

    # Optionally zero out LoRA parameters so they no longer affect anything
    lora_module.lora_A.data.zero_()
    lora_module.lora_B.data.zero_()


def merge_lora_weights(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            merge_lora_layer(module)
    return model


def finetune(model, dataset):
    logger.info("Starting finetuning process")
    csv_file = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "global_step", "loss", "learning_rate", "val_loss"])

    bridging_module = BridgingModule(in_dim=2048, out_dim=1024).to(DEVICE)
    for param in bridging_module.parameters():
        param.requires_grad = True

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(bridging_module.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    num_training_steps = len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps
    )

    if USE_WANDB:
        wandb.init(project="csm-finetuning")

    scaler = torch.amp.GradScaler() if MIXED_PRECISION else None
    global_step = 0

    model.train()
    bridging_module.train()

    current_loss = 0.0
    current_lr = LEARNING_RATE

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(dataloader):
            try:
                setup_model_caches(model, batch["target_tokens"].size(0))

                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=MIXED_PRECISION):
                    loss = forward_and_loss(model, bridging_module, batch, DEVICE)
                    if GRADIENT_ACCUMULATION_STEPS > 1:
                        loss = loss / GRADIENT_ACCUMULATION_STEPS

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf loss detected at step {step}. Skipping batch.")
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    continue

                if MIXED_PRECISION:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
                    if MIXED_PRECISION:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)

                    if MIXED_PRECISION:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad()

                    current_lr = optimizer.param_groups[0]["lr"]
                    current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS if GRADIENT_ACCUMULATION_STEPS > 1 else loss.item()
                    current_epoch = epoch + (step + 1) / len(dataloader)

                    global_step += 1

                    if USE_WANDB:
                        wandb.log({"loss": current_loss, "learning_rate": current_lr, "epoch": current_epoch,
                                   "global_step": global_step})

                    progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{current_lr:.2e}"})

                progress_bar.update(1)

            except Exception as e:
                logger.error(f"Error in batch {step}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                except:
                    pass
                progress_bar.update(1)
                continue

        checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_tensors = {
            **model.state_dict(),
            **bridging_module.state_dict()
        }
        save_file(checkpoint_tensors, os.path.join(checkpoint_dir, "model.safetensors"))

        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        # Prune older checkpoints if exceeding retention limit
        prune_old_checkpoints(OUTPUT_DIR, KEEP_LAST_N_CHECKPOINTS)

    logger.info("Merging LoRA weights into the base model...")
    merge_lora_weights(model)
    model = remove_lora_modules(model)
    merged_state = strip_bias_keys(model.state_dict())

    final_merged_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    save_file(merged_state, final_merged_path)
    logger.info(f"LoRA-merged & replaced model saved to {final_merged_path}")

    if USE_WANDB:
        wandb.finish()

    return model


def forward_and_loss(model, bridging_module, batch, device):
    target_tokens = batch["target_tokens"].to(device)
    target_masks = batch["target_masks"].to(device)
    positions = batch["positions"].to(device)

    input_tokens = target_tokens[:, :-1]
    input_masks = target_masks[:, :-1]
    input_positions = positions[:, :-1]
    labels = target_tokens[:, 1:]
    label_masks = target_masks[:, 1:]

    if input_tokens.size(1) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    # 1. Embed tokens and apply mask
    embed = model._embed_tokens(input_tokens)
    masked_embed = embed * input_masks.unsqueeze(-1)
    h = masked_embed.sum(dim=2)

    # 2. Pass through the backbone
    backbone_out = model.backbone(h, input_pos=input_positions, mask=None)

    # 3. Calculate loss for all codebooks
    loss_fct = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    num_codebooks_with_loss = 0

    c0_logits = model.codebook0_head(backbone_out)  # [B, T, V0]
    c0_labels = labels[..., 0]                     # [B, T]
    active_mask = label_masks[..., 0].reshape(-1)   # flatten safely

    if active_mask.any():
        flat_c0_logits = c0_logits.reshape(-1, c0_logits.size(-1))
        flat_c0_labels = c0_labels.reshape(-1)
        active_logits = flat_c0_logits[active_mask]
        active_labels = flat_c0_labels[active_mask]
        c0_loss = loss_fct(active_logits, active_labels)
        total_loss += c0_loss
        num_codebooks_with_loss += 1

    decoder_states = bridging_module(backbone_out)  # [B, T, D]

    num_codebooks = model.config.audio_num_codebooks
    for i in range(1, num_codebooks):
        if hasattr(model, 'audio_head') and len(model.audio_head) >= i:
            weight_i = model.audio_head[i - 1]
            logits_i = decoder_states @ weight_i  # [B, T, V_i]
            labels_i = labels[..., i]
            active_mask_i = label_masks[..., i].reshape(-1)
            if active_mask_i.any():
                flat_logits_i = logits_i.reshape(-1, logits_i.size(-1))
                flat_labels_i = labels_i.reshape(-1)
                active_logits_i = flat_logits_i[active_mask_i]
                active_labels_i = flat_labels_i[active_mask_i]
                loss_i = loss_fct(active_logits_i, active_labels_i)
                total_loss += loss_i
                num_codebooks_with_loss += 1

    if num_codebooks_with_loss > 0:
        return total_loss / num_codebooks_with_loss
    return torch.tensor(0.0, requires_grad=True, device=device)


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

    try:
        finetune(model, dataset)
        logger.info("Finetuning completed successfully!")
    except Exception as e:
        logger.error(f"Error during finetuning: {e}")
        import traceback
        logger.error(traceback.format_exc())

        try:
            # If there's an error, at least save a partial state
            partial_path = os.path.join(OUTPUT_DIR, "model_partial.safetensors")
            torch.save(model.state_dict(), partial_path)
            logger.info(f"Saved partial model to {partial_path} despite errors")
        except Exception as save_error:
            logger.error(f"Could not save partial model: {save_error}")


if __name__ == "__main__":
    main()
