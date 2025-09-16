import inspect
from typing import List, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    CsmForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

from lora import transcribe_audio_files, META_FILES, AudioTextPair

"""
Trainer-based reproduction of show_layer_model.py, simplified for batch_size=1.
Goal: Make Trainer step loss and manual loss comparable while avoiding CUDA OOM.
Changes:
- Manual pre-training loss computed in no_grad() (no activation graph kept)
- Enable gradient checkpointing to reduce activation memory
- Disable model.config.use_cache to save memory
- Delete intermediate tensors and empty CUDA cache before training
"""

model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Optional: quick memory reporter
def report_cuda(prefix=""):
    if not torch.cuda.is_available():
        return
    m_alloc = torch.cuda.memory_allocated() / 1024**2
    m_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{prefix} CUDA mem allocated={m_alloc:.1f}MB reserved={m_reserved:.1f}MB")

# ---------------------------- Dataset ---------------------------------
class ConversationDataset(Dataset):
    def __init__(self, audio_text_pairs: List[Any], processor, limit: int | None = None, max_audio_len: int | None = None):
        self.pairs = audio_text_pairs if limit is None else audio_text_pairs[:limit]
        self.processor = processor
        self.sample_rate = 24_000
        self.max_audio_len = max_audio_len  # optional truncate samples

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        audio = item.load_audio(self.sample_rate)
        if self.max_audio_len is not None and audio.numel() > self.max_audio_len:
            audio = audio[: self.max_audio_len]
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

# ------------------------- Custom Trainer ------------------------------
class StepLossRecordingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Use base prep for device placement
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = outputs.loss
        # Record raw (pre-division) loss for comparison
        if not hasattr(self, "step_losses"):
            self.step_losses = []
        # Only append during training forward (not evaluation)
        if model.training:
            self.step_losses.append(loss.detach().float().cpu().item())
        return (loss, outputs) if return_outputs else loss


def main():
    set_seed(SEED)
    processor = AutoProcessor.from_pretrained(model_id)
    model = CsmForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    # Memory optimizations
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False  # avoid keeping past key values
    model.train()
    if hasattr(model, "codec_model"):
        model.codec_model.eval()  # keep frozen/eval

    # Prepare data (still single sample). Optionally truncate audio to reduce memory.
    audio_text_pairs = [AudioTextPair(text="Бирок 15 мүнөт кеч калып телефонуңузду заряддап алсаңыз кеңседе эч ким каршы болбойт.",
                                      audio_path="audio/sample.wav",
                                      speaker_id=1)]
    dataset = ConversationDataset(audio_text_pairs, processor=processor, limit=1, max_audio_len=None)

    # Build the exact inputs that the first (and only) training step will see
    raw_sample = dataset[0]  # unbatched tensors
    manual_inputs = {k: v.unsqueeze(0).to(device) for k, v in raw_sample.items()}  # add batch dim

    report_cuda("Before manual forward:")

    # Manual pre-training loss WITHOUT building a grad graph
    model.eval()  # disable dropout for stable comparison
    with torch.no_grad():
        pre_out = model(**manual_inputs)
        manual_pre_loss = float(pre_out.loss)
    print(f"Manual pre-training loss (no_grad eval): {manual_pre_loss:.6f}")

    # Clean up to free memory before real training
    del pre_out
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    report_cuda("After cleanup:")

    # Switch back to train for actual optimization step
    model.train()

    training_args = TrainingArguments(
        output_dir="trainer_csm_output",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=1_000_000,
        fp16=False,  # keep fp32 for reproducible loss matching
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = StepLossRecordingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    train_result = trainer.train()

    # Trainer recorded the raw forward loss before optimizer step division
    trainer_step_loss = trainer.step_losses[0] if getattr(trainer, 'step_losses', None) else float('nan')

    # Post-update loss (eval mode again, no grad)
    model.eval()
    with torch.no_grad():
        post_out = model(**manual_inputs)
        manual_post_loss = float(post_out.loss)

    print("\n--- Loss Comparison ---")
    print(f"Manual pre-training loss : {manual_pre_loss:.6f}")
    print(f"Trainer step raw loss    : {trainer_step_loss:.6f}")
    print(f"Manual post-training loss: {manual_post_loss:.6f}")
    print("Pre ≈ trainer step (minor dropout / mode differences removed). Post reflects one update.")
    report_cuda("End:")

if __name__ == "__main__":
    main()
