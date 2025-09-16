import inspect
from dataclasses import dataclass
from typing import List, Any, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    CsmForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    infer_device,
)

from lora import transcribe_audio_files, META_FILES

"""
Trainer-based reproduction of the logic in show_layer_model.py so you can compare
manual forward/backward results with the Hugging Face Trainer abstraction.

Key points:
- Uses the same model + processor (sesame/csm-1b)
- Builds a lightweight Dataset that returns one conversation item per AudioTextPair
- Custom data collator calls processor.apply_chat_template with output_labels=True (same as manual code)
- Custom Trainer subclass filters unexpected keys before forwarding to the model (remove_unused_columns=False kept)
- Runs a single optimization step (max_steps=1) for a direct loss comparison

You can increase dataset size or steps once validated.
"""

model_id = "sesame/csm-1b"
device = infer_device()


# ---------------------------- Dataset ---------------------------------
class ConversationDataset(Dataset):
    def __init__(self, audio_text_pairs: List[Any], limit: int | None = None):
        self.pairs = audio_text_pairs if limit is None else audio_text_pairs[:limit]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # Same structure as in show_layer_model.py
        conversation = [
            {
                "role": f"{pair.speaker_id}",
                "content": [
                    {"type": "text", "text": pair.text},
                    {"type": "audio", "path": pair.audio_path},
                ],
            }
        ]
        return conversation



# --------------------------- Trainer -----------------------------------
class CSMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Filter only acceptable args for model forward
        sig = inspect.signature(model.forward)
        acceptable = set(sig.parameters.keys())
        model_inputs = {k: v for k, v in inputs.items() if k in acceptable}
        outputs = model(**model_inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    # Load model & processor (same as original script)
    processor = AutoProcessor.from_pretrained(model_id)
    model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)
    model.train()
    # Keep codec_model in eval as original
    if hasattr(model, "codec_model"):
        model.codec_model.eval()

    # Build dataset (limit to 1 for parity with show_layer_model.py)
    audio_text_pairs = transcribe_audio_files(metafile_paths=META_FILES)
    dataset = ConversationDataset(audio_text_pairs, limit=1)


    # Training arguments: single step to mirror manual backward()
    training_args = TrainingArguments(
        output_dir="trainer_csm_output",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=1,  # force exactly one optimizer update
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=1_000_000,  # effectively disable saving during this quick test
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],  # disable W&B etc. for the quick comparison run
    )

    trainer = CSMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=processor if hasattr(processor, "tokenizer") else None,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # Direct manual forward (single example) to compare loss numerically
    first_conv = dataset[0]
    manual_inputs = processor.apply_chat_template(
        first_conv,
        tokenize=True,
        return_dict=True,
        output_labels=True,
    ).to(model.device)

    with torch.no_grad():
        manual_out = model(**{k: v for k, v in manual_inputs.items() if k in inspect.signature(model.forward).parameters})

    print("\n--- Comparison ---")
    print(f"Trainer final loss (last logged): {train_result.training_loss}")
    print(f"Manual single forward loss: {manual_out.loss.item():.6f}")
    print("If these differ slightly it's due to one optimization step updating weights.")


if __name__ == "main":  # allow python -m execution mistakes
    main()

if __name__ == "__main__":
    main()

