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
Forward override added so the model itself filters unexpected kwargs, allowing
use of the default Trainer without a custom compute_loss.
"""

model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------- Dataset ---------------------------------
class ConversationDataset(Dataset):
    def __init__(self, audio_text_pairs: List[Any], processor, limit: int | None = None):
        self.pairs = audio_text_pairs if limit is None else audio_text_pairs[:limit]
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        conversation = [
            {
                "role": f"{pair.speaker_id}",
                "content": [
                    {"type": "text", "text": pair.text},
                    {"type": "audio", "path": pair.audio_path},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            output_labels=True,
        )
        return inputs


# --------------------------- Trainer -----------------------------------
class CSMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Move tensors to model device (batch_size=1 so cheap)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.device != model.device:
                inputs[k] = v.to(model.device)
        sig = inspect.signature(model.forward)
        acceptable = set(sig.parameters.keys())
        model_inputs = {k: v for k, v in inputs.items() if k in acceptable}
        outputs = model(**model_inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    processor = AutoProcessor.from_pretrained(model_id)
    model = CsmForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.train()
    if hasattr(model, "codec_model"):
        model.codec_model.eval()

    audio_text_pairs = [AudioTextPair(text="Бирок 15 мүнөт кеч калып телефонуңузду заряддап алсаңыз кеңседе эч ким каршы болбойт.",
                                      audio_path="audio/sample.wav",
                                      speaker_id=1)]
    dataset = ConversationDataset(audio_text_pairs, processor=processor, limit=1)

    # Debug shapes from first processed sample (pre-training)
    debug_sample = dataset[0].to(device)
    print("--- Shapes of Tensors in 'debug_sample' (pre-Training) ---")
    for k, v in debug_sample.items():
        print(f"{k}: {v.shape}")
    print("----------------------------------------------------------\n")

    # Manual forward after one optimization step
    manual_out = model(**debug_sample)

    print(f"Manual single forward loss: {manual_out.loss.item():.6f}")

    manual_inputs_batched = {k: v.unsqueeze(0).to(model.device) for k, v in debug_sample.items()}
    print("--- Shapes of Tensors in 'debug_sample' (pre-Training) ---")
    for k, v in manual_inputs_batched.items():
        print(f"{k}: {v.shape}")
    print("----------------------------------------------------------\n")

    with torch.no_grad():
        sig = inspect.signature(model.forward)
        print(sig)
        filtered = {k: v for k, v in manual_inputs_batched.items() if k in sig.parameters}
        manual_out = model(**filtered)

    print(f"222 Manual single forward loss: {manual_out.loss.item():.6f}")


    training_args = TrainingArguments(
        output_dir="trainer_csm_output",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=1_000_000,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],
        remove_unused_columns=False,  # keep all produced keys
    )

    trainer = CSMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor if hasattr(processor, "tokenizer") else None,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)


    print("\n--- Comparison ---")
    print(f"Trainer final loss (last logged): {train_result.training_loss}")
    print(f"Manual single forward loss: {manual_out.loss.item():.6f}")
    print("A difference is expected because weights updated once before manual forward.")


if __name__ == "__main__":
    main()
