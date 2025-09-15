from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device
from datasets import load_dataset, Audio

from lora import transcribe_audio_files, META_FILES

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)
model.train()
model.codec_model.eval()

audio_text_pairs = transcribe_audio_files(metafile_paths=META_FILES)

conversation = [
    {
        "role": f"{audio_text_pairs[0].speaker_id}",
        "content": [{"type": "text", "text": audio_text_pairs[0].text},
                    {"type": "audio", "path": audio_text_pairs[0].audio_path}],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(model.device)

out = model(**inputs)
out.loss.backward()