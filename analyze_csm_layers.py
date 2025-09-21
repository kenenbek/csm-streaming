from transformers import CsmForConditionalGeneration


MODEL_NAME = "sesame/csm-1b"

model = CsmForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
    )

print(model)

model.