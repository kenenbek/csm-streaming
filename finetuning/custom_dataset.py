from typing import List
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)

logger = logging.getLogger(__name__)

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

class ConversationDataset(Dataset):
    def __init__(self, audio_text_pairs, processor,  sort: bool = False, reverse: bool = False):
        self.pairs = audio_text_pairs
        self.processor = processor
        self.sample_rate = processor.feature_extractor.sampling_rate

        # Precompute lengths for sorting (prefer duration, fallback to waveform length)
        self._lengths: List[int] = [self._estimate_length(p) for p in self.pairs]
        # Build index mapping (optionally sorted by length)
        indices = list(range(len(self.pairs)))
        if sort:
            indices = sorted(indices, key=lambda i: self._lengths[i], reverse=reverse)
        self._indices = indices

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        item = self.pairs[real_idx]
        audio = item.load_audio(self.sample_rate)
        text = f"<|begin_of_text|>[{item.speaker_id}]{item.text}<|end_of_text|><|AUDIO|><|audio_eos|>"
        print(text)
        inputs = self.processor(
            text=text,
            audio=audio,
            output_labels=True,
            text_kwargs={"padding": True},
            audio_kwargs={"sampling_rate": self.sample_rate},
            common_kwargs={"return_tensors": "pt"},
        )
        cleaned = {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                   for k, v in inputs.items() if torch.is_tensor(v)}

        for k, v in cleaned.items():
            if torch.is_tensor(v):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {v}")


        print("input_ids: ", cleaned["input_ids"])
        decoded_input_ids = self.processor.tokenizer.decode(cleaned["input_ids"], skip_special_tokens=False)
        print("Decoded input_ids:", decoded_input_ids)
        print("labels: ", cleaned["labels"])
        decoded_labels = self.processor.tokenizer.decode(cleaned["labels"], skip_special_tokens=False)
        print("Decoded labels:", decoded_labels)
        print("Original text: ", text)

        print("------------------------------------")
        return cleaned

    @property
    def lengths(self) -> List[int]:
        return self._lengths

    def _estimate_length(self, item) -> int:
        if hasattr(item, "duration") and item.duration:
            return int(round(item.duration * self.sample_rate))
        audio = item.load_audio(self.sample_rate)
        return int(audio.shape[-1])


def get_speaker_name(path):
    p = Path(path)
    parts = p.parts  # tuple of all path components
    # find the date folder (pattern: DD.MM.YYYY)
    for i, part in enumerate(parts):
        if len(part.split(".")) == 3:  # crude date detection
            if i + 1 < len(parts):
                return parts[i + 1]  # the folder right after the date
    return None

def parse_file_and_create_text_audio_pairs(metafile_paths: str = None, MAX_AUDIO_FILES: int = 0):
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

            if 0 < MAX_AUDIO_FILES <= len(audio_text_pairs):
                logger.info(f"Reached MAX_AUDIO_FILES limit ({MAX_AUDIO_FILES}) while reading metafile.")
                break

    return audio_text_pairs