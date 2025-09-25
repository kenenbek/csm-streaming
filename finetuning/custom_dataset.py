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

        # Build index mapping (optionally sorted by length)
        indices = list(range(len(self.pairs)))
        if sort:
            self._lengths: List[int] = [self._estimate_length(p) for p in self.pairs]
            indices = sorted(indices, key=lambda i: self._lengths[i], reverse=reverse)
        self._indices = indices

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        item = self.pairs[real_idx]
        audio = item.load_audio(self.sample_rate)
        text = f"<|begin_of_text|>[{item.speaker_id}]{item.text}<|end_of_text|><|AUDIO|><|audio_eos|>"
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

        return cleaned

    @property
    def lengths(self) -> List[int]:
        return self._lengths

    def _estimate_length(self, item) -> int:
        if hasattr(item, "duration") and item.duration:
            return int(round(item.duration * self.sample_rate))
        audio = item.load_audio(self.sample_rate)
        return int(audio.shape[-1])


def parse_file_and_create_text_audio_pairs(MANIFEST: str = None, MAX_AUDIO_FILES: int = 0):
    audio_text_pairs = []

    # Metafile mode
    meta_df = pd.read_csv(MANIFEST, sep="|", header=None)
    # Iterate over rows
    for _, row in meta_df.iterrows():
        local_path = row[0]
        speaker = row[1]
        tone = row[2]
        _ = row[3]
        transcription = row[4]

        if "Timur" == speaker:
            speaker_id = 0
        elif "Aiganysh" == speaker:
            speaker_id = 1
        else:
            print(speaker)
            print(local_path)
            raise ValueError()

        audio_text_pairs.append(AudioTextPair(audio_path=local_path,
                                              text="<" + tone + ">" + " " + transcription,
                                              speaker_id=speaker_id))

        if 0 < MAX_AUDIO_FILES <= len(audio_text_pairs):
            logger.info(f"Reached MAX_AUDIO_FILES limit ({MAX_AUDIO_FILES}) while reading metafile.")
            break

    return audio_text_pairs