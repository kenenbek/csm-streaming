import os, glob, json
import logging
from tqdm import tqdm  # fixed import

from lora import AudioTextPair, TRANSCRIPTION_MODEL, MAX_AUDIO_FILES
import pandas as pd
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)
logger = logging.getLogger(__name__)

AUDIO_DIR = ""

def get_speaker_name(path):
    p = Path(path)
    parts = p.parts  # tuple of all path components
    # find the date folder (pattern: DD.MM.YYYY)
    for i, part in enumerate(parts):
        if len(part.split(".")) == 3:  # crude date detection
            if i + 1 < len(parts):
                return parts[i + 1]  # the folder right after the date
    return None

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
            speaker_name = get_speaker_name(local_path) # Output: Айганыш

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


# ----------------- Hugging Face datasets utilities -----------------

from datasets import load_dataset, concatenate_datasets, Audio

def prepare_data(dataset_names):
    ds1 = load_dataset(dataset_names[0], split="train", cache_dir="/mnt/d/cache_hugging_face_datasets")
    ds2 = load_dataset(dataset_names[1], split="train", cache_dir="/mnt/d/cache_hugging_face_datasets")
    ds3 = load_dataset(dataset_names[2], split="train", cache_dir="/mnt/d/cache_hugging_face_datasets")
    ds4 = load_dataset(dataset_names[3], split="train", cache_dir="/mnt/d/cache_hugging_face_datasets")

    ds = concatenate_datasets([ds1, ds2, ds3, ds4])
    print(f"[INFO] Combined dataset size: {ds.num_rows}")

    ds = ds.cast_column("audio", Audio(decode=False))  # don't use torchcodec

    for row in ds:
        audio_text_pairs.append(AudioTextPair(audio_path=audio_path,
                                              text=tone + " " + transcription,
                                              speaker_id=speaker_id))

    for row in ds:
        print(row['text'])


if __name__ == '__main__':
    dataset_names = [r"MbankAI/Timur-strict-raw-wav",
                  r"MbankAI/Aiganysh-strict-raw-wav",
                  r"MbankAI/Timur-neutral-raw-wav",
                  r"MbankAI/Aiganysh-neutral-raw-wav"]

    parent_dir = "/mnt/c/Users/k_arzymatov/PycharmProjects/TTS-data-preparator/metadata"
    meta_files = [
        "Aiganysh-neutral-linux.txt",
        "Aiganysh-strict-linux.txt",
        "Timur-neutral-linux.txt",
        "Timur-strict-linux.txt"
    ]
    metas = [os.path.join(parent_dir, meta) for meta in meta_files]

    transcribe_audio_files(metafile_paths=metas)


    #prepare_data(dataset_names=dataset_names)
