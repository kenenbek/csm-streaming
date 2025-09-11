import os, glob, json
import logging
from tqdm import tqdm  # fixed import

from lora import AudioTextPair, TRANSCRIPTION_MODEL, MAX_AUDIO_FILES


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)
logger = logging.getLogger(__name__)

AUDIO_DIR = ""


def transcribe_audio_files(
    metafile_paths: str = None,
    delimiter: str = "|",
    strict: bool = False,
    base_dir: str = None,
    normalize_paths: bool = True,
):
    """Create list[AudioTextPair] from metafile (path|transcription) or run ASR if no metafile.

    Args:
        metafile_path: Path to metafile with lines '<audio_path>|<transcription>'. If provided & exists, ASR is skipped.
        delimiter: Delimiter separating audio path and text in metafile.
        strict: If True, raise errors on malformed lines / missing audio; else skip & warn.
        base_dir: Optional directory to prepend when relative paths (or unresolved paths) are encountered.
        normalize_paths: Replace Windows backslashes with os.sep and strip quotes.

    Returns:
        list[AudioTextPair]
    """
    audio_text_pairs = []

    # Metafile mode
    for metafile_path in metafile_paths:
        logger.info(f"Loading existing transcriptions from metafile: {metafile_path}")
        try:
            with open(metafile_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read metafile {metafile_path}: {e}")
            if strict:
                raise
            return []

        seen_paths = set()
        malformed = 0
        missing_audio = 0
        duplicates = 0
        for idx, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if delimiter not in line:
                malformed += 1
                msg = f"Line {idx} missing delimiter '{delimiter}': {line[:80]}";
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
                continue

            audio_path_part, text_part = line.split(delimiter, 1)
            audio_path = audio_path_part.strip().strip('"').strip("'")
            transcription = text_part.strip()

            if normalize_paths:
                # Convert Windows style backslashes
                audio_path = audio_path.replace('\\', os.sep)
                # Remove potential drive letter if running on non-Windows and file not found
                if os.name != 'nt' and ':' in audio_path and not os.path.isfile(audio_path):
                    # Example: D:\folder -> folder after first colon/backslash sequence
                    drive_split = audio_path.split(':', 1)[1]
                    audio_path_candidate = drive_split.lstrip('\\/').lstrip(os.sep)
                    if audio_path_candidate:
                        audio_path = audio_path_candidate

            # Attempt to resolve using base_dir if provided
            if not os.path.isabs(audio_path) and base_dir:
                candidate = os.path.join(base_dir, audio_path)
                if os.path.isfile(candidate):
                    audio_path = candidate

            # Attempt resolution relative to AUDIO_DIR if still missing
            if not os.path.isfile(audio_path) and AUDIO_DIR and not os.path.isabs(audio_path):
                candidate = os.path.join(AUDIO_DIR, audio_path)
                if os.path.isfile(candidate):
                    audio_path = candidate

            if not os.path.isfile(audio_path):
                missing_audio += 1
                msg = f"Audio file not found (line {idx}): {audio_path}";
                if strict:
                    raise FileNotFoundError(msg)
                logger.warning(msg)
                continue

            abs_path = os.path.abspath(audio_path)
            if abs_path in seen_paths:
                duplicates += 1
                logger.debug(f"Duplicate audio path skipped: {abs_path}")
                continue
            seen_paths.add(abs_path)

            if "Тимур".lower() in metafile_path.lower():
                speaker_id = 0
            elif "Айганыш".lower() in metafile_path.lower():
                speaker_id = 1
            else:
                raise ValueError()

            if "neutral".lower() in metafile_path.lower():
                tone = "<neutral>"
            elif "strict".lower() in metafile_path.lower():
                tone = "<strict>"
            else:
                raise ValueError()

            audio_text_pairs.append(AudioTextPair(audio_path=audio_path,
                                                  text=tone + " " + transcription,
                                                  speaker_id=speaker_id))

            if MAX_AUDIO_FILES > 0 and len(audio_text_pairs) >= MAX_AUDIO_FILES:
                logger.info(f"Reached MAX_AUDIO_FILES limit ({MAX_AUDIO_FILES}) while reading metafile.")
                break

        logger.info(
            "Loaded %d audio-text pairs from metafile (unique). Skipped -> malformed:%d missing:%d duplicates:%d",
            len(audio_text_pairs), malformed, missing_audio, duplicates
        )
        return audio_text_pairs


if __name__ == '__main__':
    transcribe_audio_files()