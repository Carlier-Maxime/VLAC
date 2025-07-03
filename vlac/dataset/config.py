from typing import Tuple

COYO_LABELS_PATH: str = "/path/to/coyo/labels"
COYO_LABELS_KEYS_READ: Tuple[str, ...] = ("id", "url", "text")
COYO_LABELS_KEYS_OUT: Tuple[str, ...] = ("id", "url", "text")

COYO_PATH: str = "/path/to/coyo"
COYO_KEYS_READ: Tuple[str, ...] = ("img", "text")
COYO_KEYS_OUT: Tuple[str, ...] = ("vision", "text_tokens")

EMBEDS_PATH: str = "/path/to/embeds"
EMBEDS_KEYS_READ: Tuple[str, ...] = ("embeds.pth", "mask.pth", "tokens.pth")
EMBEDS_KEYS_OUT: Tuple[str, ...] = ("embeds", "attention_mask", "input_ids")

MINERL_PATH: str = "/path/to/minerl"
MINERL_KEYS_READ: Tuple[str, ...] = ('video.mp4', 'vid_len', 'fps', 'vid_width', 'vid_height', 'task_desc', 'infos.jsonl')
MINERL_KEYS_OUT: Tuple[str, ...] = ("video", "vid_len", "fps", "vid_width", "vid_height", "task", "infos")
