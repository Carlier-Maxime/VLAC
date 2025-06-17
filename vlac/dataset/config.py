from typing import Tuple

COYO_PATH: str = "/media/hdd/datasets/coyo-700m"
COYO_LENGTH: int = 4096
COYO_SHUFFLE: int = 100
COYO_KEYS_READ: Tuple[str, ...] = ("png", "text")
COYO_KEYS_OUT: Tuple[str, ...] = ("vision", "text_tokens")

EMBEDS_PATH: str = "/media/hdd/datasets/coyo_embeds"
EMBEDS_LENGTH: int = 4000
EMBEDS_SHUFFLE: int = 100
EMBEDS_KEYS_READ: Tuple[str, ...] = ("embeds.pth", "mask.pth", "tokens.pth")
EMBEDS_KEYS_OUT: Tuple[str, ...] = ("embeds", "attention_mask", "input_ids")
