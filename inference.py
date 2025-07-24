from vlac import VLAC, VLACConfig
from vlac.dataset.dataset import getDataset
from vlac.utils.memory_utils import print_memory_usage_for_model


def load_from_json():
    config = VLACConfig.from_json_file("configs/config.json")
    return VLAC(config)


def main():
    vlac = VLAC.from_pretrained('/media/hdd/maxime_carlier/checkpoints/vlac/train-vlac/checkpoint-116224').to("cuda:1")
    print_memory_usage_for_model(vlac)
    dataset = getDataset(
        "minerl",
        img_preprocess=vlac.vision_tower.image_processor,
        tokenizer=vlac.text_tokenizer,
        history_len=4,
        use_prompt_format=True
    )
    data = dataset[0]
    out_prompt, out_vision = vlac(data["prompt_in"], data["imgs"][:-1])
    print(out_prompt)
    print(out_vision)


if __name__ == "__main__":
    main()
