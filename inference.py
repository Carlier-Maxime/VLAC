import PIL.Image as Image
import torch

from vlac import VLAC, VLACConfig


def load_from_json():
    device_map = {"vision_tower": "cuda:1", "mm_projector": "cuda:1", "llm": "cuda:1"}
    config = VLACConfig.from_json_file("configs/config.json")
    config.device_map = device_map
    return VLAC(config)


def load_pretrain():
    vlac = VLAC.from_pretrained("checkpoints/vlac-train-vision_tower/checkpoint-64")
    vlac.vision_tower.to(vlac.config.device_map["vision_tower"])
    return vlac


def main():
    vlac = load_pretrain()
    prompt = """You are a PNJ in a video game. Player talk you : "Can you give me a creative stuff ?". """
    img = Image.open("/media/hdd/datasets/imagenet/val/n02119022/ILSVRC2012_val_00035165.JPEG")
    out_vision = vlac.encode_decode_images(img)[0].to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)
    Image.fromarray(out_vision.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()).save("output_image.png")
    # print("Generated response:", response)


if __name__ == "__main__":
    main()
