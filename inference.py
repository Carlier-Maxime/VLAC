import PIL.Image as Image
import torch

from vlac import VLAC, VLACConfig


def main():
    device_map = {"vision_tower": "cuda:1", "mm_projector": "cuda:0", "llm": "cuda:0"}
    config = VLACConfig.from_json_file("./config.json")
    config.device_map = device_map
    vlac = VLAC(config)

    prompt = """You are a PNJ in a video game. Player talk you : "Can you give me a creative stuff ?". """
    img = Image.open("/media/hdd/datasets/imagenet/val/n02119022/ILSVRC2012_val_00035165.JPEG")
    response, out_vision = vlac(prompt, img)
    Image.fromarray(out_vision.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()).save("output_image.png")
    print("Generated response:", response)


if __name__ == "__main__":
    main()
