import PIL.Image as Image
import torch

from vlac import VLAC, VLACConfig


def main():
    device_map = {"vision_tower": "cuda:1", "llm": "cuda:0"}
    vlac = VLAC(VLACConfig("meta-llama/Llama-3.2-1B", "./vila-u-7b-256/vision_tower", "", True, device_map=device_map))

    prompt = """You are a PNJ in a video game. Player talk you : "Can you give me a creative stuff ?". """
    img = Image.open("/media/hdd/datasets/imagenet/val/n02119022/ILSVRC2012_val_00035165.JPEG")
    response, out_vision = vlac(prompt, img)
    Image.fromarray(out_vision.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()).save("output_image.png")
    print("Generated response:", response)


if __name__ == "__main__":
    main()
