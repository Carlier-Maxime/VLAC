from vlac import VLAC, VLACConfig


def main():
    device = "cuda:0"
    vlac = VLAC(VLACConfig("meta-llama/Llama-3.2-3B", "./vila-u-7b-256/vision_tower", "", True, device_map={"": device}))

    prompt = """You are a PNJ in a video game. Player talk you : "Can you give me a creative stuff ?". """
    print("Generated response:", vlac(prompt))


if __name__ == "__main__":
    main()
