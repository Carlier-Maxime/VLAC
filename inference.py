import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model(device="cuda"):
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True
    ).to(device)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_beams=2
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    model, tokenizer = setup_model()
    prompt = """You are a PNJ in a video game. Player talk you : "Can you give me a creative stuff ?". """
    response = generate_response(model, tokenizer, prompt)
    print("Generated response:", response)


if __name__ == "__main__":
    main()
