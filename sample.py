import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
hf_token = "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"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=hf_token,
)
processor = AutoProcessor.from_pretrained(model_id, use_auth_token=hf_token)

url = "https://drive.google.com/uc?export=download&id=1uTJpO8TNaW3lsI-pqX1GSDgwm80RpOrz"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Please provide detailed information about the ships, including coordinates, size, and other relevant data. Additionally, indicate where a shot has been taken (marked as a yellow spot)."}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
