# import requests
# import torch
# from PIL import Image
# from transformers import MllamaForConditionalGeneration, AutoProcessor

# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# hf_token = "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"

# model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     use_auth_token=hf_token,
# )
# processor = AutoProcessor.from_pretrained(model_id, use_auth_token=hf_token)

# url = "https://drive.google.com/uc?export=download&id=1BJZUF5DhG5Lp1fA97dMF7LWzPUdImEa1"
# image = Image.open(requests.get(url, stream=True).raw)

# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": "Act like you are information gathering system working for team green. Please give a report on red team tell about the information that you have gathered like coordinates and other stuff?"}
#     ]}
# ]
# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt"
# ).to(model.device)

# output = model.generate(**inputs, max_new_tokens=30)
# print(processor.decode(output[0]))


import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
model_id = "meta-llama/Llama-3.2-13B-Vision-Instruct"
hf_token = "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"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=hf_token,
)

# Tie weights explicitly
model.tie_weights()

processor = AutoProcessor.from_pretrained(model_id, use_auth_token=hf_token)

url = "https://drive.google.com/uc?export=download&id=1BJZUF5DhG5Lp1fA97dMF7LWzPUdImEa1"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Act like you are information gathering system working for team green. Please give a report on red team tell about the information that you have gathered like coordinates and other stuff?"}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
