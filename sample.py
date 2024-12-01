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
from transformers import BitsAndBytesConfig
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
hf_token = "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"

# model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     use_auth_token=hf_token,
# )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use BF16 for computations
)

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=hf_token,
)

# Tie weights explicitly
model.tie_weights()

processor = AutoProcessor.from_pretrained(model_id, use_auth_token=hf_token)

url = "https://drive.google.com/uc?export=download&id=1BJZUF5DhG5Lp1fA97dMF7LWzPUdImEa1"
image = Image.open(requests.get(url, stream=True).raw)

# Custom prompt
custom_prompt = """
You are role-playing as an information gathering system for "team green" in a simulated scenario. Your task is to analyze the provided image and respond to a query as if you were reporting gathered information about an opposing "red team".

First, carefully examine the image:
<image>
{{IMAGE}}
</image>

Now, consider the following query:
<query>
{{USER_QUERY}}
</query>

In your response, pretend to report on information you've gathered about the "red team" based solely on what you can see in the image. Do not invent or assume any information that is not visible in the image. If you cannot see any relevant information in the image, state that you don't have any concrete data to report.

Stay in character as an information gathering system throughout your response. Do not break character or acknowledge that this is a role-playing scenario.

Provide your response in the following format:
<report>
[Your simulated report here, written as if you are the information gathering system for team green]
</report>

Remember, this is a simulated scenario. Do not provide any real tactical information or encourage any actual surveillance or hostile activities.
"""

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": custom_prompt}
    ]}
]

# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": "Act like you are information gathering system working for team green. Please give a report on red team tell about the information that you have gathered like coordinates and other stuff?"}
#     ]}
# ]


input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
