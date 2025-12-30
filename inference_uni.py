from ThinkGen.model import ThinkGen_Chat
import os

model_path = "JSYuuu/ThinkGen"

chat_model = ThinkGen_Chat(
    model_path=model_path,
    dtype='bf16',
    height=1024,
    width=1024
)


## Gen
messages = [
    {"type": "text", "value": '''A young woman wearing a straw hat, standing in a golden wheat field.'''}
]
results = chat_model.generate_image(messages)
output_dir = "vis/chat"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(results.images):
    save_path = os.path.join(output_dir, f"result_{i}.png")
    img.save(save_path)
    print(f"Saved to {save_path}")


## Gen-Think
messages = [
    {"type": "text", "value": '''A young woman wearing a straw hat, standing in a golden wheat field.'''}
]
results = chat_model.generate_image(messages, think=True)
output_dir = "vis/chat"
os.makedirs(output_dir, exist_ok=True)

print(f"cot & rewrite prompt: \n{results.prompt_cot}")

for i, img in enumerate(results.images):
    save_path = os.path.join(output_dir, f"result_think_{i}.png")
    img.save(save_path)
    print(f"Saved to {save_path}")


## Und
messages = [
    {"type": "image", "value": "images/teaser.png"},
    {"type": "text", "value": "Describe this image"}
]

response = chat_model.generate_text(messages)
print(response)


