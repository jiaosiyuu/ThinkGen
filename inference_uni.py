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
    {"type": "text", "value": '''A close-up image of a red apple with the words 'Tart & Sweet' in white, cursive font on its surface, forming a spiral pattern. The apple is centered in the frame, and the background is a green surface labeled 'Organic Produce' in black, bold letters. The apple has a visible stem and a small bite mark on its side with the word 'Juicy' written in a small, handwritten style near the bite.'''}
]
results = chat_model.generate_image(messages)
output_dir = "vis/chat"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(results.images):
    save_path = os.path.join(output_dir, f"result_{i}.png")
    img.save(save_path)
    print(f"Saved to {save_path}")



## Und
messages = [
    {"type": "image", "value": "images/teaser.png"},
    {"type": "text", "value": "Describe this image"}
]

response = chat_model.generate_text(messages)
print(response)


